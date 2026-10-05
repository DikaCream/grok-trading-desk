"""Toxic flow filter — deployer and metadata-clone patterns. Pure code.

Most pump.fun losses are not bad luck; they are the same few operators running
the same play on repeat. The scout sees every create event, so it can remember
who launched what and how it was dressed, and reject the patterns that only
exist to farm buyers:

  toxic_deployer_blacklisted  the creator wallet was blacklisted (one of its
                              tokens closed on an emergency dump / dev dump, or
                              a manual entry), persisted as `blacklist` log records
  toxic_serial_deployer       the creator launched more than
                              `max_launches_per_deployer` tokens inside
                              `deployer_window_minutes` (launch farm)
  toxic_uri_reuse             the exact metadata URI was already used by a
                              different mint (copy-paste farm)
  toxic_clone_same_deployer   the same deployer re-launched an identical
                              normalized name+symbol (relaunch-after-rug)
  toxic_metadata_clone        `max_clones` or more different mints already carried
                              this normalized name+symbol inside
                              `clone_window_minutes`

Memory is bounded and pruned by time.
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any

from ..models import Token

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "deployer_window_minutes": 1440,
    "max_launches_per_deployer": 2,
    "clone_window_minutes": 180,
    "max_clones": 3,
    "reject_uri_reuse": True,
    "reject_same_deployer_clone": True,
    # Close reasons that blacklist the token's deployer.
    "blacklist_on": ["emergency_dev_dump", "emergency_dump", "emergency_crash"],
    "max_tracked": 50_000,
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, strip everything but a-z0-9, fold common lookalike digits."""
    s = _NON_ALNUM.sub("", (text or "").lower())
    return s.translate(str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}))


def fingerprint(token: Token) -> str:
    name, symbol = normalize(token.name), normalize(token.symbol)
    if not name and not symbol:
        return ""
    return f"{name}|{symbol}"


class ToxicFlowFilter:
    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = {**DEFAULTS, **(((config or {}).get("crypto_toxic", {})) or {})}
        self.enabled = bool(self.cfg.get("enabled", True))
        self._deployer: dict[str, deque[tuple[float, str]]] = {}
        self._clones: dict[str, deque[tuple[float, str, str]]] = {}
        self._uris: dict[str, str] = {}
        self.blacklist: dict[str, str] = {}

    # -- persistence ----------------------------------------------------------------

    def seed_from_log(self, records: list[dict[str, Any]]) -> int:
        """Restore the blacklist from `blacklist` records in the event log."""
        added = 0
        for record in records or []:
            if record.get("type") == "blacklist" and record.get("creator"):
                self.blacklist[str(record["creator"])] = str(record.get("reason", "logged"))
                added += 1
        return added

    def should_blacklist(self, reason: str) -> bool:
        return reason in set(self.cfg.get("blacklist_on") or [])

    def add_blacklist(self, creator: str, reason: str) -> bool:
        if not creator or creator in self.blacklist:
            return False
        self.blacklist[creator] = reason
        return True

    # -- observe + judge ----------------------------------------------------------------

    def _prune(self, now: float) -> None:
        dep_h = now - float(self.cfg["deployer_window_minutes"]) * 60.0
        clone_h = now - float(self.cfg["clone_window_minutes"]) * 60.0
        for store, horizon in ((self._deployer, dep_h), (self._clones, clone_h)):
            for key in list(store):
                buf = store[key]
                while buf and buf[0][0] < horizon:
                    buf.popleft()
                if not buf:
                    del store[key]
        if len(self._uris) > int(self.cfg["max_tracked"]):
            # oldest-first dict order: drop the first half
            for key in list(self._uris)[: len(self._uris) // 2]:
                del self._uris[key]

    def observe(self, token: Token, now: float | None = None) -> None:
        """Record a create event. Call for every create, filtered or not."""
        if not token.mint:
            return
        now = now if now is not None else time.time()
        if token.creator:
            self._deployer.setdefault(token.creator, deque(maxlen=256)).append((now, token.mint))
        fp = fingerprint(token)
        if fp:
            self._clones.setdefault(fp, deque(maxlen=256)).append((now, token.mint, token.creator))
        if token.uri and token.uri not in self._uris:
            self._uris[token.uri] = token.mint
        self._prune(now)

    def reason(self, token: Token, now: float | None = None) -> str | None:
        """Judge a token that has already been observed. None = clean."""
        if not self.enabled or not token.mint:
            return None
        if token.creator and token.creator in self.blacklist:
            return "toxic_deployer_blacklisted"

        now = now if now is not None else time.time()
        if token.creator:
            launches = {m for _, m in self._deployer.get(token.creator, ())}
            launches.add(token.mint)
            if len(launches) > int(self.cfg["max_launches_per_deployer"]):
                return "toxic_serial_deployer"

        if self.cfg.get("reject_uri_reuse", True) and token.uri:
            first = self._uris.get(token.uri)
            if first and first != token.mint:
                return "toxic_uri_reuse"

        fp = fingerprint(token)
        if fp:
            prior = [(m, c) for _, m, c in self._clones.get(fp, ()) if m != token.mint]
            if (
                self.cfg.get("reject_same_deployer_clone", True)
                and token.creator
                and any(c == token.creator for _, c in prior)
            ):
                return "toxic_clone_same_deployer"
            if len({m for m, _ in prior}) >= int(self.cfg["max_clones"]):
                return "toxic_metadata_clone"
        return None

    def entry_reason(self, token: Token) -> str | None:
        """Desk-side last check before paying for models: blacklist only."""
        if self.enabled and token.creator and token.creator in self.blacklist:
            return "toxic_deployer_blacklisted"
        return None
