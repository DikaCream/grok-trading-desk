"""Post-entry trade-flow tracker for open memecoin positions.

The scout's watch window ends when the token is judged; after entry the desk
still needs a live read of the tape to (a) confirm a staged add and (b) detect a
dump early. `FlowTracker` keeps a bounded rolling buffer of trade events per
tracked mint and answers window questions:

  stats(mint, seconds)        buys / sells / volumes / price high-low-last in window
  stats_since(mint, since_ts) same, since an absolute timestamp (e.g. entry time)

Pure Python, timestamps injectable for tests. Fed by `Scout.handle_trade`.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FlowStats:
    buys: int = 0
    sells: int = 0
    buy_volume_sol: float = 0.0
    sell_volume_sol: float = 0.0
    unique_buyers: int = 0
    high: float = 0.0
    low: float = 0.0
    first: float = 0.0
    last: float = 0.0
    dev_sold: bool = False
    seconds: float = 0.0

    @property
    def trades(self) -> int:
        return self.buys + self.sells

    @property
    def buy_sell_ratio(self) -> float:
        if self.sells <= 0:
            return float(self.buys) if self.buys else 0.0
        return self.buys / self.sells

    @property
    def drop_from_high(self) -> float:
        """Fractional drop of `last` below the window high (0..1)."""
        if self.high <= 0 or self.last <= 0:
            return 0.0
        return max(0.0, (self.high - self.last) / self.high)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["buy_sell_ratio"] = round(self.buy_sell_ratio, 3)
        d["drop_from_high"] = round(self.drop_from_high, 4)
        return d


@dataclass
class _Trade:
    ts: float
    side: str
    sol: float
    trader: str
    price: float


class FlowTracker:
    """Rolling per-mint trade buffer. Bounded so a hot mint cannot eat memory."""

    def __init__(self, max_events_per_mint: int = 5000, retention_seconds: float = 7200.0):
        self.max_events = int(max_events_per_mint)
        self.retention = float(retention_seconds)
        self._events: dict[str, deque[_Trade]] = {}
        self._creators: dict[str, str] = {}
        self._reserves: dict[str, tuple[float, float]] = {}

    # -- membership -----------------------------------------------------------------

    def track(self, mint: str, creator: str = "") -> None:
        if not mint:
            return
        self._events.setdefault(mint, deque(maxlen=self.max_events))
        if creator:
            self._creators[mint] = creator

    def untrack(self, mint: str) -> None:
        self._events.pop(mint, None)
        self._creators.pop(mint, None)
        self._reserves.pop(mint, None)

    def is_tracked(self, mint: str) -> bool:
        return mint in self._events

    @property
    def tracked(self) -> list[str]:
        return list(self._events)

    # -- ingest -----------------------------------------------------------------------

    def record(self, event: dict[str, Any], now: float | None = None) -> bool:
        """Fold one PumpPortal trade frame. Returns True if it was for a tracked mint."""
        mint = str(event.get("mint", ""))
        buf = self._events.get(mint)
        if buf is None:
            return False
        side = str(event.get("txType", "")).lower()
        if side not in {"buy", "sell"}:
            return False
        ts = float(now if now is not None else event.get("ts") or time.time())
        try:
            sol = abs(float(event.get("solAmount", 0) or 0))
        except (TypeError, ValueError):
            sol = 0.0
        price = 0.0
        v_sol, v_tok = event.get("vSolInBondingCurve"), event.get("vTokensInBondingCurve")
        try:
            if v_sol and v_tok:
                price = float(v_sol) / float(v_tok)
                self._reserves[mint] = (float(v_sol), float(v_tok))
        except (TypeError, ValueError, ZeroDivisionError):
            price = 0.0
        buf.append(_Trade(ts, side, sol, str(event.get("traderPublicKey", "")), price))
        # Retention prune from the left (deque is time-ordered).
        horizon = ts - self.retention
        while buf and buf[0].ts < horizon:
            buf.popleft()
        return True

    # -- queries ----------------------------------------------------------------------

    def last_price(self, mint: str) -> float:
        buf = self._events.get(mint)
        if not buf:
            return 0.0
        for trade in reversed(buf):
            if trade.price > 0:
                return trade.price
        return 0.0

    def reserves(self, mint: str) -> tuple[float, float] | None:
        return self._reserves.get(mint)

    def stats_since(self, mint: str, since_ts: float, now: float | None = None) -> FlowStats:
        now = now if now is not None else time.time()
        out = FlowStats(seconds=max(0.0, now - since_ts))
        buf = self._events.get(mint)
        if not buf:
            return out
        creator = self._creators.get(mint, "")
        buyers: set[str] = set()
        for trade in buf:
            if trade.ts < since_ts or trade.ts > now:
                continue
            if trade.side == "buy":
                out.buys += 1
                out.buy_volume_sol += trade.sol
                if trade.trader:
                    buyers.add(trade.trader)
            else:
                out.sells += 1
                out.sell_volume_sol += trade.sol
                if creator and trade.trader == creator:
                    out.dev_sold = True
            if trade.price > 0:
                if out.first <= 0:
                    out.first = trade.price
                out.high = max(out.high, trade.price)
                out.low = trade.price if out.low <= 0 else min(out.low, trade.price)
                out.last = trade.price
        out.unique_buyers = len(buyers)
        out.buy_volume_sol = round(out.buy_volume_sol, 6)
        out.sell_volume_sol = round(out.sell_volume_sol, 6)
        return out

    def stats(self, mint: str, seconds: float, now: float | None = None) -> FlowStats:
        now = now if now is not None else time.time()
        return self.stats_since(mint, now - float(seconds), now)
