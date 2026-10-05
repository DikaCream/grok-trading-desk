"""Regime gate for new memecoin entries — pure code.

Memecoins are a high-beta derivative of SOL and of risk appetite. When the
crypto pulse is lukewarm or SOL itself is dumping, the same setup that works in
a hot tape bleeds out. The gate turns the pulse read (and, when available, a
measured SOL price change) into one of four states:

  normal     go_signal >= weak_go_signal and SOL not dumping
             → no change
  caution    weak go_signal (min_go_signal <= go < weak_go_signal) OR neutral regime
             → min_score += caution.min_score_add, size *= caution.size_mult
  defensive  SOL dump regime (sol_trend == "down" or measured SOL drop beyond
             sol_dump_pct) OR risk_off regime that still clears min_go_signal
             → min_score += defensive.min_score_add, size *= defensive.size_mult
  paused     go_signal < min_go_signal (reason veto_market_paused), or
             weak AND dumping together with pause_on_weak_and_dump
             → no new entries (reason regime_paused)

Exits are never gated: the regime only governs new risk.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "weak_go_signal": 0.50,
    "sol_dump_pct": -0.04,            # measured SOL change over sol_lookback_minutes
    "sol_lookback_minutes": 60,
    "pause_on_weak_and_dump": True,
    "caution": {"min_score_add": 0.05, "size_mult": 0.60},
    "defensive": {"min_score_add": 0.10, "size_mult": 0.40},
}


@dataclass
class RegimeDecision:
    state: str = "normal"
    allow_entries: bool = True
    min_score_add: float = 0.0
    size_mult: float = 1.0
    reason: str = "regime_normal"
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RegimeGate:
    """Assess a pulse dict (+ optional measured SOL move) into a RegimeDecision."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        cfg = dict(DEFAULTS)
        for key, value in (config.get("regime", {}) or {}).items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.min_go_signal = float((config.get("pulse", {}) or {}).get("min_go_signal", 0.3))
        self._sol: deque[tuple[float, float]] = deque(maxlen=2048)
        self.last: RegimeDecision | None = None

    # -- SOL price feed (optional) ------------------------------------------------

    def observe_sol_price(self, price: float, ts: float | None = None) -> None:
        if price and price > 0:
            self._sol.append((ts if ts is not None else time.time(), float(price)))

    def sol_change_pct(self, now: float | None = None) -> float | None:
        """Change from the oldest observation inside the lookback window to the latest."""
        if len(self._sol) < 2:
            return None
        now = now if now is not None else time.time()
        horizon = now - float(self.cfg["sol_lookback_minutes"]) * 60.0
        window = [(t, p) for t, p in self._sol if t >= horizon]
        if len(window) < 2:
            return None
        first, last = window[0][1], window[-1][1]
        return (last - first) / first if first > 0 else None

    # -- decision -----------------------------------------------------------------

    def assess(
        self,
        pulse: dict[str, Any] | None,
        sol_change_pct: float | None = None,
        now: float | None = None,
    ) -> RegimeDecision:
        pulse = pulse or {}
        go = float(pulse.get("go_signal", 0.0) or 0.0)
        regime = str(pulse.get("regime", "unknown")).lower()
        trend = str(pulse.get("sol_trend", "flat")).lower()
        if sol_change_pct is None:
            measured = pulse.get("sol_change_pct")
            sol_change_pct = float(measured) if measured is not None else self.sol_change_pct(now)

        weak = go < float(self.cfg["weak_go_signal"]) or regime == "neutral"
        dumping = trend == "down" or (
            sol_change_pct is not None and sol_change_pct <= float(self.cfg["sol_dump_pct"])
        )
        signals = {
            "go_signal": go,
            "regime": regime,
            "sol_trend": trend,
            "sol_change_pct": sol_change_pct,
            "weak": weak,
            "sol_dump": dumping,
        }

        if go < self.min_go_signal:
            decision = RegimeDecision("paused", False, 0.0, 0.0, "veto_market_paused", signals)
        elif not self.enabled:
            decision = RegimeDecision("normal", True, 0.0, 1.0, "regime_disabled", signals)
        elif weak and dumping and self.cfg.get("pause_on_weak_and_dump", True):
            decision = RegimeDecision("paused", False, 0.0, 0.0, "regime_paused", signals)
        elif dumping or regime == "risk_off":
            d = self.cfg["defensive"]
            decision = RegimeDecision(
                "defensive", True, float(d["min_score_add"]), float(d["size_mult"]),
                "regime_sol_dump" if dumping else "regime_risk_off", signals,
            )
        elif weak:
            c = self.cfg["caution"]
            decision = RegimeDecision(
                "caution", True, float(c["min_score_add"]), float(c["size_mult"]),
                "regime_weak_pulse", signals,
            )
        else:
            decision = RegimeDecision("normal", True, 0.0, 1.0, "regime_normal", signals)
        self.last = decision
        return decision
