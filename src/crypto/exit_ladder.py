"""Exit ladder for memecoin positions — pure code.

Evaluation order on every mark (first match wins):

  1. emergency dump      dev sold / flow flip + crash        → CLOSE (emergency_*)
  2. hard stop           price <= stop_price                  → CLOSE (stop_loss, or
                                                                breakeven_stop once the
                                                                ladder raised the stop)
  3. catastrophic crash  >= catastrophic_pct off window high  → CLOSE (emergency_crash)
  4. runner cap          price >= take_profit_price           → CLOSE (take_profit)
  5. ladder rungs        +15% sell 33%, +30% sell 33% of base → TRIM (tp_rung_N)
                         (gaps through several rungs sell them together)
  6. runner trail        after rung 1, trail the rest         → CLOSE (runner_trail)
     legacy trailing     positions without ladder state       → CLOSE (trailing_stop)
  7. stale exit          no +stale_min_gain_pct within        → CLOSE (stale_exit)
                         stale_minutes and no rung hit
  8. time-stop           hold >= max_hold_minutes             → CLOSE (time_stop)

Ladder state lives in `position.meta["ladder"]` and is only created for
positions the desk opened with the ladder enabled; older positions keep the
legacy single TP/SL/trailing plan. `evaluate` never mutates state except
`peak_price`; the desk calls `apply_trim` after a successful partial sell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from ..models import Market, Position
from .dump_detector import detect_catastrophic, detect_dump, dump_config
from .flow import FlowStats

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "rungs": [
        {"gain_pct": 0.15, "sell_fraction": 0.33},
        {"gain_pct": 0.30, "sell_fraction": 0.33},
    ],
    "breakeven_after_rung": 1,
    "breakeven_buffer_pct": 0.01,
    "runner_trail_pct": 0.12,
    "runner_take_profit_pct": 1.50,
    "stale_minutes": 20,
    "stale_min_gain_pct": 0.05,
    "dust_fraction": 0.05,
}


@dataclass
class ExitSignal:
    action: str = "HOLD"           # HOLD | TRIM | CLOSE
    reason: str = ""
    fraction: float = 0.0          # of CURRENT quantity (1.0 for CLOSE)
    rungs: tuple[int, ...] = ()
    new_stop: float | None = None
    emergency: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rungs"] = list(self.rungs)
        return d


def ladder_config(config: dict[str, Any] | None) -> dict[str, Any]:
    exits = (config or {}).get("exits", {}) or {}
    return {**DEFAULTS, **(exits.get("ladder", {}) or {})}


def _hold_minutes(position: Position, now: datetime | None) -> float:
    if now is None:
        return position.hold_time_minutes
    opened = position.opened_at
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    ref = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return (ref - opened).total_seconds() / 60.0


class ExitLadder:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.exits = config.get("exits", {}) or {}
        self.cfg = ladder_config(config)
        self.dump_cfg = dump_config(config)
        self.enabled = bool(self.cfg.get("enabled", True))
        self.rungs = sorted(
            ({"gain_pct": float(r["gain_pct"]), "sell_fraction": float(r["sell_fraction"])}
             for r in self.cfg.get("rungs") or []),
            key=lambda r: r["gain_pct"],
        )

    # -- state ----------------------------------------------------------------------

    def init_state(self, base_qty: float) -> dict[str, Any]:
        return {"base_qty": float(base_qty), "rungs_hit": [], "stop_raised": False}

    def runner_take_profit(self, entry_price: float) -> float | None:
        cap = self.cfg.get("runner_take_profit_pct")
        if entry_price <= 0 or cap is None:
            return None
        return entry_price * (1.0 + float(cap))

    def apply_trim(self, position: Position, signal: ExitSignal) -> None:
        """Record rungs as hit and raise the stop. Call after the sell filled."""
        state = position.meta.get("ladder")
        if not isinstance(state, dict):
            return
        hit = set(state.get("rungs_hit", []))
        hit.update(signal.rungs)
        state["rungs_hit"] = sorted(hit)
        if signal.new_stop is not None:
            current = float(position.stop_price or 0.0)
            if signal.new_stop > current:
                position.stop_price = signal.new_stop
                state["stop_raised"] = True

    # -- evaluation ---------------------------------------------------------------------

    def evaluate(
        self,
        position: Position,
        price: float | None,
        flow: FlowStats | None = None,
        now: datetime | None = None,
    ) -> ExitSignal:
        if position.market != Market.CRYPTO:
            return ExitSignal()
        px = float(price) if price is not None else float(position.current_price or 0.0)
        entry = float(position.entry_price or 0.0)
        state = position.meta.get("ladder") if self.enabled else None
        if not isinstance(state, dict):
            state = None

        if px > 0:
            peak = float(position.peak_price or entry or 0.0)
            if px > peak:
                peak = px
            position.peak_price = peak if peak > 0 else px
        peak = float(position.peak_price or 0.0)

        if px > 0:
            # 1. emergency dump (dev sold, or flow flip + crash)
            reason = detect_dump(flow, px, peak, self.dump_cfg)
            if reason:
                return ExitSignal("CLOSE", reason, 1.0, emergency=True)

            # 2. hard / protected stop
            if position.stop_price is not None and px <= float(position.stop_price):
                raised = bool(state and state.get("stop_raised"))
                return ExitSignal("CLOSE", "breakeven_stop" if raised else "stop_loss", 1.0)

            # 3. catastrophic crash
            reason = detect_catastrophic(flow, px, peak, self.dump_cfg)
            if reason:
                return ExitSignal("CLOSE", reason, 1.0, emergency=True)

            # 4. runner cap / legacy full take-profit
            if position.take_profit_price is not None and px >= float(position.take_profit_price):
                return ExitSignal("CLOSE", "take_profit", 1.0)

            gain = (px - entry) / entry if entry > 0 else 0.0
            peak_gain = (peak - entry) / entry if entry > 0 else 0.0

            # 5. ladder rungs
            if state is not None and entry > 0:
                signal = self._rungs(position, state, gain, entry)
                if signal is not None:
                    return signal

            # 6. trailing
            if state is not None:
                if state.get("rungs_hit") and peak > 0:
                    trail = float(self.cfg["runner_trail_pct"])
                    if px <= peak * (1.0 - trail):
                        return ExitSignal("CLOSE", "runner_trail", 1.0)
            else:
                activate = self.exits.get("trailing_stop_activate_pct")
                trail = self.exits.get("trailing_stop_pct")
                if activate is not None and trail is not None and peak_gain >= float(activate):
                    if px <= peak * (1.0 - float(trail)):
                        return ExitSignal("CLOSE", "trailing_stop", 1.0)

            # 7. stale exit (opportunity cost) — ladder positions only
            if state is not None and not state.get("rungs_hit"):
                stale = self.cfg.get("stale_minutes")
                if stale is not None and _hold_minutes(position, now) >= float(stale):
                    if peak_gain < float(self.cfg["stale_min_gain_pct"]):
                        return ExitSignal("CLOSE", "stale_exit", 1.0)

        # 8. time-stop (checked even without a mark)
        max_hold = self.exits.get("max_hold_minutes")
        if max_hold is not None and _hold_minutes(position, now) >= float(max_hold):
            return ExitSignal("CLOSE", "time_stop", 1.0)
        return ExitSignal()

    def _rungs(
        self, position: Position, state: dict[str, Any], gain: float, entry: float
    ) -> ExitSignal | None:
        hit = set(state.get("rungs_hit", []))
        crossed = [
            i for i, rung in enumerate(self.rungs)
            if i not in hit and gain >= rung["gain_pct"]
        ]
        if not crossed:
            return None
        base = float(state.get("base_qty") or position.quantity or 0.0)
        qty = float(position.quantity or 0.0)
        sell_qty = base * sum(self.rungs[i]["sell_fraction"] for i in crossed)
        fraction = min(1.0, sell_qty / qty) if qty > 0 else 1.0
        remaining = qty - qty * fraction
        new_stop = None
        top = max(crossed) + 1
        if top >= int(self.cfg.get("breakeven_after_rung", 1)):
            new_stop = entry * (1.0 + float(self.cfg.get("breakeven_buffer_pct", 0.0)))
        reason = "tp_rung_" + "_".join(str(i + 1) for i in crossed)
        if base > 0 and remaining <= base * float(self.cfg.get("dust_fraction", 0.05)):
            return ExitSignal("CLOSE", reason, 1.0, tuple(crossed), new_stop)
        return ExitSignal("TRIM", reason, round(fraction, 6), tuple(crossed), new_stop)
