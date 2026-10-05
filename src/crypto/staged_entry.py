"""Staged (scale-in) entry — pure code.

A memecoin entry is a bet that the tape keeps going. Paying full size before
the tape confirms is how a desk donates to the snipers it was trying to avoid.
So every entry is split:

  probe   `initial_fraction` (40%) of the planned size, filled immediately
  add     the remaining `1 - initial_fraction` (60%), filled only if, within
          `confirm_window_minutes` (Y) of the probe:
            * price is >= +confirm_gain_pct (X) over the probe fill, and
            * price is <= +max_gain_for_add_pct (don't chase a vertical candle;
              a pullback into the band can still add), and
            * post-entry flow confirms: >= min_buys buys, buy/sell ratio
              >= min_buy_sell_ratio, >= min_unique_buyers distinct buyers, and
            * the deployer has not sold, and no exit-ladder rung has fired.
          With `require_flow` (default) no flow data means no add — never blind.
          After the window, the add expires and the position stays probe-only.

State lives in `position.meta["staged"]`. The desk executes the add through the
executor (paper fills simulated at the quote price) and calls `apply_add`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..models import Position
from .flow import FlowStats

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "initial_fraction": 0.40,
    "confirm_gain_pct": 0.08,
    "max_gain_for_add_pct": 0.30,
    "confirm_window_minutes": 10,
    "min_buys": 8,
    "min_buy_sell_ratio": 1.3,
    "min_unique_buyers": 5,
    "require_flow": True,
}


@dataclass
class AddDecision:
    action: str   # add | wait | expire | skip
    reason: str
    gain: float = 0.0


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class StagedEntry:
    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = {**DEFAULTS, **(((config or {}).get("staged_entry", {})) or {})}
        self.enabled = bool(self.cfg.get("enabled", True))

    def split(self, planned_usd: float) -> tuple[float, float]:
        """(probe_usd, add_usd). Disabled → everything in the probe."""
        planned = max(0.0, float(planned_usd))
        if not self.enabled:
            return round(planned, 2), 0.0
        frac = max(0.0, min(1.0, float(self.cfg["initial_fraction"])))
        probe = round(planned * frac, 2)
        return probe, round(planned - probe, 2)

    def init_state(
        self, planned_usd: float, probe_usd: float, add_usd: float, probe_price: float,
        opened_at: datetime,
    ) -> dict[str, Any]:
        return {
            "stage": "probe" if (self.enabled and add_usd > 0) else "full",
            "planned_usd": round(float(planned_usd), 2),
            "probe_usd": round(float(probe_usd), 2),
            "add_usd": round(float(add_usd), 2),
            "probe_price": float(probe_price),
            "opened_at": _aware(opened_at).isoformat(),
        }

    def evaluate_add(
        self,
        position: Position,
        price: float,
        flow: FlowStats | None,
        now: datetime | None = None,
    ) -> AddDecision:
        state = position.meta.get("staged")
        if not self.enabled or not isinstance(state, dict) or state.get("stage") != "probe":
            return AddDecision("skip", "not_staged")
        now = _aware(now or datetime.now(timezone.utc))
        opened = _aware(position.opened_at)
        elapsed_min = (now - opened).total_seconds() / 60.0
        if elapsed_min > float(self.cfg["confirm_window_minutes"]):
            return AddDecision("expire", "add_window_expired")

        ladder = position.meta.get("ladder")
        if isinstance(ladder, dict) and ladder.get("rungs_hit"):
            return AddDecision("expire", "ladder_already_trimming")
        if flow is not None and flow.dev_sold:
            return AddDecision("expire", "dev_sold")

        probe_price = float(state.get("probe_price") or position.entry_price or 0.0)
        px = float(price or 0.0)
        if probe_price <= 0 or px <= 0:
            return AddDecision("wait", "no_price")
        gain = (px - probe_price) / probe_price
        if gain < float(self.cfg["confirm_gain_pct"]):
            return AddDecision("wait", "awaiting_gain", gain)
        if gain > float(self.cfg["max_gain_for_add_pct"]):
            return AddDecision("wait", "add_chase_guard", gain)

        if flow is None or flow.trades == 0:
            if self.cfg.get("require_flow", True):
                return AddDecision("wait", "awaiting_volume", gain)
            return AddDecision("add", "confirmed_price_only", gain)
        if flow.buys < int(self.cfg["min_buys"]):
            return AddDecision("wait", "awaiting_volume", gain)
        if flow.buy_sell_ratio < float(self.cfg["min_buy_sell_ratio"]):
            return AddDecision("wait", "weak_flow", gain)
        if flow.unique_buyers < int(self.cfg["min_unique_buyers"]):
            return AddDecision("wait", "too_few_buyers", gain)
        return AddDecision("add", "confirmed", gain)

    @staticmethod
    def apply_add(
        position: Position, add_usd: float, fill_price: float, now: datetime | None = None
    ) -> None:
        """Fold an add fill into the position (desk units: qty = usd / price)."""
        if fill_price <= 0 or add_usd <= 0:
            return
        add_qty = add_usd / fill_price
        new_qty = float(position.quantity) + add_qty
        new_amount = float(position.amount_usd) + add_usd
        if new_qty > 0:
            position.entry_price = new_amount / new_qty
        position.quantity = new_qty
        position.amount_usd = round(new_amount, 6)
        state = position.meta.setdefault("staged", {})
        state["stage"] = "full"
        state["add_price"] = float(fill_price)
        state["added_usd"] = round(float(add_usd), 2)
        state["added_at"] = _aware(now or datetime.now(timezone.utc)).isoformat()
        ladder = position.meta.get("ladder")
        if isinstance(ladder, dict):
            ladder["base_qty"] = new_qty

    @staticmethod
    def expire(position: Position, reason: str) -> None:
        state = position.meta.get("staged")
        if isinstance(state, dict) and state.get("stage") == "probe":
            state["stage"] = "probe_only"
            state["expired_reason"] = reason
