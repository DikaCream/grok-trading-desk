"""Dump detection — the emergency exit. Pure code.

A memecoin rug or coordinated dump has a recognisable signature on the tape: the
order flow flips from buys to sells *and* price falls hard inside a short window.
Either alone is noise (a big seller into strong bids; a thin-book wick). Both
together mean exit now, at market, ahead of every other rule.

Signals (window = `dump.window_seconds`, default 90s):

  emergency_dev_dump  the deployer sold after our entry
  emergency_dump      flow flip (sells >= flip_sell_buy_ratio x buys with
                      >= min_sells, or sell SOL >= flip_volume_ratio x buy SOL)
                      AND price crash (>= crash_pct off the window high, or
                      >= crash_from_peak_pct off the position peak)
  emergency_crash     price >= catastrophic_pct off the window high (or off the
                      position peak when no flow is available) — no flow needed
"""

from __future__ import annotations

from typing import Any

from .flow import FlowStats

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "window_seconds": 90,
    "flip_sell_buy_ratio": 2.0,
    "flip_volume_ratio": 2.0,
    "min_sells": 6,
    "crash_pct": 0.15,
    "crash_from_peak_pct": 0.25,
    "catastrophic_pct": 0.35,
    "dev_sell_is_emergency": True,
}


def dump_config(config: dict[str, Any] | None) -> dict[str, Any]:
    exits = (config or {}).get("exits", {}) or {}
    return {**DEFAULTS, **(exits.get("dump", {}) or {})}


def flow_flipped(flow: FlowStats | None, cfg: dict[str, Any]) -> bool:
    if flow is None or flow.trades == 0:
        return False
    by_count = flow.sells >= int(cfg["min_sells"]) and flow.sells >= float(
        cfg["flip_sell_buy_ratio"]
    ) * max(flow.buys, 1)
    by_volume = (
        flow.sell_volume_sol > 0
        and flow.sells >= max(2, int(cfg["min_sells"]) // 2)
        and flow.sell_volume_sol >= float(cfg["flip_volume_ratio"]) * max(flow.buy_volume_sol, 1e-9)
    )
    return by_count or by_volume


def _drop(reference: float, price: float) -> float:
    if reference <= 0 or price <= 0:
        return 0.0
    return max(0.0, (reference - price) / reference)


def detect_dump(
    flow: FlowStats | None,
    price: float,
    peak: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> str | None:
    """Return an emergency reason, or None. Never raises."""
    c = {**DEFAULTS, **(cfg or {})}
    if not c.get("enabled", True):
        return None

    if flow is not None and flow.dev_sold and c.get("dev_sell_is_emergency", True):
        return "emergency_dev_dump"

    px = float(price or 0.0)
    if px <= 0 and flow is not None:
        px = flow.last
    if px <= 0:
        return None

    window_high = flow.high if (flow is not None and flow.high > 0) else 0.0
    drop_window = _drop(window_high, px)
    drop_peak = _drop(float(peak or 0.0), px)

    crashed = drop_window >= float(c["crash_pct"]) or drop_peak >= float(c["crash_from_peak_pct"])
    if crashed and flow_flipped(flow, c):
        return "emergency_dump"
    return None


def detect_catastrophic(
    flow: FlowStats | None,
    price: float,
    peak: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> str | None:
    """Price-only emergency: a crash so large flow confirmation is irrelevant."""
    c = {**DEFAULTS, **(cfg or {})}
    if not c.get("enabled", True):
        return None
    px = float(price or 0.0)
    if px <= 0:
        return None
    limit = float(c["catastrophic_pct"])
    if flow is not None and flow.high > 0 and _drop(flow.high, px) >= limit:
        return "emergency_crash"
    if (flow is None or flow.trades == 0) and _drop(float(peak or 0.0), px) >= limit:
        return "emergency_crash"
    return None
