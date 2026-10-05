#!/usr/bin/env python3
"""Replay the event log: what the desk did, and what it made.

    python scripts/replay.py [--log logs/desk.jsonl] [--days 7] [--market crypto]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.log import read_log  # noqa: E402


def parse_ts(value: str) -> datetime | None:
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def summarize(records: list[dict], days: int | None, market: str | None) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days) if days else None

    buys: list[dict] = []
    closes: list[dict] = []
    skips: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    allocations: list[dict] = []
    costs: list[dict] = []
    staged_adds = 0.0  # crypto scale-in fills are `action: ADD`, not `buy`

    for record in records:
        when = parse_ts(record.get("ts", ""))
        if cutoff and when and when < cutoff:
            continue
        if market and record.get("market") not in (market, None) and record.get("type") != "action":
            continue

        kind = record.get("type")
        if kind == "buy":
            buys.append(record)
        elif kind == "close":
            closes.append(record)
        elif kind == "skip":
            skips[record.get("reason", "unknown")] += 1
        elif kind == "action":
            actions[record.get("action", "unknown")] += 1
            if record.get("action") == "ADD":
                staged_adds += float(record.get("add_usd", 0) or 0)
        elif kind == "allocation":
            allocations.append(record)
        elif kind == "cost":
            costs.append(record)

    per_market: dict[str, dict] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "hold_hours": 0.0}
    )
    for record in closes:
        bucket = per_market[record.get("market", "unknown")]
        pnl = float(record.get("pnl", 0) or 0)
        bucket["trades"] += 1
        bucket["pnl"] += pnl
        bucket["hold_hours"] += float(record.get("hold_time", 0) or 0)
        if pnl >= 0:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1

    for bucket in per_market.values():
        trades = bucket["trades"] or 1
        bucket["win_rate"] = round(bucket["wins"] / trades, 3)
        bucket["avg_pnl"] = round(bucket["pnl"] / trades, 2)
        bucket["avg_hold_hours"] = round(bucket["hold_hours"] / trades, 2)
        bucket["pnl"] = round(bucket["pnl"], 2)

    # Cost records are cumulative snapshots, so the last one is the running total.
    latest_cost = costs[-1] if costs else {}
    model_spend = float(latest_cost.get("cost_usd", 0) or 0)
    total_pnl = round(sum(b["pnl"] for b in per_market.values()), 2)

    return {
        "buys": buys,
        "closes": closes,
        "skips": skips,
        "actions": actions,
        "allocations": allocations,
        "per_market": dict(per_market),
        "total_pnl": total_pnl,
        "deployed": round(sum(float(b.get("amount", 0) or 0) for b in buys) + staged_adds, 2),
        "model_spend": round(model_spend, 4),
        "net_pnl": round(total_pnl - model_spend, 2),
        "cost_detail": latest_cost,
    }


def render(summary: dict) -> str:
    lines = ["", "=" * 62, "  REPLAY", "=" * 62, ""]

    lines.append(f"  buys logged       {len(summary['buys']):>12}")
    lines.append(f"  positions closed  {len(summary['closes']):>12}")
    lines.append(f"  capital deployed  {'$' + format(summary['deployed'], ',.2f'):>12}")
    lines.append(f"  realised PnL      {'$' + format(summary['total_pnl'], ',.2f'):>12}")
    if summary["model_spend"]:
        lines.append(f"  model spend       {'$' + format(summary['model_spend'], ',.4f'):>12}")
        lines.append(f"  net of inference  {'$' + format(summary['net_pnl'], ',.2f'):>12}")
    lines.append("")

    if summary["per_market"]:
        lines.append("  PnL by market")
        lines.append("  " + "-" * 58)
        lines.append(f"  {'market':<10}{'trades':>8}{'win rate':>10}{'PnL':>14}{'avg hold':>12}")
        for name, bucket in sorted(summary["per_market"].items()):
            lines.append(
                f"  {name:<10}{bucket['trades']:>8}{bucket['win_rate']:>10.0%}"
                f"{bucket['pnl']:>14,.2f}{bucket['avg_hold_hours']:>11.1f}h"
            )
        lines.append("")

    if summary["skips"]:
        lines.append("  Why candidates were skipped")
        lines.append("  " + "-" * 58)
        for reason, count in summary["skips"].most_common(12):
            lines.append(f"  {reason:<40}{count:>6}")
        lines.append("")

    if summary["actions"]:
        lines.append("  Exit-manager actions")
        lines.append("  " + "-" * 58)
        for action, count in summary["actions"].most_common():
            lines.append(f"  {action:<40}{count:>6}")
        lines.append("")

    detail = summary["cost_detail"]
    if detail:
        lines.append("  Inference")
        lines.append("  " + "-" * 58)
        lines.append(
            f"  {detail.get('calls', 0)} calls, "
            f"{detail.get('fallbacks', 0)} fallbacks, "
            f"{detail.get('sources_used', 0)} live-search sources, "
            f"{detail.get('cache_hit_rate', 0):.0%} prompt cache"
        )
        for agent, spend in list((detail.get("by_agent") or {}).items())[:6]:
            lines.append(f"    {agent:<38}${spend:>8.4f}")
        lines.append("")

    if summary["allocations"]:
        latest = summary["allocations"][-1]
        lines.append(
            f"  Latest allocation: crypto {float(latest.get('crypto_pct', 0)):.0%} / "
            f"stocks {float(latest.get('stocks_pct', 0)):.0%}"
        )
        if latest.get("reason"):
            lines.append(f"    reason: {latest['reason']}")
        lines.append("")

    lines.append("=" * 62)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the desk event log")
    parser.add_argument("--log", default="logs/desk.jsonl")
    parser.add_argument("--days", type=int, default=None, help="only the last N days")
    parser.add_argument("--market", choices=["crypto", "stocks"], default=None)
    args = parser.parse_args()

    records = read_log(args.log)
    if not records:
        print(f"no records in {args.log}")
        return
    print(render(summarize(records, args.days, args.market)))


if __name__ == "__main__":
    main()
