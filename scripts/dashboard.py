#!/usr/bin/env python3
"""CLI dashboard: today's activity at a glance.

    python scripts/dashboard.py [--config config.yaml] [--log logs/desk.jsonl] [--watch 30]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.log import read_log  # noqa: E402

WIDTH = 64


def parse_ts(value: str) -> datetime | None:
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def bar(fraction: float, width: int = 24) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "█" * filled + "·" * (width - filled)


def render(records: list[dict], budget: float | None = None) -> str:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    recent = [r for r in records if (ts := parse_ts(r.get("ts", ""))) and ts >= since]

    buys = [r for r in recent if r.get("type") == "buy"]
    closes = [r for r in recent if r.get("type") == "close"]
    skips = Counter(r.get("reason", "?") for r in recent if r.get("type") == "skip")
    actions = Counter(r.get("action", "?") for r in recent if r.get("type") == "action")
    allocations = [r for r in records if r.get("type") == "allocation"]
    cost_records = [r for r in records if r.get("type") == "cost"]
    latest_cost = cost_records[-1] if cost_records else {}

    pnl = sum(float(r.get("pnl", 0) or 0) for r in closes)
    deployed = sum(float(r.get("amount", 0) or 0) for r in buys) + sum(
        float(r.get("add_usd", 0) or 0)
        for r in recent
        if r.get("type") == "action" and r.get("action") == "ADD"
    )
    crypto_buys = sum(1 for r in buys if r.get("market") == "crypto")
    stock_buys = len(buys) - crypto_buys

    lines = [
        "",
        "╔" + "═" * WIDTH + "╗",
        "║" + "  GROK TRADING DESK — last 24h".ljust(WIDTH) + "║",
        "╠" + "═" * WIDTH + "╣",
    ]

    def row(text: str) -> None:
        lines.append("║  " + text.ljust(WIDTH - 2) + "║")

    row(f"buys        {len(buys):>4}   (crypto {crypto_buys}, stocks {stock_buys})")
    row(f"closes      {len(closes):>4}")
    row(f"deployed    ${deployed:>10,.2f}")
    row(f"realised    ${pnl:>10,.2f}")
    if latest_cost:
        spend = float(latest_cost.get("cost_usd", 0) or 0)
        row(f"inference   ${spend:>10,.4f}   ({latest_cost.get('calls', 0)} calls, "
            f"{latest_cost.get('fallbacks', 0)} fallback)")
        row(f"net         ${pnl - spend:>10,.2f}")
    if budget:
        row(f"budget use  {bar(min(1.0, deployed / budget))} {deployed / budget:>5.0%}")
    row("")

    if allocations:
        latest = allocations[-1]
        crypto_pct = float(latest.get("crypto_pct", 0.5))
        row(f"allocation  crypto {bar(crypto_pct, 16)} {crypto_pct:>4.0%}")
        row(f"            stocks {bar(1 - crypto_pct, 16)} {1 - crypto_pct:>4.0%}")
        row("")

    if actions:
        row("exit actions")
        for action, count in actions.most_common():
            row(f"  {action:<12}{count:>4}")
        row("")

    if skips:
        row("top skip reasons")
        for reason, count in skips.most_common(6):
            row(f"  {reason:<34}{count:>4}")
        row("")

    if buys:
        row("latest buys")
        for record in buys[-5:]:
            symbol = str(record.get("symbol", "?"))[:14]
            row(
                f"  {record.get('market', '?'):<7}{symbol:<15}"
                f"score {float(record.get('score', 0)):.2f}  ${float(record.get('amount', 0)):,.0f}"
            )

    lines.append("╚" + "═" * WIDTH + "╝")
    return "\n".join(lines)


def load_budget(config_path: str) -> float | None:
    try:
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return float((config.get("risk", {}) or {}).get("total_budget_usd", 0)) or None
    except Exception:  # noqa: BLE001 - the dashboard works fine without a config
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Grok Trading Desk dashboard")
    parser.add_argument("--log", default="logs/desk.jsonl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    args = parser.parse_args()

    budget = load_budget(args.config)
    while True:
        records = read_log(args.log)
        if args.watch:
            print("\033[2J\033[H", end="")
        print(render(records, budget))
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
