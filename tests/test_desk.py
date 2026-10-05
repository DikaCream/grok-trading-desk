"""Desk-level wiring: the paths that only break when the pieces are combined."""

import json
from datetime import datetime, timezone

import pytest
import yaml

from tests.conftest import FakeClient
from src.desk import TradingDesk
from src.models import Market, Position, Stock, Token
from src.stocks.stock_executor import OrderRejected, classify_rejection

GOOD = {
    "crypto_pulse": {"regime": "risk_on", "go_signal": 0.9, "risk_appetite": 0.8},
    "auditor": {"coordinated_buys": False, "wash_trading": False, "bundled_launch": False,
                "sniper_pct": 0.02, "insider_pct": 0.01, "safety_score": 0.95, "red_flags": []},
    "narrative": {"meme_score": 0.85, "originality": 0.8, "virality": 0.9,
                  "community_signal": 0.7, "is_derivative": False, "theme": "dog"},
    "crypto_checker": {"approve": True, "confidence": 0.8, "adjusted_score": 0.8,
                       "kill_reasons": []},
    "market_pulse": {"regime": "risk_on", "go_signal": 0.8, "volatility": "low"},
    "analyst": {"fundamentals_score": 0.8, "technicals_score": 0.85, "trend": "up",
                "support": 45, "resistance": 60, "valuation": "fair"},
    "radar": {"sentiment_score": 0.8, "news_momentum": 0.7, "controversy": 0.05,
              "catalysts": ["earnings"]},
    "insider": {"insider_buying": 0.7, "insider_selling": 0.1, "institutional_flow": 0.8,
                "cluster_buying": True},
    "stock_checker": {"approve": True, "confidence": 0.8, "adjusted_score": 0.75,
                      "suggested_stop_pct": 0.07, "suggested_target_pct": 0.18},
    "allocator": {"crypto_pct": 0.55, "stocks_pct": 0.45, "reason": "crypto hot"},
    "exit_manager": {"action": "HOLD", "reason": "intact", "confidence": 0.6},
}

# Curve reserves are part of every real token: without a quote price the paper
# book refuses to open (no_quote_price), exactly as a live venue would.
TOKEN = Token(mint="M1", symbol="WIF2", holders=200, buys=90, sells=25, unique_traders=60,
              liquidity_usd=40000, mint_revoked=True, age_seconds=300,
              curve_sol=32.0, curve_tokens=800_000_000.0)
STOCK = Stock(symbol="ACME", sector="technology", price=50, prev_close=46,
              avg_volume=2e6, volume=6e6, market_cap=5e9)


def build(tmp_path, overrides=None, **kwargs) -> TradingDesk:
    config = yaml.safe_load(open("config.example.yaml"))
    config["logging"] = {"path": str(tmp_path / "desk.jsonl"), "echo_stdout": False,
                         "cost_report_every": 0}
    desk = TradingDesk(config, dry_run=kwargs.pop("dry_run", True), **kwargs)

    replies = {**GOOD, **(overrides or {})}
    for name, reply in replies.items():
        getattr(desk, name)._client = FakeClient([reply])
    return desk


# --- crypto path ------------------------------------------------------------------

async def test_a_clean_token_is_bought(tmp_path):
    desk = build(tmp_path)
    result = await desk.evaluate_token(TOKEN)
    assert result["bought"] is True
    records = [json.loads(line) for line in open(tmp_path / "desk.jsonl")]
    assert records[-1]["type"] == "buy"
    assert records[-1]["market"] == "crypto"


async def test_a_wash_traded_token_never_reaches_the_checker(tmp_path):
    desk = build(tmp_path, {"auditor": {**GOOD["auditor"], "wash_trading": True}})
    result = await desk.evaluate_token(TOKEN)
    assert result["reason"] == "veto_wash_trading"
    assert desk.crypto_checker._client.calls == []   # veto is free


async def test_a_paused_crypto_market_blocks_everything(tmp_path):
    desk = build(tmp_path, {"crypto_pulse": {"regime": "risk_off", "go_signal": 0.1}})
    assert (await desk.evaluate_token(TOKEN))["reason"] == "veto_market_paused"


async def test_the_checker_can_veto_a_high_scoring_token(tmp_path):
    desk = build(tmp_path, {"crypto_checker": {"approve": False, "confidence": 0.9,
                                               "kill_reasons": ["liquidity too thin"]}})
    assert (await desk.evaluate_token(TOKEN))["reason"] == "checker_rejected"


# --- stock path -------------------------------------------------------------------

async def test_a_clean_stock_is_bought(tmp_path):
    desk = build(tmp_path)
    pulse = await desk.market_pulse.run()
    assert (await desk.evaluate_stock(STOCK, pulse))["bought"] is True


async def test_controversy_vetoes_before_the_checker(tmp_path):
    desk = build(tmp_path, {"radar": {**GOOD["radar"], "controversy": 0.95}})
    pulse = await desk.market_pulse.run()
    assert (await desk.evaluate_stock(STOCK, pulse))["reason"] == "veto_controversy"
    assert desk.stock_checker._client.calls == []


async def test_a_pdt_block_is_logged_as_its_own_reason(tmp_path):
    desk = build(tmp_path, dry_run=False)

    class Blocked:
        paper = True

        async def buy_bracket(self, *a, **k):
            raise OrderRejected("pdt_blocked", "403 forbidden")

        async def get_positions(self):
            return []

    desk.stock_executor = Blocked()
    pulse = await desk.market_pulse.run()
    result = await desk.evaluate_stock(STOCK, pulse)

    assert result["reason"] == "pdt_blocked"
    records = [json.loads(line) for line in open(tmp_path / "desk.jsonl")]
    assert records[-1]["reason"] == "pdt_blocked"


def test_rejection_classifier_recognises_the_common_refusals():
    class Err(Exception):
        def __init__(self, msg, status=None):
            super().__init__(msg)
            self.status_code = status

    assert classify_rejection(Err("forbidden", 403)) == "pdt_blocked"
    assert classify_rejection(Err("potential wash trade detected")) == "wash_trade_blocked"
    assert classify_rejection(Err("insufficient buying power")) == "insufficient_buying_power"
    assert classify_rejection(Err("asset is not active")) == "asset_not_tradable"
    assert classify_rejection(Err("slow down", 429)) == "broker_rate_limited"
    assert classify_rejection(Err("something odd")) == "order_rejected"


# --- risk interaction ---------------------------------------------------------------

async def test_the_daily_loss_limit_stops_new_entries(tmp_path):
    desk = build(tmp_path)
    desk.risk.record_close(Market.CRYPTO, -desk.risk.daily_loss_limit_usd)
    assert (await desk.evaluate_token(TOKEN))["reason"] == "daily_loss_limit_reached"


async def test_the_sector_cap_is_enforced_through_the_desk(tmp_path):
    desk = build(tmp_path)
    desk.positions = [
        Position(market=Market.STOCKS, symbol=s, quantity=1, entry_price=10, sector="technology")
        for s in ("A", "B")
    ]
    pulse = await desk.market_pulse.run()
    assert (await desk.evaluate_stock(STOCK, pulse))["reason"] == "max_per_sector"


# --- allocation, exits, memory --------------------------------------------------------

async def test_allocation_is_clamped_and_logged(tmp_path):
    desk = build(tmp_path, {"allocator": {"crypto_pct": 0.99, "stocks_pct": 0.01,
                                          "reason": "all in"}})
    allocation = await desk.run_allocation()
    assert allocation.crypto_pct <= 0.99
    assert desk.risk.allocation.crypto_pct == allocation.crypto_pct
    kinds = [json.loads(line)["type"] for line in open(tmp_path / "desk.jsonl")]
    assert "allocation" in kinds and "cost" in kinds


async def test_exit_pass_holds_and_logs_an_action(tmp_path):
    desk = build(tmp_path)
    desk.positions = [Position(market=Market.CRYPTO, symbol="WIF2", quantity=1000,
                               entry_price=0.001, current_price=0.002)]
    decisions = await desk.run_exit_pass()
    assert decisions[0]["action"] == "HOLD"
    actions = [json.loads(l) for l in open(tmp_path / "desk.jsonl") if '"action"' in l]
    assert actions[-1]["action"] == "HOLD"


async def test_memory_reaches_the_checker_but_not_the_analyst(tmp_path):
    desk = build(tmp_path)
    assert desk.crypto_checker.memory is desk.memory
    assert desk.stock_checker.memory is desk.memory
    assert desk.exit_manager.memory is desk.memory
    assert desk.allocator.memory is desk.memory
    # analysts describe what is in front of them; old trades would only anchor them
    assert desk.analyst.memory is None
    assert desk.auditor.memory is None


async def test_past_outcomes_reach_the_checker_prompt(tmp_path):
    desk = build(tmp_path)
    # a prior losing dog trade, written through the desk's own log
    desk.log.buy("crypto", "PUPPY", 0.7, {"narrative": {"theme": "dog"}}, 100.0, "tx")
    desk.log.close("crypto", "PUPPY", -60.0, 4.0)
    desk.refresh_memory()

    await desk.evaluate_token(TOKEN)
    sent = desk.crypto_checker._client.calls[0]["json"]["messages"][1]["content"]
    assert "past_outcomes" in sent and "PUPPY" in sent


# --- costs -----------------------------------------------------------------------------

async def test_costs_accumulate_across_the_whole_desk(tmp_path):
    desk = build(tmp_path)
    from tests.conftest import FakeResponse

    usage = {"prompt_tokens": 100, "completion_tokens": 20, "cost_in_usd_ticks": 10_000_000_000}
    for name in ("crypto_pulse", "auditor", "narrative", "crypto_checker"):
        bot = getattr(desk, name)
        bot._client = FakeClient([FakeResponse(json.dumps(GOOD[name]), usage=usage)])

    await desk.evaluate_token(TOKEN)
    snap = desk.costs.snapshot()
    assert snap["calls"] == 4
    assert snap["cost_usd"] == pytest.approx(4.0)
    assert set(snap["by_agent"]) == {"crypto_pulse", "auditor", "narrative", "crypto_checker"}


# --- config sanity ------------------------------------------------------------------------

def test_shipped_config_uses_live_model_slugs(tmp_path):
    desk = build(tmp_path)
    # grok-4-fast and grok-4 were retired 2026-05-15
    assert desk.analyst.model == "grok-4.3"
    assert desk.stock_checker.model == "grok-4.6"
    assert desk.crypto_checker.model != desk.narrative.model


def test_every_retrieval_agent_declares_a_search_policy(tmp_path):
    desk = build(tmp_path)
    for name in ("radar", "insider", "analyst", "market_pulse", "crypto_pulse", "narrative"):
        assert getattr(desk, name).SEARCH is not None, name


def test_market_hours_window(tmp_path):
    desk = build(tmp_path)
    weekday = datetime(2026, 8, 27, 10, 0, tzinfo=timezone.utc)  # a Thursday
    assert desk.market_is_open(weekday.replace(hour=10, minute=0)) is True
    assert desk.market_is_open(weekday.replace(hour=8, minute=0)) is False
    saturday = datetime(2026, 8, 29, 10, 0, tzinfo=timezone.utc)
    assert desk.market_is_open(saturday) is False


# --- mechanical TP/SL for crypto (improve-memecoin-exits) ----------------------

from src.desk import check_crypto_stops, crypto_stop_tp_prices, curve_price


def test_crypto_stop_tp_prices_from_entry():
    stop, tp = crypto_stop_tp_prices(1.0, take_profit_pct=0.20, stop_loss_pct=0.08)
    assert stop == pytest.approx(0.92)
    assert tp == pytest.approx(1.20)
    assert crypto_stop_tp_prices(0.0) == (None, None)


def test_check_crypto_stops_triggers_close_on_stop_and_tp():
    pos = Position(
        market=Market.CRYPTO, symbol="WIF2", quantity=1000,
        entry_price=1.0, current_price=1.0,
        stop_price=0.92, take_profit_price=1.20,
    )
    assert check_crypto_stops(pos, 0.92) == "stop_loss"
    assert check_crypto_stops(pos, 0.91) == "stop_loss"
    assert check_crypto_stops(pos, 1.20) == "take_profit"
    assert check_crypto_stops(pos, 1.25) == "take_profit"
    assert check_crypto_stops(pos, 1.0) is None
    assert check_crypto_stops(pos, 0.0) is None  # no usable price


def test_check_crypto_stops_ignores_stocks():
    pos = Position(
        market=Market.STOCKS, symbol="ACME", quantity=10,
        entry_price=50, current_price=40,
        stop_price=46, take_profit_price=60,
    )
    assert check_crypto_stops(pos, 40) is None


async def test_open_crypto_sets_stop_and_take_profit(tmp_path):
    desk = build(tmp_path)
    token = Token(
        mint="M1", symbol="WIF2", holders=200, buys=90, sells=25,
        unique_traders=60, liquidity_usd=40000, mint_revoked=True,
        age_seconds=300, curve_sol=32.0, curve_tokens=800_000_000.0,
    )
    result = await desk.evaluate_token(token)
    assert result["bought"] is True
    assert len(desk.positions) == 1
    pos = desk.positions[0]
    quote = curve_price(token)
    # Paper fill: at the quote plus curve impact and the protocol fee, never better.
    assert quote < pos.entry_price <= quote * 1.05
    entry = pos.entry_price
    stop_pct = desk.exits_cfg["stop_loss_pct"]
    runner_cap = desk.exits_cfg["ladder"]["runner_take_profit_pct"]
    assert pos.stop_price == pytest.approx(entry * (1 - stop_pct))
    # Ladder on: take_profit_price is the runner cap; rungs live in meta["ladder"].
    assert pos.take_profit_price == pytest.approx(entry * (1 + runner_cap))
    assert pos.meta["ladder"]["rungs_hit"] == []
    assert result["stop_price"] == pos.stop_price
    assert result["take_profit_price"] == pos.take_profit_price


async def test_check_open_crypto_stops_closes_on_mocked_price(tmp_path):
    desk = build(tmp_path)
    entry = 1.0
    pos = Position(
        market=Market.CRYPTO, symbol="WIF2", quantity=1000,
        entry_price=entry, current_price=entry, amount_usd=100.0,
        stop_price=0.92, take_profit_price=1.20,
        meta={"mint": "M1"},
    )
    desk.positions = [pos]
    desk.risk.record_fill(Market.CRYPTO, 100.0)

    closed = await desk.check_open_crypto_stops(prices={"M1": 0.90})
    assert len(closed) == 1
    assert closed[0]["reason"] == "stop_loss"
    assert closed[0]["closed"] is True
    assert desk.positions == []

    records = [json.loads(line) for line in open(tmp_path / "desk.jsonl")]
    types = [r["type"] for r in records]
    assert "action" in types and "close" in types
    action = next(r for r in records if r["type"] == "action")
    assert action["action"] == "CLOSE" and action["reason"] == "stop_loss"


async def test_check_open_crypto_stops_closes_at_take_profit(tmp_path):
    desk = build(tmp_path)
    pos = Position(
        market=Market.CRYPTO, symbol="WIF2", quantity=1000,
        entry_price=1.0, current_price=1.0, amount_usd=100.0,
        stop_price=0.92, take_profit_price=1.20,
        meta={"mint": "M1"},
    )
    desk.positions = [pos]
    desk.risk.record_fill(Market.CRYPTO, 100.0)

    closed = await desk.check_open_crypto_stops(prices={"M1": 1.25})
    assert closed[0]["reason"] == "take_profit"
    assert desk.positions == []


async def test_bundled_launch_veto_reaches_the_desk(tmp_path):
    desk = build(tmp_path, {"auditor": {**GOOD["auditor"], "bundled_launch": True}})
    result = await desk.evaluate_token(TOKEN)
    assert result["reason"] == "veto_bundled_launch"
    assert desk.crypto_checker._client.calls == []
