"""Enricher, price poller, time-stop, liquidity cap, confidence floor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from src.crypto.enricher import (
    OnChainEnricher,
    build_mint_account_data,
    parse_mint_revoked,
    parse_mint_supply,
)
from src.crypto.price_poller import CryptoPricePoller
from src.crypto.scout import Scout, Watch, filter_reason, parse_create_event
from src.desk import TradingDesk, check_crypto_exits, check_crypto_stops
from src.models import Market, Position, Token
from src.shared.risk import RiskManager
from tests.conftest import FakeClient
from tests.test_scout_filter import CREATE_EVENT, LAUNCH_FILTER, WATCH_FILTER, trade


# --- enricher primitives -------------------------------------------------------

def test_parse_mint_revoked_true_when_authority_none():
    data = build_mint_account_data(authority=None, supply=1_000_000)
    assert parse_mint_revoked(data) is True
    assert parse_mint_supply(data) == 1_000_000


def test_parse_mint_revoked_false_when_authority_set():
    from solders.pubkey import Pubkey

    authority = Pubkey.from_string("11111111111111111111111111111111")
    data = build_mint_account_data(authority=authority)
    assert parse_mint_revoked(data) is False


def test_parse_mint_revoked_none_on_garbage():
    assert parse_mint_revoked(None) is None
    assert parse_mint_revoked(b"\x00\x01") is None


class FakeRpc:
    """Duck-typed AsyncClient for enricher unit tests."""

    def __init__(
        self,
        mint_data: bytes | None,
        largest=None,
        supply_ui: float = 1000.0,
        accounts: dict | None = None,
    ):
        self.mint_data = mint_data
        self.largest = largest or []
        self.supply_ui = supply_ui
        self.accounts = accounts or {}
        self.calls: list[str] = []

    async def get_account_info(self, pubkey, *a, **k):
        key = str(pubkey)
        self.calls.append(f"account:{key}")
        if key in self.accounts:
            data = self.accounts[key]
            if data is None:
                return SimpleNamespace(value=None)
            return SimpleNamespace(value=SimpleNamespace(data=data))
        if self.mint_data is not None:
            return SimpleNamespace(value=SimpleNamespace(data=self.mint_data))
        return SimpleNamespace(value=None)

    async def get_token_largest_accounts(self, pubkey, *a, **k):
        self.calls.append("largest")
        return SimpleNamespace(value=self.largest)

    async def get_token_supply(self, pubkey, *a, **k):
        self.calls.append("supply")
        return SimpleNamespace(value={"ui_amount": self.supply_ui})

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_enricher_fills_mint_revoked_and_top10():
    mint_data = build_mint_account_data(authority=None, supply=1_000_000_000)
    largest = [
        {"ui_amount": 100.0},
        {"ui_amount": 50.0},
        {"ui_amount": 25.0},
    ]
    rpc = FakeRpc(mint_data, largest=largest, supply_ui=1000.0)
    enricher = OnChainEnricher({}, rpc=rpc)
    token = Token(mint="So11111111111111111111111111111111111111112", creator="")
    out = await enricher.enrich(token)
    assert out.mint_revoked is True
    assert out.top10_holder_pct == pytest.approx(0.175, abs=0.001)
    assert out.holders_known is True
    assert out.holders == 3


@pytest.mark.asyncio
async def test_enricher_unknown_mint_leaves_none_on_rpc_failure():
    class BoomRpc:
        async def get_account_info(self, *a, **k):
            raise RuntimeError("rpc down")

        async def get_token_largest_accounts(self, *a, **k):
            raise RuntimeError("rpc down")

        async def get_token_supply(self, *a, **k):
            raise RuntimeError("rpc down")

        async def close(self):
            pass

    enricher = OnChainEnricher({}, rpc=BoomRpc())
    token = Token(mint="So11111111111111111111111111111111111111112")
    out = await enricher.enrich(token)
    assert out.mint_revoked is None
    assert out.holders_known is False


# --- filter: unknown must not pass hard requires --------------------------------

STRICT = {
    **WATCH_FILTER,
    "require_mint_revoked": True,
    "require_lp_burned": True,
}


def _watched(**over) -> Token:
    base = dict(
        mint="MINT1",
        symbol="WIF2",
        liquidity_usd=20000.0,
        holders=100,
        holders_known=True,
        unique_traders=40,
        age_seconds=300,
        buys=60,
        sells=20,
        mint_revoked=True,
        lp_burned=True,
    )
    base.update(over)
    return Token(**base)


def test_require_mint_revoked_rejects_unknown():
    assert filter_reason(_watched(mint_revoked=None), STRICT) == "mint_unknown"


def test_require_mint_revoked_rejects_false():
    assert filter_reason(_watched(mint_revoked=False), STRICT) == "mint_not_revoked"


def test_require_mint_revoked_passes_true():
    assert filter_reason(_watched(mint_revoked=True, lp_burned=True), STRICT) is None


def test_require_lp_burned_rejects_unknown():
    assert filter_reason(_watched(lp_burned=None), STRICT) == "lp_unknown"


def test_holders_threshold_only_when_known():
    # Unknown holder count: skip min_holders, still enforce unique_traders.
    filt = {**WATCH_FILTER, "min_holders": 25, "min_unique_traders": 12}
    assert filter_reason(_watched(holders=0, holders_known=False), filt) is None
    assert filter_reason(_watched(holders=5, holders_known=True), filt) == "too_few_holders"
    assert filter_reason(
        _watched(holders=5, holders_known=False, unique_traders=2), filt
    ) == "too_few_traders"


# --- price poller + mechanical exits -------------------------------------------

@pytest.mark.asyncio
async def test_price_poller_injected_fetch_updates_and_triggers_stop():
    async def fetch(mints):
        return {mints[0]: 0.90}

    poller = CryptoPricePoller({}, fetch=fetch)
    prices = await poller.fetch_prices(["M1"])
    assert prices["M1"] == pytest.approx(0.90)

    pos = Position(
        market=Market.CRYPTO,
        symbol="WIF2",
        quantity=1000,
        entry_price=1.0,
        current_price=1.0,
        amount_usd=100.0,
        stop_price=0.92,
        take_profit_price=1.20,
        meta={"mint": "M1"},
    )
    assert check_crypto_stops(pos, prices["M1"]) == "stop_loss"


@pytest.mark.asyncio
async def test_price_poller_degrades_to_empty_on_fetch_error():
    async def boom(_mints):
        raise RuntimeError("network")

    poller = CryptoPricePoller({}, fetch=boom)
    assert await poller.fetch_prices(["M1"]) == {}


def test_time_stop_closes_after_max_hold_minutes():
    opened = datetime.now(timezone.utc) - timedelta(minutes=90)
    pos = Position(
        market=Market.CRYPTO,
        symbol="WIF2",
        quantity=100,
        entry_price=1.0,
        current_price=1.05,
        opened_at=opened,
        stop_price=0.5,
        take_profit_price=5.0,
    )
    cfg = {"max_hold_minutes": 60}
    assert check_crypto_exits(pos, 1.05, exits_cfg=cfg) == "time_stop"


def test_trailing_stop_after_activation():
    pos = Position(
        market=Market.CRYPTO,
        symbol="WIF2",
        quantity=100,
        entry_price=1.0,
        current_price=1.30,
        peak_price=1.40,
        stop_price=0.5,
        take_profit_price=5.0,
    )
    cfg = {"trailing_stop_activate_pct": 0.25, "trailing_stop_pct": 0.10}
    # peak 1.40, trail 10% → stop at 1.26; mark 1.25 fires
    assert check_crypto_exits(pos, 1.25, exits_cfg=cfg) == "trailing_stop"
    # still above trail stop
    pos2 = pos.model_copy(update={"peak_price": 1.40, "current_price": 1.30})
    assert check_crypto_exits(pos2, 1.30, exits_cfg=cfg) is None


def test_liquidity_size_cap_crypto_only():
    config = {
        "risk": {
            "total_budget_usd": 2000.0,
            "daily_loss_limit_usd": 1000.0,  # roomy so liquidity is the binding cap
            "max_position_pct_of_market": 0.50,
            "max_position_pct_of_remaining_loss": 0.50,
            "max_position_pct_of_liquidity": 0.03,
            "crypto_max_pct": 1.0,
            "stock_max_pct": 1.0,
        }
    }
    risk = RiskManager(config)
    # $10k curve liquidity → max $300 at 3%
    capped = risk.position_size(Market.CRYPTO, score=1.0, liquidity_usd=10_000.0)
    uncapped = risk.position_size(Market.CRYPTO, score=1.0)
    assert capped == pytest.approx(300.0)
    assert uncapped > capped
    # Stocks ignore liquidity_usd
    stock = risk.position_size(Market.STOCKS, score=1.0, liquidity_usd=10_000.0)
    assert stock == risk.position_size(Market.STOCKS, score=1.0)


# --- desk wiring: confidence floor + poller stop --------------------------------

GOOD = {
    "crypto_pulse": {"regime": "risk_on", "go_signal": 0.9, "risk_appetite": 0.8},
    "auditor": {
        "coordinated_buys": False,
        "wash_trading": False,
        "bundled_launch": False,
        "sniper_pct": 0.02,
        "insider_pct": 0.01,
        "safety_score": 0.95,
        "red_flags": [],
    },
    "narrative": {
        "meme_score": 0.85,
        "originality": 0.8,
        "virality": 0.9,
        "community_signal": 0.7,
        "is_derivative": False,
        "theme": "dog",
    },
    "crypto_checker": {
        "approve": True,
        "confidence": 0.8,
        "adjusted_score": 0.8,
        "kill_reasons": [],
    },
    "market_pulse": {"regime": "risk_on", "go_signal": 0.8, "volatility": "low"},
    "analyst": {
        "fundamentals_score": 0.8,
        "technicals_score": 0.85,
        "trend": "up",
        "support": 45,
        "resistance": 60,
        "valuation": "fair",
    },
    "radar": {
        "sentiment_score": 0.8,
        "news_momentum": 0.7,
        "controversy": 0.05,
        "catalysts": ["earnings"],
    },
    "insider": {
        "insider_buying": 0.7,
        "insider_selling": 0.1,
        "institutional_flow": 0.8,
        "cluster_buying": True,
    },
    "stock_checker": {
        "approve": True,
        "confidence": 0.8,
        "adjusted_score": 0.75,
        "suggested_stop_pct": 0.07,
        "suggested_target_pct": 0.18,
    },
    "allocator": {"crypto_pct": 0.55, "stocks_pct": 0.45, "reason": "crypto hot"},
    "exit_manager": {"action": "HOLD", "reason": "intact", "confidence": 0.6},
}

TOKEN = Token(
    mint="M1",
    symbol="WIF2",
    holders=200,
    holders_known=True,
    buys=90,
    sells=25,
    unique_traders=60,
    liquidity_usd=40000,
    mint_revoked=True,
    age_seconds=300,
    curve_sol=32.0,
    curve_tokens=800_000_000.0,
)


def build(tmp_path, overrides=None, **kwargs) -> TradingDesk:
    config = yaml.safe_load(open("config.example.yaml"))
    config["logging"] = {
        "path": str(tmp_path / "desk.jsonl"),
        "echo_stdout": False,
        "cost_report_every": 0,
    }
    desk = TradingDesk(config, dry_run=kwargs.pop("dry_run", True), **kwargs)
    replies = {**GOOD, **(overrides or {})}
    for name, reply in replies.items():
        getattr(desk, name)._client = FakeClient([reply])
    return desk


@pytest.mark.asyncio
async def test_checker_confidence_floor_rejects(tmp_path):
    desk = build(
        tmp_path,
        {
            "crypto_checker": {
                "approve": True,
                "confidence": 0.4,
                "adjusted_score": 0.8,
                "kill_reasons": [],
            }
        },
    )
    result = await desk.evaluate_token(TOKEN)
    assert result["bought"] is False
    assert result["reason"] == "checker_low_confidence"


@pytest.mark.asyncio
async def test_desk_poller_triggers_mechanical_stop(tmp_path):
    desk = build(tmp_path)

    async def fetch(mints):
        return {mints[0]: 0.90}

    desk.price_poller = CryptoPricePoller({}, fetch=fetch)
    pos = Position(
        market=Market.CRYPTO,
        symbol="WIF2",
        quantity=1000,
        entry_price=1.0,
        current_price=1.0,
        amount_usd=100.0,
        stop_price=0.92,
        take_profit_price=1.20,
        meta={"mint": "M1"},
    )
    desk.positions = [pos]
    desk.risk.record_fill(Market.CRYPTO, 100.0)

    prices = await desk.poll_crypto_marks()
    assert prices["M1"] == pytest.approx(0.90)
    closed = await desk.check_open_crypto_stops(prices)
    assert len(closed) == 1
    assert closed[0]["reason"] == "stop_loss"
    assert desk.positions == []


@pytest.mark.asyncio
async def test_open_crypto_respects_liquidity_cap(tmp_path):
    desk = build(tmp_path)
    # $1k liquidity would (rightly) fail the scorecard's liquidity kill floor;
    # switch it off so this test isolates the sizing cap.
    desk.scorecard.enabled = False
    # Tiny liquidity → size capped hard
    token = TOKEN.model_copy(
        update={
            "liquidity_usd": 1000.0,
            "curve_sol": 5.0,
            "curve_tokens": 1_000_000_000.0,
        }
    )
    result = await desk.evaluate_token(token)
    assert result["bought"] is True
    # 3% of $1000 = $30
    assert result["amount"] <= 30.0 + 1e-6
    assert desk.positions[0].amount_usd <= 30.0 + 1e-6


@pytest.mark.asyncio
async def test_scout_mature_runs_enricher_before_filter():
    class TrackingEnricher:
        def __init__(self):
            self.called = False

        async def enrich(self, token):
            self.called = True
            return token.model_copy(
                update={
                    "mint_revoked": True,
                    "holders": 80,
                    "holders_known": True,
                    "top10_holder_pct": 0.2,
                    "dev_holding_pct": 0.02,
                }
            )

    enricher = TrackingEnricher()
    config = {
        "crypto_launch_filter": LAUNCH_FILTER,
        "crypto_filter": {**WATCH_FILTER, "require_mint_revoked": True},
        "pump_fun": {
            "sol_price_usd": 200.0,
            "watch": {
                "enabled": True,
                "window_seconds": 300,
                "max_concurrent": 3,
                "min_trades_to_score": 12,
            },
        },
    }
    scout = Scout(config, enricher=enricher)
    token = scout.handle_create(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)
    scout.watching[token.mint] = watch
    for i in range(30):
        watch.record(trade(trader=f"T{i}"))

    ready = await scout.mature(now=400.0)
    assert enricher.called is True
    assert [t.mint for t in ready] == ["MINT1"]
    assert ready[0].holders_known is True
    assert ready[0].mint_revoked is True


@pytest.mark.asyncio
async def test_scout_mature_rejects_mint_unknown_without_enricher():
    config = {
        "crypto_launch_filter": LAUNCH_FILTER,
        "crypto_filter": {**WATCH_FILTER, "require_mint_revoked": True},
        "pump_fun": {
            "sol_price_usd": 200.0,
            "watch": {
                "enabled": True,
                "window_seconds": 300,
                "max_concurrent": 3,
                "min_trades_to_score": 12,
            },
        },
    }
    scout = Scout(config)  # no enricher → mint_revoked stays None → mint_unknown
    token = scout.handle_create(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)
    scout.watching[token.mint] = watch
    for i in range(30):
        watch.record(trade(trader=f"T{i}"))
    ready = await scout.mature(now=400.0)
    assert ready == []


def test_watch_result_does_not_proxy_holders_as_unique_traders():
    token = parse_create_event(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)
    for i in range(15):
        watch.record(trade(trader=f"T{i}"))
    result = watch.result(sol_usd=200.0, now=120.0)
    assert result.unique_traders == 15
    assert result.holders == 0
    assert result.holders_known is False
