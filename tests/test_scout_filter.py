"""Scout tests, built on the payload PumpPortal actually sends.

The old fixtures invented fields (holders, buys, age) that a `create` event
never carries, so the filter passed in tests and could never pass in production.
These use the documented event shape.
"""

import json

import pytest

from src.crypto.scout import (
    Scout,
    Watch,
    filter_reason,
    launch_reason,
    parse_create_event,
    passes_filter,
)
from src.models import Token

# A real subscribeNewToken frame: this is every field the feed sends.
CREATE_EVENT = {
    "signature": "3hJ7abc",
    "mint": "MINT1",
    "traderPublicKey": "DEV1",
    "txType": "create",
    "name": "dogwifhat2",
    "symbol": "WIF2",
    "uri": "https://ipfs.io/ipfs/abc",
    "initialBuy": 100000000,
    "solAmount": 0.5,
    "bondingCurveKey": "CURVE1",
    "vTokensInBondingCurve": 900000000.0,
    "vSolInBondingCurve": 32.0,
    "marketCapSol": 40.0,
    "pool": "pump",
}

LAUNCH_FILTER = {
    "min_curve_sol": 20.0,
    "max_curve_sol": 200.0,
    "min_market_cap_sol": 25.0,
    "max_market_cap_sol": 2000.0,
    "max_dev_initial_buy_sol": 2.0,
    "require_metadata": True,
    "require_socials": False,
}

WATCH_FILTER = {
    "min_liquidity_usd": 5000.0,
    "max_liquidity_usd": 400000.0,
    "min_holders": 25,
    "min_unique_traders": 12,
    "min_age_seconds": 60,
    "min_buys": 15,
    "min_buy_sell_ratio": 1.2,
}


def trade(side="buy", trader="T1", sol=0.4, curve_sol=33.0, curve_tokens=880000000.0, mint="MINT1"):
    return {
        "signature": "sig",
        "mint": mint,
        "traderPublicKey": trader,
        "txType": side,
        "solAmount": sol,
        "vSolInBondingCurve": curve_sol,
        "vTokensInBondingCurve": curve_tokens,
        "marketCapSol": 45.0,
    }


# --- parsing -------------------------------------------------------------------

def test_parse_create_event_reads_the_documented_fields():
    token = parse_create_event(CREATE_EVENT)
    assert token.mint == "MINT1"
    assert token.symbol == "WIF2"
    assert token.creator == "DEV1"
    assert token.curve_sol == 32.0
    assert token.curve_tokens == 900000000.0
    assert token.market_cap_sol == 40.0
    assert token.dev_initial_buy_sol == 0.5
    assert token.pool == "pump"


def test_create_event_carries_no_watch_metrics():
    # The point of the two-stage design: these are all zero at creation.
    token = parse_create_event(CREATE_EVENT)
    assert token.holders == 0
    assert token.buys == 0 and token.sells == 0
    assert token.age_seconds == 0
    assert token.unique_traders == 0


def test_sol_denominated_fields_convert_to_usd():
    token = parse_create_event(CREATE_EVENT).priced(200.0)
    assert token.liquidity_usd == pytest.approx(6400.0)
    assert token.market_cap_usd == pytest.approx(8000.0)
    # a zero rate must not silently zero the token out
    assert parse_create_event(CREATE_EVENT).priced(0).liquidity_usd == 0.0


def test_socials_are_collected_from_either_shape():
    flat = parse_create_event({**CREATE_EVENT, "twitter": "https://x.com/a"})
    assert flat.socials["twitter"] == "https://x.com/a"
    nested = parse_create_event({**CREATE_EVENT, "socials": {"telegram": "t.me/b"}})
    assert nested.socials["telegram"] == "t.me/b"


# --- stage one -----------------------------------------------------------------

def test_clean_launch_passes_stage_one():
    assert launch_reason(parse_create_event(CREATE_EVENT), LAUNCH_FILTER) is None


@pytest.mark.parametrize(
    "over,reason",
    [
        ({"vSolInBondingCurve": 5.0}, "curve_too_small"),
        ({"vSolInBondingCurve": 900.0}, "curve_too_large"),
        ({"marketCapSol": 3.0}, "market_cap_too_small"),
        ({"marketCapSol": 99999.0}, "market_cap_too_large"),
        ({"solAmount": 9.0}, "dev_initial_buy_too_large"),
        ({"uri": ""}, "incomplete_metadata"),
        ({"name": ""}, "incomplete_metadata"),
        ({"mint": ""}, "no_mint"),
    ],
)
def test_stage_one_rejection_reasons(over, reason):
    token = parse_create_event({**CREATE_EVENT, **over})
    assert launch_reason(token, LAUNCH_FILTER) == reason


def test_socials_only_required_when_configured():
    token = parse_create_event(CREATE_EVENT)
    assert launch_reason(token, LAUNCH_FILTER) is None
    strict = {**LAUNCH_FILTER, "require_socials": True}
    assert launch_reason(token, strict) == "no_socials"


def test_stage_one_boundaries_are_inclusive():
    edge = parse_create_event(
        {**CREATE_EVENT, "vSolInBondingCurve": 20.0, "marketCapSol": 25.0, "solAmount": 2.0}
    )
    assert launch_reason(edge, LAUNCH_FILTER) is None


# --- stage two -----------------------------------------------------------------

def watched_token(**over) -> Token:
    base = dict(
        mint="MINT1", symbol="WIF2", liquidity_usd=20000.0, holders=100,
        holders_known=True, unique_traders=40, age_seconds=300, buys=60, sells=20,
        mint_revoked=True,
    )
    base.update(over)
    return Token(**base)


def test_clean_watched_token_passes():
    assert filter_reason(watched_token(), WATCH_FILTER) is None
    assert passes_filter(watched_token(), WATCH_FILTER)


@pytest.mark.parametrize(
    "over,reason",
    [
        ({"liquidity_usd": 1000.0}, "liquidity_too_low"),
        ({"liquidity_usd": 900000.0}, "liquidity_too_high"),
        ({"holders": 5}, "too_few_holders"),
        ({"unique_traders": 2}, "too_few_traders"),
        ({"age_seconds": 10}, "too_young"),
        ({"buys": 3, "sells": 1}, "too_few_buys"),
        ({"buys": 20, "sells": 40}, "weak_buy_sell_ratio"),
        ({"dev_sold": True}, "dev_sold"),
        ({"mint": ""}, "no_mint"),
    ],
)
def test_stage_two_rejection_reasons(over, reason):
    assert filter_reason(watched_token(**over), WATCH_FILTER) == reason


def test_unknown_mint_authority_is_rejected_when_required():
    # After the enricher path: None means unknown and must NOT silently pass.
    strict = {**WATCH_FILTER, "require_mint_revoked": True, "require_lp_burned": True}
    assert filter_reason(watched_token(mint_revoked=None), strict) == "mint_unknown"
    assert filter_reason(watched_token(mint_revoked=True, lp_burned=None), strict) == "lp_unknown"


def test_a_confirmed_live_mint_authority_still_rejects():
    strict = {**WATCH_FILTER, "require_mint_revoked": True}
    assert filter_reason(watched_token(mint_revoked=False), strict) == "mint_not_revoked"
    assert filter_reason(watched_token(mint_revoked=True), strict) is None


def test_absent_thresholds_are_not_failures():
    # An empty filter means "no opinion", not "reject everything".
    assert filter_reason(Token(mint="M"), {}) is None


def test_buy_sell_ratio_with_zero_sells():
    assert watched_token(buys=30, sells=0).buy_sell_ratio == 30.0
    assert Token(mint="M").buy_sell_ratio == 0.0


# --- the watch window ------------------------------------------------------------

def test_watch_accumulates_trade_statistics():
    token = parse_create_event(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)

    for i in range(20):
        watch.record(trade(side="buy", trader=f"T{i}"))
    for i in range(5):
        watch.record(trade(side="sell", trader=f"T{i}"))

    result = watch.result(sol_usd=200.0, now=120.0)
    assert result.buys == 20
    assert result.sells == 5
    assert result.unique_traders == 20
    assert result.observed_seconds == 120.0
    assert result.age_seconds == 120.0
    assert result.volume_sol == pytest.approx(10.0)
    assert result.liquidity_usd == pytest.approx(33.0 * 200.0)


def test_watch_tracks_price_change_from_the_curve():
    token = parse_create_event(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)
    # curve moves from 32/900M to 64/880M -> price roughly doubles
    watch.record(trade(curve_sol=64.0, curve_tokens=880000000.0))
    assert watch.result(200.0, now=10.0).price_change_pct > 1.0


def test_watch_flags_the_deployer_selling():
    token = parse_create_event(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)
    watch.record(trade(side="sell", trader="DEV1"))
    assert watch.dev_sold is True
    assert watch.result(200.0, now=5.0).dev_sold is True


def test_watch_ignores_unknown_event_types():
    watch = Watch(parse_create_event(CREATE_EVENT), window_seconds=300, now=0.0)
    watch.record({"txType": "migrate", "traderPublicKey": "X"})
    assert watch.buys == 0 and watch.sells == 0 and watch.traders == set()


def test_watch_expiry():
    watch = Watch(parse_create_event(CREATE_EVENT), window_seconds=300, now=0.0)
    assert watch.expired(now=299.0) is False
    assert watch.expired(now=300.0) is True


# --- the scout ---------------------------------------------------------------------

def scout(**over) -> Scout:
    config = {
        "crypto_launch_filter": LAUNCH_FILTER,
        "crypto_filter": WATCH_FILTER,
        "pump_fun": {
            "sol_price_usd": 200.0,
            "watch": {"enabled": True, "window_seconds": 300,
                      "max_concurrent": 3, "min_trades_to_score": 12},
        },
    }
    config["pump_fun"]["watch"].update(over)
    return Scout(config)


def test_handle_create_admits_then_dedups():
    s = scout()
    assert s.handle_create(CREATE_EVENT).mint == "MINT1"
    assert s.handle_create(CREATE_EVENT) is None


def test_handle_create_rejects_a_bad_launch():
    s = scout()
    assert s.handle_create({**CREATE_EVENT, "solAmount": 50.0}) is None


def test_watchlist_capacity_is_enforced():
    s = scout()
    for i in range(3):
        token = s.handle_create({**CREATE_EVENT, "mint": f"M{i}"})
        s.watching[token.mint] = Watch(token, 300, now=0.0)
    # the 4th create is dropped: trade subscriptions are metered
    assert s.handle_create({**CREATE_EVENT, "mint": "M99"}) is None


async def test_mature_returns_only_survivors_of_the_window():
    s = scout()
    # Without enricher, require_mint_revoked would kill every token — leave it off
    # here so this test stays about the watch window + trade filter.
    s.filter = {k: v for k, v in s.filter.items() if k != "require_mint_revoked"}
    token = s.handle_create(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)
    s.watching[token.mint] = watch
    for i in range(30):
        watch.record(trade(trader=f"T{i}"))

    assert await s.mature(now=100.0) == []            # window still open
    ready = await s.mature(now=400.0)
    assert [t.mint for t in ready] == ["MINT1"]
    assert ready[0].buys == 30
    assert s.watching == {}                      # evicted


async def test_mature_drops_a_token_with_too_little_trading():
    s = scout()
    token = s.handle_create(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)
    s.watching[token.mint] = watch
    watch.record(trade())                        # 1 trade, below min_trades_to_score
    assert await s.mature(now=400.0) == []


async def test_mature_evicts_immediately_when_the_deployer_sells():
    s = scout()
    token = s.handle_create(CREATE_EVENT)
    watch = Watch(token, window_seconds=300, now=0.0)
    s.watching[token.mint] = watch
    watch.record(trade(side="sell", trader="DEV1"))
    assert await s.mature(now=1.0) == []               # before the window closes
    assert s.watching == {}


def test_handle_trade_routes_by_mint():
    s = scout()
    token = s.handle_create(CREATE_EVENT)
    s.watching[token.mint] = Watch(token, 300, now=0.0)
    s.handle_trade(trade(mint="MINT1"))
    s.handle_trade(trade(mint="SOMETHING_ELSE"))
    assert s.watching["MINT1"].buys == 1


def test_decode_rejects_junk():
    s = scout()
    assert s._decode("not json") is None
    assert s._decode(json.dumps([1, 2, 3])) is None
    assert s._decode(json.dumps({"a": 1})) == {"a": 1}


async def test_on_message_end_to_end_through_the_socket():
    s = scout()
    sent: list[dict] = []

    class FakeWS:
        async def send(self, raw):
            sent.append(json.loads(raw))

    s._ws = FakeWS()

    assert await s._on_message(json.dumps(CREATE_EVENT)) == []
    assert sent[-1] == {"method": "subscribeTokenTrade", "keys": ["MINT1"]}

    for i in range(30):
        await s._on_message(json.dumps(trade(trader=f"T{i}")))

    s.watching["MINT1"].started_at -= 400        # force the window closed
    ready = await s._on_message(json.dumps(trade(trader="T99")))
    assert [t.mint for t in ready] == ["MINT1"]
    assert sent[-1] == {"method": "unsubscribeTokenTrade", "keys": ["MINT1"]}


async def test_watch_disabled_yields_immediately():
    s = scout(enabled=False)
    ready = await s._on_message(json.dumps(CREATE_EVENT))
    assert [t.mint for t in ready] == ["MINT1"]
    assert ready[0].liquidity_usd == pytest.approx(6400.0)


# --- anti-chase via price_change_pct (improve-memecoin-exits) -----------------

ANTI_CHASE_FILTER = {
    **WATCH_FILTER,
    "max_window_pump_pct": 0.80,
    "min_window_pump_pct": -0.15,
}


def test_anti_chase_rejects_window_pump_above_threshold():
    assert filter_reason(
        watched_token(price_change_pct=0.81), ANTI_CHASE_FILTER
    ) == "window_pump_too_high"
    # exclusive above: exactly 0.80 still passes
    assert filter_reason(
        watched_token(price_change_pct=0.80), ANTI_CHASE_FILTER
    ) is None


def test_anti_chase_rejects_window_dump_below_threshold():
    assert filter_reason(
        watched_token(price_change_pct=-0.16), ANTI_CHASE_FILTER
    ) == "window_dump_too_hard"
    assert filter_reason(
        watched_token(price_change_pct=-0.15), ANTI_CHASE_FILTER
    ) is None


def test_anti_chase_absent_keys_do_not_reject():
    # Old configs without the keys keep working — no silent anti-chase.
    assert filter_reason(watched_token(price_change_pct=5.0), WATCH_FILTER) is None
    assert filter_reason(watched_token(price_change_pct=-0.99), WATCH_FILTER) is None


def test_anti_chase_moderate_move_passes():
    assert filter_reason(
        watched_token(price_change_pct=0.40), ANTI_CHASE_FILTER
    ) is None
