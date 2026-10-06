"""Memecoin playbook: scorecard, regime gate, staged entry, exit ladder,
dump detection, toxic flow, flow tracker and paper fill simulation."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from src.crypto.crypto_executor import (
    BUY_DISCRIMINATOR,
    SELL_DISCRIMINATOR,
    CryptoExecutor,
    LiveTradingDisabled,
    PaperFillSimulator,
    choose_venue,
    curve_buy_quote,
    curve_sell_quote,
    derive_bonding_curve,
    encode_curve_buy,
    scrub_config,
)
from src.crypto.dump_detector import detect_catastrophic, detect_dump, flow_flipped
from src.crypto.entry_scorecard import EntryScorecard, buys_per_minute, ramp
from src.crypto.exit_ladder import ExitLadder, ExitSignal
from src.crypto.flow import FlowStats, FlowTracker
from src.crypto.regime import RegimeGate
from src.crypto.scout import Scout
from src.crypto.staged_entry import StagedEntry
from src.crypto.toxic_flow import ToxicFlowFilter, fingerprint, normalize
from src.desk import TradingDesk
from src.models import Market, Position, Token
from tests.conftest import FakeClient
from tests.test_desk import GOOD, TOKEN

REAL_MINT = "So11111111111111111111111111111111111111112"
NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


def load_config(tmp_path) -> dict:
    config = yaml.safe_load(open("config.example.yaml"))
    config["logging"] = {"path": str(tmp_path / "desk.jsonl"), "echo_stdout": False,
                         "cost_report_every": 0}
    return config


def build(tmp_path, overrides=None, config=None, **kwargs) -> TradingDesk:
    desk = TradingDesk(config or load_config(tmp_path), dry_run=kwargs.pop("dry_run", True), **kwargs)
    for name, reply in {**GOOD, **(overrides or {})}.items():
        getattr(desk, name)._client = FakeClient([reply])
    return desk


def records(tmp_path) -> list[dict]:
    return [json.loads(line) for line in open(tmp_path / "desk.jsonl")]


def crypto_pos(**over) -> Position:
    base = dict(market=Market.CRYPTO, symbol="WIF2", quantity=1000.0, entry_price=1.0,
                current_price=1.0, amount_usd=1000.0, stop_price=0.90,
                take_profit_price=2.5, peak_price=1.0, opened_at=NOW, meta={"mint": "M1"})
    base.update(over)
    return Position(**base)


def ladder_pos(ladder: ExitLadder, **over) -> Position:
    pos = crypto_pos(**over)
    pos.meta["ladder"] = ladder.init_state(pos.quantity)
    return pos


# =============================================================================
# Entry scorecard
# =============================================================================

GOOD_NARR = GOOD["narrative"]
GOOD_AUDIT = GOOD["auditor"]


def test_ramp_handles_normal_and_inverted_ranges():
    assert ramp(5, 0, 10) == 0.5
    assert ramp(-1, 0, 10) == 0.0 and ramp(99, 0, 10) == 1.0
    assert ramp(0.10, 0.45, 0.20) == 1.0   # lower concentration is better
    assert ramp(0.45, 0.45, 0.20) == 0.0


def test_buys_per_minute_prefers_observed_window():
    t = Token(mint="M", buys=60, observed_seconds=120, age_seconds=600)
    assert buys_per_minute(t) == pytest.approx(30.0)
    assert buys_per_minute(Token(mint="M", buys=60, age_seconds=600)) == pytest.approx(6.0)
    assert buys_per_minute(Token(mint="M", buys=60)) == 0.0


def test_strong_token_passes_with_grade_a():
    card = EntryScorecard({}).score(TOKEN, GOOD_NARR, GOOD_AUDIT)
    assert card["pass"] is True and card["reason"] == "scorecard_ok"
    assert card["grade"] == "A" and card["size_mult"] == 1.0
    assert set(card["factors"]) == {"velocity", "holder_quality", "liquidity", "social_heat", "audit"}


def test_slow_tape_is_killed_by_velocity_floor():
    slow = TOKEN.model_copy(update={"buys": 3, "sells": 2, "age_seconds": 600})
    card = EntryScorecard({}).score(slow, GOOD_NARR, GOOD_AUDIT)
    assert card["pass"] is False and card["reason"] == "scorecard_weak_velocity"


def test_wash_churn_lowers_holder_quality():
    sc = EntryScorecard({})
    organic = TOKEN.model_copy(update={"buys": 90, "sells": 25, "unique_traders": 60})
    churn = TOKEN.model_copy(update={"buys": 90, "sells": 25, "unique_traders": 13})
    assert sc.holder_quality(churn) < sc.holder_quality(organic) - 0.3


def test_unknown_concentration_is_neutral_not_perfect():
    sc = EntryScorecard({})
    unknown = Token(mint="M", buys=50, sells=10, unique_traders=40)
    clean = unknown.model_copy(update={"holders_known": True, "holders": 250,
                                       "top10_holder_pct": 0.10, "dev_holding_pct": 0.0})
    concentrated = clean.model_copy(update={"top10_holder_pct": 0.60, "dev_holding_pct": 0.2})
    assert sc.holder_quality(concentrated) < sc.holder_quality(unknown) < sc.holder_quality(clean)


def test_social_heat_penalises_derivatives_and_rewards_socials():
    sc = EntryScorecard({})
    t = Token(mint="M")
    base = sc.social_heat(t, GOOD_NARR)
    assert sc.social_heat(t, {**GOOD_NARR, "is_derivative": True}) < base
    with_socials = t.model_copy(update={"socials": {"twitter": "x", "telegram": "t"}})
    assert sc.social_heat(with_socials, GOOD_NARR) == pytest.approx(base + 0.06, abs=1e-4)


def test_audit_factor_kill_floor():
    shaky = {**GOOD_AUDIT, "safety_score": 0.45, "red_flags": ["a", "b"]}
    card = EntryScorecard({}).score(TOKEN, GOOD_NARR, shaky)
    assert card["reason"] == "scorecard_weak_audit"


def test_regime_add_raises_the_threshold():
    sc = EntryScorecard({})
    card = sc.score(TOKEN, GOOD_NARR, GOOD_AUDIT)
    harsh = sc.score(TOKEN, GOOD_NARR, GOOD_AUDIT, min_score_add=1.0 - card["threshold"] + 0.01)
    assert harsh["pass"] is False and harsh["reason"] == "scorecard_below_threshold"


def test_prescreen_uses_code_factors_only():
    sc = EntryScorecard({})
    assert sc.prescreen(TOKEN) is None
    thin = TOKEN.model_copy(update={"liquidity_usd": 2000.0})
    assert sc.prescreen(thin) == "scorecard_weak_liquidity"
    sc.enabled = False
    assert sc.prescreen(thin) is None


async def test_desk_prescreen_rejects_before_any_model_call(tmp_path):
    desk = build(tmp_path)
    slow = TOKEN.model_copy(update={"buys": 2, "sells": 2, "age_seconds": 900})
    result = await desk.evaluate_token(slow)
    assert result["reason"] == "scorecard_weak_velocity"
    for bot in ("crypto_pulse", "auditor", "narrative", "crypto_checker"):
        assert getattr(desk, bot)._client.calls == [], bot


async def test_desk_scorecard_rejection_skips_the_checker(tmp_path):
    desk = build(tmp_path, {"narrative": {**GOOD_NARR, "meme_score": 0.1, "virality": 0.05,
                                          "community_signal": 0.05}})
    desk.weights = {"crypto": {"min_score_to_buy": 0.1}}   # let the matrix through
    result = await desk.evaluate_token(TOKEN)
    assert result["reason"] == "scorecard_weak_social_heat"
    assert desk.crypto_checker._client.calls == []


# =============================================================================
# Regime gate
# =============================================================================

def test_regime_states():
    gate = RegimeGate({})
    assert gate.assess({"regime": "risk_on", "go_signal": 0.8}).state == "normal"
    caution = gate.assess({"regime": "risk_on", "go_signal": 0.4})
    assert caution.state == "caution" and caution.size_mult == 0.6 and caution.min_score_add == 0.05
    assert gate.assess({"regime": "neutral", "go_signal": 0.7}).state == "caution"
    dump = gate.assess({"regime": "risk_on", "go_signal": 0.8, "sol_trend": "down"})
    assert dump.state == "defensive" and dump.reason == "regime_sol_dump" and dump.size_mult == 0.4
    assert gate.assess({"regime": "risk_off", "go_signal": 0.6}).reason == "regime_risk_off"
    paused = gate.assess({"regime": "risk_on", "go_signal": 0.4, "sol_trend": "down"})
    assert paused.allow_entries is False and paused.reason == "regime_paused"
    hard = gate.assess({"regime": "risk_on", "go_signal": 0.1})
    assert hard.allow_entries is False and hard.reason == "veto_market_paused"


def test_regime_measured_sol_dump():
    gate = RegimeGate({})
    t0 = 1_000_000.0
    gate.observe_sol_price(200.0, t0)
    gate.observe_sol_price(190.0, t0 + 1800)
    assert gate.sol_change_pct(now=t0 + 1800) == pytest.approx(-0.05)
    assert gate.assess({"regime": "risk_on", "go_signal": 0.8}, now=t0 + 1800).state == "defensive"
    # Outside the lookback window there is no measurement.
    assert gate.sol_change_pct(now=t0 + 3 * 3600) is None


def test_regime_pause_on_weak_and_dump_is_configurable():
    gate = RegimeGate({"regime": {"pause_on_weak_and_dump": False}})
    d = gate.assess({"regime": "risk_on", "go_signal": 0.4, "sol_trend": "down"})
    assert d.allow_entries is True and d.state == "defensive"


async def test_desk_regime_pause_skips_before_audit(tmp_path):
    desk = build(tmp_path, {"crypto_pulse": {"regime": "neutral", "go_signal": 0.45,
                                             "sol_trend": "down", "risk_appetite": 0.3}})
    result = await desk.evaluate_token(TOKEN)
    assert result["reason"] == "regime_paused"
    assert desk.auditor._client.calls == [] and desk.narrative._client.calls == []


async def test_desk_regime_shrinks_size(tmp_path):
    normal = build(tmp_path / "a")
    defensive = build(tmp_path / "b", {"crypto_pulse": {"regime": "risk_on", "go_signal": 0.9,
                                                        "sol_trend": "down", "risk_appetite": 0.6}})
    a = await normal.evaluate_token(TOKEN)
    b = await defensive.evaluate_token(TOKEN)
    assert a["bought"] and b["bought"]
    assert b["planned"] == pytest.approx(a["planned"] * 0.4, abs=0.02)


async def test_desk_regime_tightens_matrix_threshold(tmp_path):
    desk = build(tmp_path, {"crypto_pulse": {"regime": "risk_on", "go_signal": 0.9,
                                             "sol_trend": "down", "risk_appetite": 0.6}})
    desk.weights = {"crypto": {"min_score_to_buy": 0.80}}
    # matrix score for TOKEN (~0.87) clears 0.80 but not 0.90 (= 0.80 + defensive 0.10)
    result = await desk.evaluate_token(TOKEN)
    assert result["reason"] == "below_regime_threshold"
    assert desk.crypto_checker._client.calls == []


# =============================================================================
# Flow tracker
# =============================================================================

def trade(side="buy", trader="T", sol=0.5, v_sol=30.0, v_tok=1e9, mint="M1"):
    return {"mint": mint, "txType": side, "traderPublicKey": trader, "solAmount": sol,
            "vSolInBondingCurve": v_sol, "vTokensInBondingCurve": v_tok}


def test_flow_tracker_window_stats_and_dev_sell():
    ft = FlowTracker()
    assert ft.record(trade(), now=0) is False  # untracked mints are ignored
    ft.track("M1", creator="DEV")
    for i in range(5):
        ft.record(trade(trader=f"B{i}", v_sol=30 + i), now=100 + i)
    ft.record(trade("sell", trader="DEV", v_sol=28), now=110)
    st = ft.stats("M1", 60, now=120)
    assert st.buys == 5 and st.sells == 1 and st.unique_buyers == 5
    assert st.dev_sold is True
    assert st.high == pytest.approx(34e-9) and st.last == pytest.approx(28e-9)
    assert st.drop_from_high == pytest.approx(6 / 34, abs=1e-6)
    assert ft.stats("M1", 5, now=120).buys == 0
    assert ft.reserves("M1") == (28.0, 1e9)
    ft.untrack("M1")
    assert ft.last_price("M1") == 0.0


# =============================================================================
# Dump detection
# =============================================================================

def flow(**kw) -> FlowStats:
    return FlowStats(**{"buys": 3, "sells": 3, "buy_volume_sol": 1.0, "sell_volume_sol": 1.0,
                        "high": 1.0, "last": 1.0, **kw})


def test_flip_plus_crash_is_an_emergency():
    f = flow(buys=2, sells=10, sell_volume_sol=8.0, high=1.0, last=0.82)
    assert flow_flipped(f, {"min_sells": 6, "flip_sell_buy_ratio": 2.0, "flip_volume_ratio": 2.0})
    assert detect_dump(f, 0.82, peak=1.0) == "emergency_dump"


def test_flip_without_crash_or_crash_without_flip_is_not():
    flipped_flat = flow(buys=2, sells=10, sell_volume_sol=8.0, high=1.0, last=0.97)
    assert detect_dump(flipped_flat, 0.97, peak=1.0) is None
    crash_bid = flow(buys=20, sells=5, buy_volume_sol=10.0, sell_volume_sol=3.0, high=1.0, last=0.8)
    assert detect_dump(crash_bid, 0.80, peak=1.0) is None


def test_volume_flip_counts_even_with_few_large_sells():
    f = flow(buys=6, sells=4, buy_volume_sol=1.0, sell_volume_sol=9.0, high=1.0, last=0.8)
    assert detect_dump(f, 0.80, peak=1.0) == "emergency_dump"


def test_peak_crash_counts_as_crash():
    f = flow(buys=1, sells=8, sell_volume_sol=5.0, high=0.80, last=0.74)
    # window drop only 7.5% but 26% off the 1.0 peak
    assert detect_dump(f, 0.74, peak=1.0) == "emergency_dump"


def test_dev_sell_and_catastrophic():
    assert detect_dump(flow(dev_sold=True), 1.0) == "emergency_dev_dump"
    assert detect_catastrophic(flow(high=1.0, last=0.6), 0.6) == "emergency_crash"
    assert detect_catastrophic(None, 0.6, peak=1.0) == "emergency_crash"
    assert detect_catastrophic(None, 0.8, peak=1.0) is None
    assert detect_dump(flow(dev_sold=True), 1.0, cfg={"enabled": False}) is None


# =============================================================================
# Exit ladder
# =============================================================================

def test_first_rung_trims_a_third_and_raises_stop_to_breakeven():
    ladder = ExitLadder({})
    pos = ladder_pos(ladder)
    sig = ladder.evaluate(pos, 1.16, now=NOW)
    assert sig.action == "TRIM" and sig.reason == "tp_rung_1" and sig.rungs == (0,)
    assert sig.fraction == pytest.approx(0.33)
    assert sig.new_stop == pytest.approx(1.01)
    pos.quantity *= 1 - sig.fraction
    ladder.apply_trim(pos, sig)
    assert pos.stop_price == pytest.approx(1.01) and pos.meta["ladder"]["rungs_hit"] == [0]
    # Second rung: 33% of BASE qty out of the 67% left.
    sig2 = ladder.evaluate(pos, 1.31, now=NOW)
    assert sig2.reason == "tp_rung_2" and sig2.fraction == pytest.approx(0.33 / 0.67, abs=1e-4)


def test_gap_through_both_rungs_sells_them_together():
    ladder = ExitLadder({})
    sig = ladder.evaluate(ladder_pos(ladder), 1.40, now=NOW)
    assert sig.action == "TRIM" and sig.rungs == (0, 1) and sig.fraction == pytest.approx(0.66)


def test_runner_trails_after_rungs_and_breakeven_stop_label():
    ladder = ExitLadder({})
    pos = ladder_pos(ladder)
    sig = ladder.evaluate(pos, 1.40, now=NOW)
    pos.quantity *= 1 - sig.fraction
    ladder.apply_trim(pos, sig)
    assert ladder.evaluate(pos, 1.60, now=NOW).action == "HOLD"     # new peak
    trail = ladder.evaluate(pos, 1.60 * 0.87, now=NOW)              # 13% off peak
    assert trail.action == "CLOSE" and trail.reason == "runner_trail"
    assert ladder.evaluate(pos, 1.005, now=NOW).reason == "breakeven_stop"


def test_hard_stop_runner_cap_stale_and_time_stop():
    ladder = ExitLadder({"exits": {"max_hold_minutes": 60, "ladder": {"stale_minutes": 20}}})
    pos = ladder_pos(ladder)
    assert ladder.evaluate(pos, 0.89, now=NOW).reason == "stop_loss"
    assert ladder.evaluate(ladder_pos(ladder), 2.6, now=NOW).reason == "take_profit"
    stale = ladder.evaluate(ladder_pos(ladder), 1.02, now=NOW + timedelta(minutes=25))
    assert stale.reason == "stale_exit"
    # A position that ran +20% earlier is not stale (rung would have fired; simulate peak only)
    ran = ladder_pos(ladder, peak_price=1.12)
    assert ladder.evaluate(ran, 1.03, now=NOW + timedelta(minutes=25)).action == "HOLD"
    old = ladder_pos(ladder, peak_price=1.12)
    assert ladder.evaluate(old, 1.03, now=NOW + timedelta(minutes=61)).reason == "time_stop"


def test_emergency_dump_beats_everything_else():
    ladder = ExitLadder({})
    pos = ladder_pos(ladder)
    f = flow(buys=1, sells=12, sell_volume_sol=10.0, high=1.30, last=1.05)
    sig = ladder.evaluate(pos, 1.05, flow=f, now=NOW)
    assert sig.action == "CLOSE" and sig.emergency and sig.reason == "emergency_dump"


def test_positions_without_ladder_state_keep_legacy_plan():
    ladder = ExitLadder({"exits": {"trailing_stop_activate_pct": 0.25, "trailing_stop_pct": 0.10}})
    legacy = crypto_pos(stop_price=None, take_profit_price=None)
    assert ladder.evaluate(legacy, 2.0, now=NOW).action == "HOLD"   # no rungs
    assert ladder.evaluate(legacy, 1.75, now=NOW).reason == "trailing_stop"


def test_dust_remainder_closes_instead_of_trimming():
    ladder = ExitLadder({"exits": {"ladder": {"rungs": [{"gain_pct": 0.1, "sell_fraction": 0.97}]}}})
    sig = ladder.evaluate(ladder_pos(ladder), 1.2, now=NOW)
    assert sig.action == "CLOSE" and sig.reason == "tp_rung_1"


# =============================================================================
# Staged entry
# =============================================================================

def staged_pos(se: StagedEntry, opened=NOW, **over) -> Position:
    pos = crypto_pos(opened_at=opened, amount_usd=40.0, quantity=40.0, **over)
    pos.meta["staged"] = se.init_state(100.0, 40.0, 60.0, 1.0, opened)
    return pos


def good_flow(**kw) -> FlowStats:
    return FlowStats(**{"buys": 12, "sells": 4, "unique_buyers": 9, **kw})


def test_split_40_60_and_disabled():
    assert StagedEntry({}).split(100.0) == (40.0, 60.0)
    assert StagedEntry({"staged_entry": {"enabled": False}}).split(100.0) == (100.0, 0.0)


def test_add_requires_gain_and_volume():
    se = StagedEntry({})
    pos = staged_pos(se)
    later = NOW + timedelta(minutes=3)
    assert se.evaluate_add(pos, 1.04, good_flow(), later).reason == "awaiting_gain"
    assert se.evaluate_add(pos, 1.10, None, later).reason == "awaiting_volume"
    assert se.evaluate_add(pos, 1.10, good_flow(buys=3), later).reason == "awaiting_volume"
    assert se.evaluate_add(pos, 1.10, good_flow(sells=11), later).reason == "weak_flow"
    assert se.evaluate_add(pos, 1.10, good_flow(unique_buyers=2), later).reason == "too_few_buyers"
    assert se.evaluate_add(pos, 1.45, good_flow(), later).reason == "add_chase_guard"
    ok = se.evaluate_add(pos, 1.10, good_flow(), later)
    assert ok.action == "add" and ok.gain == pytest.approx(0.10)


def test_add_expires_after_window_on_dev_sell_or_after_rung():
    se = StagedEntry({})
    assert se.evaluate_add(staged_pos(se), 1.1, good_flow(), NOW + timedelta(minutes=11)).action == "expire"
    assert se.evaluate_add(staged_pos(se), 1.1, good_flow(dev_sold=True), NOW).reason == "dev_sold"
    pos = staged_pos(se)
    pos.meta["ladder"] = {"rungs_hit": [0]}
    assert se.evaluate_add(pos, 1.1, good_flow(), NOW).reason == "ladder_already_trimming"
    se.expire(pos, "x")
    assert se.evaluate_add(pos, 1.1, good_flow(), NOW).action == "skip"


def test_apply_add_averages_entry_and_resets_ladder_base():
    se = StagedEntry({})
    pos = staged_pos(se)
    pos.meta["ladder"] = {"base_qty": 40.0, "rungs_hit": []}
    se.apply_add(pos, 60.0, 1.10, NOW)
    assert pos.amount_usd == pytest.approx(100.0)
    assert pos.quantity == pytest.approx(40.0 + 60.0 / 1.10)
    assert pos.entry_price == pytest.approx(100.0 / (40.0 + 60.0 / 1.10))
    assert pos.meta["staged"]["stage"] == "full"
    assert pos.meta["ladder"]["base_qty"] == pytest.approx(pos.quantity)


async def test_desk_opens_probe_then_adds_on_confirmation(tmp_path):
    desk = build(tmp_path)
    result = await desk.evaluate_token(TOKEN)
    assert result["bought"] and result["amount"] == pytest.approx(result["planned"] * 0.4, abs=0.01)
    pos = desk.positions[0]
    assert pos.meta["staged"]["stage"] == "probe"
    assert desk.scout.flow.is_tracked("M1")
    deployed_probe = desk.risk.deployed_usd[Market.CRYPTO]

    # No confirmation yet → nothing happens.
    assert await desk.check_staged_adds(prices={"M1": pos.entry_price * 1.02}) == []

    # Tape confirms: +10% with real buyers.
    up = pos.entry_price * 1.10
    now_ts = time.time()
    for i in range(10):
        desk.scout.flow.record(trade(trader=f"B{i}", v_sol=up * 1e9, v_tok=1e9), now=now_ts)
    events = await desk.check_staged_adds()
    assert events and events[0]["action"] == "ADD"
    assert pos.meta["staged"]["stage"] == "full"
    assert pos.amount_usd == pytest.approx(result["planned"], abs=0.02)
    assert desk.risk.deployed_usd[Market.CRYPTO] == pytest.approx(deployed_probe + events[0]["add_usd"])
    assert pos.stop_price == pytest.approx(pos.entry_price * (1 - desk.exits_cfg["stop_loss_pct"]))
    assert any(r.get("action") == "ADD" for r in records(tmp_path))


async def test_desk_add_expires_after_window(tmp_path):
    desk = build(tmp_path)
    await desk.evaluate_token(TOKEN)
    events = await desk.check_staged_adds(now=datetime.now(timezone.utc) + timedelta(minutes=30))
    assert events == [{"action": "ADD_EXPIRED", "reason": "add_window_expired"}]
    assert desk.positions[0].meta["staged"]["stage"] == "probe_only"


# =============================================================================
# Desk: ladder, dump detection, toxic blacklist end to end
# =============================================================================

async def test_desk_ladder_trim_realizes_pnl_then_close_reports_total(tmp_path):
    desk = build(tmp_path)
    pos = ladder_pos(desk.ladder, amount_usd=100.0, quantity=100.0, stop_price=0.9)
    desk.positions = [pos]
    desk.risk.record_fill(Market.CRYPTO, 100.0)

    out = await desk.check_open_crypto_stops(prices={"M1": 1.20})
    assert out[0]["action"] == "TRIM" and out[0]["trimmed"]
    assert pos.quantity == pytest.approx(67.0)
    assert pos.amount_usd == pytest.approx(67.0)
    assert pos.stop_price == pytest.approx(1.01)
    realized = desk.risk.realized_pnl_today
    assert 0 < realized < 33 * 0.20                      # slippage + fees bite
    assert desk.risk.deployed_usd[Market.CRYPTO] == pytest.approx(67.0)

    out = await desk.check_open_crypto_stops(prices={"M1": 1.0})   # breakeven stop
    assert out[0]["reason"] == "breakeven_stop" and desk.positions == []
    close = [r for r in records(tmp_path) if r["type"] == "close"][-1]
    assert close["pnl"] == pytest.approx(desk.risk.realized_pnl_today, abs=0.01)
    assert desk.risk.deployed_usd[Market.CRYPTO] == pytest.approx(0.0)


async def test_desk_dump_detection_closes_and_blacklists_deployer(tmp_path):
    desk = build(tmp_path)
    await desk.evaluate_token(TOKEN.model_copy(update={"creator": "RUGGER"}))
    pos = desk.positions[0]
    entry = pos.entry_price
    now_ts = time.time()
    desk.scout.flow.record(trade(trader="B1", v_sol=entry * 1.1e9, v_tok=1e9), now=now_ts - 30)
    for i in range(10):
        desk.scout.flow.record(trade("sell", trader=f"S{i}", sol=2.0,
                                     v_sol=entry * 0.85e9, v_tok=1e9), now=now_ts - 5)
    out = await desk.check_open_crypto_stops()
    assert out[0]["reason"] == "emergency_dump" and out[0]["emergency"] is True
    assert desk.positions == [] and not desk.scout.flow.is_tracked("M1")
    assert "RUGGER" in desk.toxic.blacklist
    assert any(r["type"] == "blacklist" and r["creator"] == "RUGGER" for r in records(tmp_path))

    # The same deployer's next launch dies before any model call.
    nxt = TOKEN.model_copy(update={"mint": "M2", "creator": "RUGGER"})
    calls_before = len(desk.auditor._client.calls)
    assert (await desk.evaluate_token(nxt))["reason"] == "toxic_deployer_blacklisted"
    assert len(desk.auditor._client.calls) == calls_before

    # And the blacklist survives a restart via the event log.
    reborn = build(tmp_path)
    assert "RUGGER" in reborn.toxic.blacklist


async def test_desk_legacy_mode_when_ladder_disabled(tmp_path):
    config = load_config(tmp_path)
    config["exits"]["ladder"]["enabled"] = False
    desk = build(tmp_path, config=config)
    pos = crypto_pos(amount_usd=100.0, quantity=100.0, take_profit_price=1.2)
    desk.positions = [pos]
    out = await desk.check_open_crypto_stops(prices={"M1": 1.21})
    assert out[0]["reason"] == "take_profit"


# =============================================================================
# Toxic flow
# =============================================================================

def tok(mint, creator="DEV", name="Doge Wif Hat", symbol="DWH", uri=None) -> Token:
    return Token(mint=mint, creator=creator, name=name, symbol=symbol, uri=uri or f"ipfs://{mint}")


def test_normalize_and_fingerprint():
    assert normalize("D0GE-W1F!") == "dogewif"
    assert fingerprint(tok("A", name="Doge Wif", symbol="$DWH")) == fingerprint(
        tok("B", name="d0ge_wif", symbol="dwh"))


def test_serial_deployer():
    f = ToxicFlowFilter({})
    for i, m in enumerate(["A", "B", "C"]):
        t = tok(m, name=f"n{i}", symbol=f"s{i}")
        f.observe(t, now=100 + i)
        verdict = f.reason(t, now=100 + i)
    assert verdict == "toxic_serial_deployer"


def test_uri_reuse_and_same_deployer_clone():
    f = ToxicFlowFilter({})
    a = tok("A", creator="X", uri="ipfs://same")
    b = tok("B", creator="Y", name="other", symbol="oth", uri="ipfs://same")
    f.observe(a, now=1); f.observe(b, now=2)
    assert f.reason(a, now=2) is None
    assert f.reason(b, now=2) == "toxic_uri_reuse"
    c = tok("C", creator="X")
    f.observe(c, now=3)
    assert f.reason(c, now=3) == "toxic_clone_same_deployer"


def test_metadata_clone_threshold_and_window():
    f = ToxicFlowFilter({"crypto_toxic": {"max_clones": 3}})
    for i in range(3):
        t = tok(f"M{i}", creator=f"C{i}")
        f.observe(t, now=10 + i)
        assert f.reason(t, now=10 + i) is None
    fourth = tok("M3", creator="C3")
    f.observe(fourth, now=20)
    assert f.reason(fourth, now=20) == "toxic_metadata_clone"
    # After the clone window the old copies age out.
    late = tok("M9", creator="C9")
    f.observe(late, now=20 + 4 * 3600)
    assert f.reason(late, now=20 + 4 * 3600) is None


def test_blacklist_seed_and_rules():
    f = ToxicFlowFilter({})
    assert f.seed_from_log([{"type": "blacklist", "creator": "BAD", "reason": "emergency_dump"},
                            {"type": "close", "creator": "OK"}]) == 1
    assert f.entry_reason(tok("Z", creator="BAD")) == "toxic_deployer_blacklisted"
    assert f.should_blacklist("emergency_dev_dump") and not f.should_blacklist("stop_loss")


def test_scout_stage_one_rejects_toxic_launches():
    from tests.test_scout_filter import CREATE_EVENT, LAUNCH_FILTER

    scout = Scout({"crypto_launch_filter": LAUNCH_FILTER, "crypto_toxic": {"enabled": True}})
    assert scout.handle_create(CREATE_EVENT) is not None
    clone = {**CREATE_EVENT, "mint": "MINT2", "traderPublicKey": "DEV9"}  # same uri
    assert scout.handle_create(clone) is None
    assert scout.reject_counts == {"toxic_uri_reuse": 1}
    # Without the section the scout behaves exactly as before.
    assert Scout({"crypto_launch_filter": LAUNCH_FILTER}).toxic is None


async def test_scout_keeps_subscription_for_tracked_positions():
    sent: list[dict] = []

    class WS:
        async def send(self, msg):
            sent.append(json.loads(msg))

    scout = Scout({})
    scout._ws = WS()
    await scout.track_position("M1", "DEV")
    assert sent[-1] == {"method": "subscribeTokenTrade", "keys": ["M1"]}
    scout.handle_trade(trade())
    assert scout.flow.stats("M1", 60).buys == 1
    await scout.untrack_position("M1")
    assert sent[-1] == {"method": "unsubscribeTokenTrade", "keys": ["M1"]}


# =============================================================================
# Executor + paper fills
# =============================================================================

def test_curve_quotes_respect_constant_product():
    v_sol, v_tok = 30.0, 1_073_000_000.0
    q = curve_buy_quote(1.0, v_sol, v_tok, fee_bps=100)
    net = 0.99
    assert (v_sol + net) * (v_tok - q["tokens_out"]) == pytest.approx(v_sol * v_tok, rel=1e-9)
    assert q["avg_price"] > q["spot_price"] and q["impact_pct"] > 0.01
    s = curve_sell_quote(q["tokens_out"], v_sol + net, v_tok - q["tokens_out"], fee_bps=100)
    assert s["sol_out"] < 1.0   # round trip loses fees + impact
    assert curve_buy_quote(0, v_sol, v_tok)["tokens_out"] == 0.0


def test_instruction_data_and_venue():
    data = encode_curve_buy(1_000_000, 50_000_000)
    assert data[:8] == BUY_DISCRIMINATOR == bytes([102, 6, 61, 18, 1, 218, 235, 234])
    assert SELL_DISCRIMINATOR == bytes([51, 230, 133, 164, 1, 127, 131, 173])
    assert len(data) == 24
    assert choose_venue(False) == "pump_curve" and choose_venue(True) == "amm"
    assert derive_bonding_curve(REAL_MINT) and derive_bonding_curve("M1") is None


def test_paper_fill_simulator_buy_and_sell():
    sim = PaperFillSimulator({"solana": {"slippage_bps": 500, "priority_fee_microlamports": 200000,
                                         "jito": {"enabled": True, "tip_lamports": 100000}},
                              "pump_fun": {"sol_price_usd": 200.0}})
    spot = 32.0 / 800_000_000.0
    buy = sim.buy("M1", 100.0, spot, reserves=(32.0, 800_000_000.0))
    assert buy["filled"] and buy["simulated"] and buy["tx_id"].startswith("PAPER-")
    assert buy["price"] > spot and buy["quantity"] == pytest.approx(100.0 / buy["price"])
    assert buy["fees_usd"] == pytest.approx((5000 + 24000 + 100000) / 1e9 * 200.0)
    no_quote = sim.buy("M1", 100.0, 0.0)
    assert no_quote["filled"] is False and no_quote["reason"] == "no_quote_price"
    huge = sim.buy("M1", 2000.0, spot, reserves=(32.0, 800_000_000.0))
    assert huge["filled"] is False and huge["reason"] == "slippage_exceeded"
    sell = sim.sell("M1", buy["quantity"], spot * 1.2, reserves=(33.0, 780_000_000.0))
    assert sell["filled"] and sell["price"] < spot * 1.2
    assert sell["proceeds_usd"] == pytest.approx(buy["quantity"] * sell["price"])
    fallback = sim.buy("M1", 10.0, 1.0)  # no reserves → configured slippage + fee
    assert fallback["price"] == pytest.approx(1.0 * 1.015 * 1.01)


async def test_executor_paper_round_trip_tracks_book():
    ex = CryptoExecutor({"pump_fun": {"sol_price_usd": 200.0}})
    assert ex.paper and not ex.live_armed
    fill = await ex.buy("M1", 50.0, price=1e-7, reserves=(30.0, 3e8))
    assert fill["mode"] == "paper" and fill["intent"]["side"] == "buy"
    assert ex.paper_book["M1"] == pytest.approx(fill["quantity"])
    half = await ex.sell("M1", 0.5, price=1.1e-7)
    assert half["quantity"] == pytest.approx(fill["quantity"] / 2)
    await ex.close_position("M1", price=1.1e-7)
    assert "M1" not in ex.paper_book
    assert (await ex.tighten_stop("M1", 0.9))["desk_side"] is True
    assert await ex.get_positions() == []


async def test_executor_never_broadcasts_without_every_gate():
    base = {"pump_fun": {"sol_price_usd": 200.0}}
    assert CryptoExecutor({**base, "mode": "live"}, live_ack=True).paper is True       # no flag
    assert CryptoExecutor({**base, "mode": "live", "solana": {"live_broadcast": True}}).paper is True
    assert CryptoExecutor({**base, "mode": "paper", "solana": {"live_broadcast": True}},
                          live_ack=True).paper is True
    paper = CryptoExecutor(base)
    with pytest.raises(LiveTradingDisabled):
        await paper._broadcast(paper.build_buy_intent("M1", 10.0, price=1e-7), {})

    armed = CryptoExecutor({**base, "mode": "live", "solana": {"live_broadcast": True}}, live_ack=True)
    assert armed.live_armed and not armed.paper
    with pytest.raises(NotImplementedError):
        await armed.buy("M1", 10.0, price=1e-7)


async def test_desk_live_armed_executor_skips_cleanly(tmp_path):
    config = load_config(tmp_path)
    config["mode"] = "live"
    config["solana"]["live_broadcast"] = True
    desk = build(tmp_path, config=config, dry_run=False, live_ack=True)
    assert desk.crypto_executor.live_armed
    result = await desk.evaluate_token(TOKEN)
    assert result["reason"] == "executor_not_implemented" and desk.positions == []


async def test_desk_paper_mode_uses_executor_fills(tmp_path):
    desk = build(tmp_path, dry_run=False)
    result = await desk.evaluate_token(TOKEN)
    assert result["bought"] and result["tx_id"].startswith("PAPER-")
    assert desk.crypto_executor.paper_book["M1"] == pytest.approx(desk.positions[0].quantity)


def test_dry_run_signing_and_key_hygiene(caplog):
    from solders.keypair import Keypair

    kp = Keypair()
    secret = str(kp)
    config = {"mode": "live", "solana": {"wallet_key": secret, "live_broadcast": True},
              "pump_fun": {"sol_price_usd": 200.0}}
    with caplog.at_level(logging.DEBUG):
        ex = CryptoExecutor(config)   # live requested, not acked → paper + warning
        intent = ex.build_buy_intent(REAL_MINT, 25.0, price=1e-7, reserves=(30.0, 3e8))
        signed = ex.sign_intent(intent)
    assert signed["signed"] is True and signed["broadcast"] is False
    assert signed["pubkey"] == str(kp.pubkey())
    assert kp.pubkey().__class__  # signature verifies against the canonical bytes
    from solders.signature import Signature

    assert Signature.from_string(signed["signature"]).verify(kp.pubkey(), intent.canonical_bytes())
    assert intent.bonding_curve == derive_bonding_curve(REAL_MINT)
    assert intent.instruction_data_hex.startswith(BUY_DISCRIMINATOR.hex())
    blobs = [repr(ex), str(ex.config), json.dumps(intent.to_dict()), json.dumps(signed), caplog.text]
    assert all(secret not in blob for blob in blobs)
    assert config["solana"]["wallet_key"] == secret   # caller's dict untouched
    assert scrub_config(config)["solana"]["wallet_key"] != secret


def test_placeholder_or_garbage_key_never_signs_or_leaks():
    ex = CryptoExecutor({"solana": {"wallet_key": "REPLACE_ME_BASE58_SECRET_KEY"}})
    assert ex.sign_intent(ex.build_buy_intent("M1", 1.0))["reason"] == "no_wallet_key"
    bad = CryptoExecutor({"solana": {"wallet_key": "notakey-supersecret"}})
    out = bad.sign_intent(bad.build_buy_intent("M1", 1.0))
    assert out["signed"] is False and "supersecret" not in json.dumps(out)


# =============================================================================
# Enricher: bonding-curve account is not a holder
# =============================================================================

async def test_enricher_excludes_bonding_curve_account_from_top10():
    from src.crypto.enricher import OnChainEnricher, build_mint_account_data, curve_token_accounts
    from tests.test_enricher_exits import FakeRpc

    token = Token(mint=REAL_MINT)
    curve_atas = sorted(curve_token_accounts(token))
    assert len(curve_atas) == 3   # curve PDA + SPL + Token-2022 ATAs
    curve_ata = next(a for a in curve_atas if a != derive_bonding_curve(REAL_MINT))
    largest = [{"address": curve_ata, "ui_amount": 800.0},
               {"address": "H1", "ui_amount": 30.0}, {"address": "H2", "ui_amount": 20.0}]
    rpc = FakeRpc(build_mint_account_data(authority=None), largest=largest, supply_ui=1000.0)
    out = await OnChainEnricher({}, rpc=rpc).enrich(token)
    assert out.top10_holder_pct == pytest.approx(0.05)
    assert out.holders == 2 and out.holders_known is True
