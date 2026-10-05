import pytest

from src.crypto import crypto_scoring as cs
from src.models import Stock, Token
from src.stocks import stock_scoring as ss

TOKEN = Token(mint="M", symbol="X", holders=300, buys=90, sells=30, liquidity_usd=100_000)
STOCK = Stock(symbol="ACME", price=50, prev_close=46, avg_volume=2e6, volume=6e6, market_cap=5e9)

CLEAN_AUDIT = {
    "coordinated_buys": False, "wash_trading": False, "bundled_launch": False,
    "sniper_pct": 0.0, "insider_pct": 0.0, "safety_score": 1.0,
}
STRONG_NARRATIVE = {
    "meme_score": 1.0, "virality": 1.0, "originality": 1.0,
    "community_signal": 1.0, "is_derivative": False,
}
OPEN_PULSE = {"go_signal": 1.0}


# --- crypto ------------------------------------------------------------------

def test_crypto_perfect_inputs_score_one_and_buy():
    result = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["score"] == 1.0
    assert result["buy"] is True
    assert result["vetoed"] is False


def test_crypto_coordinated_buys_veto_beats_everything_else():
    audit = {**CLEAN_AUDIT, "coordinated_buys": True}
    result = cs.score_token(TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["vetoed"] is True
    assert result["reason"] == "veto_coordinated_buys"
    assert result["score"] == 0.0 and result["buy"] is False


def test_crypto_wash_trading_veto():
    audit = {**CLEAN_AUDIT, "wash_trading": True}
    assert cs.score_token(TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE)["reason"] == "veto_wash_trading"


def test_crypto_paused_market_veto():
    result = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, {"go_signal": 0.29})
    assert result["reason"] == "veto_market_paused"


def test_crypto_go_signal_exactly_at_the_gate_is_not_vetoed():
    result = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, {"go_signal": 0.30})
    assert result["vetoed"] is False


def test_crypto_pessimistic_fallbacks_are_vetoed():
    from src.crypto.auditor import Auditor
    from src.crypto.crypto_pulse import CryptoPulse
    from src.crypto.narrative import Narrative

    result = cs.score_token(
        TOKEN, Auditor({}).fallback(), Narrative({}).fallback(), CryptoPulse({}).fallback()
    )
    assert result["buy"] is False and result["vetoed"] is True


def test_crypto_threshold_is_exclusive_below_inclusive_at():
    weights = {**cs.DEFAULT_WEIGHTS, "min_score_to_buy": 1.0}
    at = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE, weights=weights)
    assert at["score"] == 1.0 and at["buy"] is True

    weights_above = {**cs.DEFAULT_WEIGHTS, "min_score_to_buy": 1.01}
    below = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE, weights=weights_above)
    assert below["buy"] is False and below["reason"] == "below_threshold"


def test_crypto_audit_score_penalises_snipers_and_bundling():
    assert cs.audit_score(CLEAN_AUDIT) == 1.0
    assert cs.audit_score({**CLEAN_AUDIT, "sniper_pct": 0.4}) == pytest.approx(0.8)
    assert cs.audit_score({**CLEAN_AUDIT, "bundled_launch": True}) == pytest.approx(0.7)
    assert cs.audit_score({"safety_score": 0.1, "sniper_pct": 1.0}) == 0.0  # clamped, never negative


def test_crypto_derivative_narrative_is_discounted():
    original = cs.narrative_score(STRONG_NARRATIVE)
    copy = cs.narrative_score({**STRONG_NARRATIVE, "is_derivative": True})
    assert copy == pytest.approx(original * 0.7)


def test_crypto_momentum_saturates_and_floors():
    assert cs.momentum_score(Token(mint="M", buys=0, sells=0, holders=0, liquidity_usd=0)) == 0.0
    hot = Token(mint="M", buys=1000, sells=1, holders=99999, liquidity_usd=1e9)
    assert cs.momentum_score(hot) == 1.0


def test_crypto_score_is_bounded_by_components():
    result = cs.score_token(
        Token(mint="M"), {"safety_score": 0.5}, {"meme_score": 0.5}, {"go_signal": 0.5}
    )
    assert 0.0 <= result["score"] <= 1.0


# --- stocks ------------------------------------------------------------------

CLEAN_RADAR = {"sentiment_score": 1.0, "news_momentum": 1.0, "controversy": 0.0}
CLEAN_INSIDER = {
    "insider_buying": 1.0, "insider_selling": 0.0,
    "institutional_flow": 1.0, "cluster_buying": True,
}
STRONG_ANALYST = {"fundamentals_score": 1.0, "technicals_score": 1.0}


def test_stock_perfect_inputs_score_one_and_buy():
    result = ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, CLEAN_INSIDER, OPEN_PULSE)
    assert result["score"] == 1.0
    assert result["buy"] is True


def test_stock_controversy_veto_at_the_boundary():
    just_under = ss.score_stock(STOCK, STRONG_ANALYST, {**CLEAN_RADAR, "controversy": 0.7},
                                CLEAN_INSIDER, OPEN_PULSE)
    assert just_under["vetoed"] is False   # 0.7 is not > 0.7

    just_over = ss.score_stock(STOCK, STRONG_ANALYST, {**CLEAN_RADAR, "controversy": 0.71},
                               CLEAN_INSIDER, OPEN_PULSE)
    assert just_over["reason"] == "veto_controversy"


def test_stock_insider_veto_needs_both_conditions():
    heavy_selling_but_also_buying = {**CLEAN_INSIDER, "insider_selling": 0.9, "insider_buying": 0.5}
    assert ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, heavy_selling_but_also_buying,
                          OPEN_PULSE)["vetoed"] is False

    dumping = {**CLEAN_INSIDER, "insider_selling": 0.81, "insider_buying": 0.19}
    assert ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, dumping,
                          OPEN_PULSE)["reason"] == "veto_insider_selling"


def test_stock_insider_veto_boundary_is_exclusive():
    edge = {**CLEAN_INSIDER, "insider_selling": 0.8, "insider_buying": 0.2}
    assert ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, edge, OPEN_PULSE)["vetoed"] is False


def test_stock_paused_market_veto():
    result = ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, CLEAN_INSIDER, {"go_signal": 0.1})
    assert result["reason"] == "veto_market_paused"


def test_stock_pessimistic_fallbacks_are_vetoed():
    from src.stocks.analyst import Analyst
    from src.stocks.insider import Insider
    from src.stocks.market_pulse import MarketPulse
    from src.stocks.radar import Radar

    result = ss.score_stock(
        STOCK, Analyst({}).fallback(), Radar({}).fallback(),
        Insider({}).fallback(), MarketPulse({}).fallback(),
    )
    assert result["buy"] is False and result["vetoed"] is True


def test_stock_news_score_subtracts_controversy():
    assert ss.news_score(CLEAN_RADAR) == 1.0
    assert ss.news_score({**CLEAN_RADAR, "controversy": 0.5}) == pytest.approx(0.75)
    assert ss.news_score({"sentiment_score": 0.0, "news_momentum": 0.0, "controversy": 0.5}) == 0.0


def test_stock_insider_score_rewards_clusters_and_punishes_selling():
    assert ss.insider_score(CLEAN_INSIDER) == 1.0
    assert ss.insider_score({**CLEAN_INSIDER, "cluster_buying": False}) == pytest.approx(0.85)
    assert ss.insider_score({"insider_buying": 0.0, "insider_selling": 1.0}) == 0.0


def test_stock_missing_keys_default_pessimistically():
    result = ss.score_stock(STOCK, {}, {}, {}, {})
    assert result["vetoed"] is True   # missing controversy defaults to 1.0
    assert result["buy"] is False


def test_stock_custom_weights_change_the_verdict():
    mediocre_fundamentals = {"fundamentals_score": 0.0, "technicals_score": 1.0}
    fundamentals_heavy = {**ss.DEFAULT_WEIGHTS, "fundamentals": 0.9, "technicals": 0.02,
                          "news_sentiment": 0.02, "insider": 0.03, "pulse": 0.03}
    result = ss.score_stock(STOCK, mediocre_fundamentals, CLEAN_RADAR, CLEAN_INSIDER,
                            OPEN_PULSE, weights=fundamentals_heavy)
    assert result["buy"] is False


# --- hard vetoes: bundled / sniper / insider (improve-memecoin-exits) ----------

def test_crypto_bundled_launch_is_hard_vetoed():
    audit = {**CLEAN_AUDIT, "bundled_launch": True}
    result = cs.score_token(TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["vetoed"] is True
    assert result["reason"] == "veto_bundled_launch"
    assert result["buy"] is False and result["score"] == 0.0


def test_crypto_sniper_pct_above_threshold_is_vetoed():
    audit = {**CLEAN_AUDIT, "sniper_pct": 0.26}
    result = cs.score_token(TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["reason"] == "veto_sniper_pct"

    edge = cs.score_token(
        TOKEN, {**CLEAN_AUDIT, "sniper_pct": 0.25}, STRONG_NARRATIVE, OPEN_PULSE
    )
    assert edge["vetoed"] is False  # exclusive above threshold


def test_crypto_insider_pct_above_threshold_is_vetoed():
    audit = {**CLEAN_AUDIT, "insider_pct": 0.16}
    result = cs.score_token(TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["reason"] == "veto_insider_pct"

    edge = cs.score_token(
        TOKEN, {**CLEAN_AUDIT, "insider_pct": 0.15}, STRONG_NARRATIVE, OPEN_PULSE
    )
    assert edge["vetoed"] is False


def test_crypto_veto_thresholds_are_configurable():
    loose = {"veto_bundled_launch": False, "max_sniper_pct": 0.90, "max_insider_pct": 0.90}
    audit = {**CLEAN_AUDIT, "bundled_launch": True, "sniper_pct": 0.5, "insider_pct": 0.4}
    result = cs.score_token(
        TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE, vetoes=loose
    )
    assert result["vetoed"] is False
    assert result["buy"] is True


def test_crypto_veto_defaults_apply_when_config_omitted():
    # hard_veto with no vetoes arg still enforces bundled/sniper/insider defaults
    assert cs.hard_veto({**CLEAN_AUDIT, "bundled_launch": True}, OPEN_PULSE) == "veto_bundled_launch"
    assert cs.hard_veto({**CLEAN_AUDIT, "sniper_pct": 0.30}, OPEN_PULSE) == "veto_sniper_pct"
    assert cs.hard_veto({**CLEAN_AUDIT, "insider_pct": 0.20}, OPEN_PULSE) == "veto_insider_pct"
