"""Entry quality scorecard — pure code, no model.

The weighted matrix in `crypto_scoring.py` answers "do the bots like it?". This
scorecard answers a different question: "is the *tape* good enough to enter?".
It grades five factors on 0..1 using linear ramps between documented floor and
target values, applies per-factor kill floors (one terrible factor is a veto no
matter how good the rest look), and returns a composite plus a size grade.

Factors (all thresholds overridable under `entry_scorecard:` in config):

  velocity        buys per minute over the watch window + buy share of trades
  holder_quality  unique-trader ratio (wash/bot detector), unique count,
                  top-10 concentration, deployer holding, real holder count
  liquidity       curve liquidity USD depth + volume turnover vs liquidity
  social_heat     narrative virality / community / meme score, derivative
                  penalty, presence of socials
  audit           auditor safety minus sniper / insider share and red flags

Code-only factors (velocity, holder_quality, liquidity) can be checked before a
single model call is paid for: see `prescreen`.

See STRATEGY.md for the reasoning behind each threshold.
"""

from __future__ import annotations

from typing import Any

from ..models import Token

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "weights": {
        "velocity": 0.25,
        "holder_quality": 0.20,
        "liquidity": 0.15,
        "social_heat": 0.20,
        "audit": 0.20,
    },
    # Composite must clear this (regime gate may add to it).
    "min_entry_score": 0.55,
    # Any single factor below its floor kills the entry.
    "kill_floors": {
        "velocity": 0.20,
        "holder_quality": 0.25,
        "liquidity": 0.15,
        "social_heat": 0.25,
        "audit": 0.40,
    },
    # Ramp endpoints: value at/below `lo` scores 0, at/above `hi` scores 1.
    "velocity": {
        "buys_per_min_lo": 1.0,
        "buys_per_min_hi": 8.0,
        "buy_share_lo": 0.50,   # buys / (buys + sells)
        "buy_share_hi": 0.75,
    },
    "holder_quality": {
        "unique_ratio_lo": 0.15,  # unique_traders / trades; low = few wallets churning
        "unique_ratio_hi": 0.50,
        "unique_traders_lo": 12,
        "unique_traders_hi": 60,
        "top10_good": 0.20,       # <= this concentration scores 1
        "top10_bad": 0.45,        # >= this scores 0
        "dev_good": 0.02,
        "dev_bad": 0.10,
        "holders_lo": 25,
        "holders_hi": 250,
    },
    "liquidity": {
        "usd_lo": 5_000.0,
        "usd_hi": 60_000.0,
        "turnover_lo": 0.30,      # window volume SOL / curve SOL
        "turnover_hi": 2.00,
    },
    "social_heat": {
        "derivative_penalty": 0.75,
        "socials_bonus_each": 0.03,
        "socials_bonus_max": 0.09,
    },
    "audit": {
        "red_flag_penalty": 0.03,
    },
    # Size grade: composite >= A → full planned size, >= B → 0.75, else 0.5.
    "grades": {"A": 0.75, "B": 0.65},
    "grade_size_mult": {"A": 1.0, "B": 0.75, "C": 0.5},
}

CODE_ONLY_FACTORS = ("velocity", "holder_quality", "liquidity")


def _merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0..1 between lo and hi (clamped). Handles inverted ranges."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if hi == lo:
        return 1.0 if value >= hi else 0.0
    if hi > lo:
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))
    # inverted: lower is better
    return max(0.0, min(1.0, (lo - value) / (lo - hi)))


def buys_per_minute(token: Token) -> float:
    seconds = float(token.observed_seconds or 0) or float(token.age_seconds or 0)
    if seconds <= 0:
        return 0.0
    return token.buys / (seconds / 60.0)


class EntryScorecard:
    """Multi-factor entry grade. Construct once from config; call `score`."""

    def __init__(self, config: dict[str, Any] | None = None, sol_usd: float = 0.0):
        cfg = (config or {}).get("entry_scorecard", {}) if config else {}
        self.cfg = _merge(DEFAULTS, cfg or {})
        self.enabled = bool(self.cfg.get("enabled", True))
        self.sol_usd = float(sol_usd or 0)

    # -- factors ----------------------------------------------------------------

    def velocity(self, token: Token) -> float:
        c = self.cfg["velocity"]
        bpm = buys_per_minute(token)
        trades = token.buys + token.sells
        share = token.buys / trades if trades else 0.0
        return round(
            0.7 * ramp(bpm, c["buys_per_min_lo"], c["buys_per_min_hi"])
            + 0.3 * ramp(share, c["buy_share_lo"], c["buy_share_hi"]),
            4,
        )

    def holder_quality(self, token: Token) -> float:
        c = self.cfg["holder_quality"]
        trades = token.buys + token.sells
        unique_ratio = (token.unique_traders / trades) if trades else 0.0
        parts = {
            "unique_ratio": ramp(unique_ratio, c["unique_ratio_lo"], c["unique_ratio_hi"]),
            "unique_traders": ramp(
                token.unique_traders, c["unique_traders_lo"], c["unique_traders_hi"]
            ),
        }
        # Concentration / dev / holder count are only meaningful once the
        # enricher ran. A zero default means "unknown" → neutral 0.5, never 1.0.
        enriched = token.holders_known or token.top10_holder_pct > 0
        parts["top10"] = (
            ramp(token.top10_holder_pct, c["top10_bad"], c["top10_good"]) if enriched else 0.5
        )
        parts["dev"] = (
            ramp(token.dev_holding_pct, c["dev_bad"], c["dev_good"]) if enriched else 0.5
        )
        if token.holders_known:
            holders = ramp(token.holders, c["holders_lo"], c["holders_hi"])
            score = (
                0.30 * parts["unique_ratio"]
                + 0.20 * parts["unique_traders"]
                + 0.20 * parts["top10"]
                + 0.15 * parts["dev"]
                + 0.15 * holders
            )
        else:
            score = (
                0.35 * parts["unique_ratio"]
                + 0.25 * parts["unique_traders"]
                + 0.25 * parts["top10"]
                + 0.15 * parts["dev"]
            )
        return round(score, 4)

    def liquidity(self, token: Token) -> float:
        c = self.cfg["liquidity"]
        depth = ramp(token.liquidity_usd, c["usd_lo"], c["usd_hi"])
        if token.volume_sol > 0 and token.curve_sol > 0:
            turnover = token.volume_sol / token.curve_sol
            return round(0.75 * depth + 0.25 * ramp(turnover, c["turnover_lo"], c["turnover_hi"]), 4)
        return round(depth, 4)

    def social_heat(self, token: Token, narrative: dict[str, Any] | None) -> float:
        c = self.cfg["social_heat"]
        n = narrative or {}
        heat = (
            0.40 * float(n.get("virality", 0.0) or 0.0)
            + 0.30 * float(n.get("community_signal", 0.0) or 0.0)
            + 0.30 * float(n.get("meme_score", 0.0) or 0.0)
        )
        if n.get("is_derivative"):
            heat *= float(c["derivative_penalty"])
        socials = sum(1 for k in ("twitter", "telegram", "website") if token.socials.get(k))
        heat += min(float(c["socials_bonus_max"]), socials * float(c["socials_bonus_each"]))
        return round(max(0.0, min(1.0, heat)), 4)

    def audit(self, audit: dict[str, Any] | None) -> float:
        c = self.cfg["audit"]
        a = audit or {}
        safety = float(a.get("safety_score", 0.0) or 0.0)
        penalty = 0.5 * float(a.get("sniper_pct", 0.0) or 0.0) + 0.5 * float(
            a.get("insider_pct", 0.0) or 0.0
        )
        penalty += float(c["red_flag_penalty"]) * len(a.get("red_flags") or [])
        if a.get("bundled_launch"):
            penalty += 0.3
        return round(max(0.0, min(1.0, safety - penalty)), 4)

    # -- verdicts ---------------------------------------------------------------

    def code_factors(self, token: Token) -> dict[str, float]:
        return {
            "velocity": self.velocity(token),
            "holder_quality": self.holder_quality(token),
            "liquidity": self.liquidity(token),
        }

    def prescreen(self, token: Token) -> str | None:
        """Kill-floor check on code-only factors. Run before any model call."""
        if not self.enabled:
            return None
        floors = self.cfg["kill_floors"]
        for name, value in self.code_factors(token).items():
            floor = floors.get(name)
            if floor is not None and value < float(floor):
                return f"scorecard_weak_{name}"
        return None

    def grade(self, score: float) -> str:
        g = self.cfg["grades"]
        if score >= float(g["A"]):
            return "A"
        if score >= float(g["B"]):
            return "B"
        return "C"

    def score(
        self,
        token: Token,
        narrative: dict[str, Any] | None = None,
        audit: dict[str, Any] | None = None,
        min_score_add: float = 0.0,
    ) -> dict[str, Any]:
        """Composite verdict: {score, pass, reason, grade, size_mult, factors, threshold}."""
        factors = self.code_factors(token)
        factors["social_heat"] = self.social_heat(token, narrative)
        factors["audit"] = self.audit(audit)

        weights = self.cfg["weights"]
        denom = sum(float(weights.get(k, 0)) for k in factors) or 1.0
        composite = round(
            sum(factors[k] * float(weights.get(k, 0)) for k in factors) / denom, 4
        )
        threshold = round(float(self.cfg["min_entry_score"]) + float(min_score_add), 4)
        grade = self.grade(composite)
        result: dict[str, Any] = {
            "score": composite,
            "factors": factors,
            "threshold": threshold,
            "grade": grade,
            "size_mult": float(self.cfg["grade_size_mult"].get(grade, 0.5)),
            "buys_per_min": round(buys_per_minute(token), 3),
        }
        if not self.enabled:
            result.update({"pass": True, "reason": "scorecard_disabled"})
            return result

        floors = self.cfg["kill_floors"]
        for name, value in factors.items():
            floor = floors.get(name)
            if floor is not None and value < float(floor):
                result.update({"pass": False, "reason": f"scorecard_weak_{name}"})
                return result
        if composite < threshold:
            result.update({"pass": False, "reason": "scorecard_below_threshold"})
            return result
        result.update({"pass": True, "reason": "scorecard_ok"})
        return result
