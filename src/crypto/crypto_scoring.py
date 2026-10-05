"""Crypto scoring matrix. Pure code — no model gets to move these weights.

Hard vetoes live here, not in prompts: a coordinated buy ring or wash trading
kills the candidate before any weighted score is computed.
"""

from __future__ import annotations

from typing import Any

from ..models import Token

DEFAULT_WEIGHTS = {
    "audit_safety": 0.35,
    "narrative": 0.30,
    "momentum": 0.20,
    "pulse": 0.15,
    "min_score_to_buy": 0.62,
}

# Hard veto defaults — old configs without crypto_vetoes still get these gates.
DEFAULT_VETOES = {
    "veto_bundled_launch": True,
    "max_sniper_pct": 0.25,
    "max_insider_pct": 0.15,
}


def hard_veto(
    audit: dict[str, Any],
    pulse: dict[str, Any],
    min_go_signal: float = 0.3,
    vetoes: dict[str, Any] | None = None,
) -> str | None:
    """Return the veto reason, or None. Checked before scoring."""
    v = {**DEFAULT_VETOES, **(vetoes or {})}

    if audit.get("coordinated_buys"):
        return "veto_coordinated_buys"
    if audit.get("wash_trading"):
        return "veto_wash_trading"
    if v.get("veto_bundled_launch", True) and audit.get("bundled_launch"):
        return "veto_bundled_launch"

    max_sniper = v.get("max_sniper_pct")
    if max_sniper is not None and float(audit.get("sniper_pct", 0.0) or 0.0) > float(max_sniper):
        return "veto_sniper_pct"

    max_insider = v.get("max_insider_pct")
    if max_insider is not None and float(audit.get("insider_pct", 0.0) or 0.0) > float(max_insider):
        return "veto_insider_pct"

    if float(pulse.get("go_signal", 0.0)) < min_go_signal:
        return "veto_market_paused"
    return None


def momentum_score(token: Token) -> float:
    """Buy pressure and holder count, squashed into 0..1."""
    ratio = token.buy_sell_ratio
    ratio_component = min(ratio / 3.0, 1.0) if ratio > 0 else 0.0
    holder_component = min(token.holders / 300.0, 1.0)
    liquidity_component = min(token.liquidity_usd / 100_000.0, 1.0)
    return round(
        0.5 * ratio_component + 0.3 * holder_component + 0.2 * liquidity_component, 4
    )


def narrative_score(narrative: dict[str, Any]) -> float:
    meme = float(narrative.get("meme_score", 0.0))
    virality = float(narrative.get("virality", 0.0))
    originality = float(narrative.get("originality", 0.0))
    community = float(narrative.get("community_signal", 0.0))
    score = 0.4 * meme + 0.3 * virality + 0.15 * originality + 0.15 * community
    if narrative.get("is_derivative"):
        score *= 0.7  # a copy of a running meme is worth less than the original
    return round(max(0.0, min(1.0, score)), 4)


def audit_score(audit: dict[str, Any]) -> float:
    safety = float(audit.get("safety_score", 0.0))
    penalty = 0.5 * float(audit.get("sniper_pct", 0.0)) + 0.5 * float(audit.get("insider_pct", 0.0))
    if audit.get("bundled_launch"):
        penalty += 0.3
    return round(max(0.0, min(1.0, safety - penalty)), 4)


def score_token(
    token: Token,
    audit: dict[str, Any],
    narrative: dict[str, Any],
    pulse: dict[str, Any],
    weights: dict[str, Any] | None = None,
    min_go_signal: float = 0.3,
    vetoes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Weighted score plus the buy/skip verdict and its components."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    veto = hard_veto(audit, pulse, min_go_signal, vetoes=vetoes)
    components = {
        "audit_safety": audit_score(audit),
        "narrative": narrative_score(narrative),
        "momentum": momentum_score(token),
        "pulse": float(pulse.get("go_signal", 0.0)),
    }

    if veto is not None:
        return {
            "score": 0.0,
            "buy": False,
            "reason": veto,
            "vetoed": True,
            "components": components,
        }

    denominator = sum(w[k] for k in components) or 1.0
    score = round(sum(components[k] * w[k] for k in components) / denominator, 4)
    threshold = float(w["min_score_to_buy"])
    return {
        "score": score,
        "buy": score >= threshold,
        "reason": "above_threshold" if score >= threshold else "below_threshold",
        "vetoed": False,
        "components": components,
        "threshold": threshold,
    }
