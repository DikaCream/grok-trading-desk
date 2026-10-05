"""Pydantic models shared by both markets."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Market(str, Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"


class ExitAction(str, Enum):
    HOLD = "HOLD"
    TIGHTEN = "TIGHTEN"
    TRIM = "TRIM"
    CLOSE = "CLOSE"


class Token(BaseModel):
    """A pump.fun launch.

    Fields split into two groups. The first is what a `subscribeNewToken` event
    actually carries; the second is what the watch window measures afterwards.
    Only the first is populated at creation time.
    """

    mint: str
    symbol: str = ""
    name: str = ""
    creator: str = ""
    liquidity_usd: float = 0.0
    market_cap_usd: float = 0.0
    holders: int = 0
    # True only when an on-chain enricher filled `holders`. Watch-window unique
    # traders must not be treated as a holder count (see scout + enricher).
    holders_known: bool = False
    top10_holder_pct: float = 0.0
    dev_holding_pct: float = 0.0
    age_seconds: float = 0.0
    buys: int = 0
    sells: int = 0
    # Tri-state on purpose. None = unknown (enricher did not confirm). After
    # enrichment, stage-two hard requires reject None as mint_unknown / lp_unknown
    # rather than silently passing.
    mint_revoked: bool | None = None
    lp_burned: bool | None = None
    socials: dict[str, str] = Field(default_factory=dict)

    # -- straight off the create event (denominated in SOL) --
    curve_sol: float = 0.0            # vSolInBondingCurve
    curve_tokens: float = 0.0         # vTokensInBondingCurve
    market_cap_sol: float = 0.0
    dev_initial_buy_sol: float = 0.0  # the deployer's own opening buy
    uri: str = ""
    pool: str = ""

    # -- accumulated during the watch window --
    unique_traders: int = 0
    observed_seconds: float = 0.0
    volume_sol: float = 0.0
    price_change_pct: float = 0.0
    dev_sold: bool = False

    raw: dict[str, Any] = Field(default_factory=dict)
    seen_at: datetime = Field(default_factory=_utcnow)

    @property
    def buy_sell_ratio(self) -> float:
        if self.sells <= 0:
            return float(self.buys) if self.buys else 0.0
        return self.buys / self.sells

    @property
    def trades(self) -> int:
        return self.buys + self.sells

    def priced(self, sol_usd: float) -> "Token":
        """Return a copy with the SOL-denominated fields converted to USD.

        The feed speaks SOL; every threshold in the config speaks USD.
        """
        if sol_usd <= 0:
            return self
        return self.model_copy(
            update={
                "liquidity_usd": self.curve_sol * sol_usd,
                "market_cap_usd": self.market_cap_sol * sol_usd,
            }
        )


class Stock(BaseModel):
    """A screener candidate."""

    symbol: str
    name: str = ""
    sector: str = "unknown"
    price: float = 0.0
    prev_close: float = 0.0
    avg_volume: float = 0.0
    volume: float = 0.0
    market_cap: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)
    seen_at: datetime = Field(default_factory=_utcnow)

    @property
    def rel_volume(self) -> float:
        if self.avg_volume <= 0:
            return 0.0
        return self.volume / self.avg_volume

    @property
    def gap_pct(self) -> float:
        if self.prev_close <= 0:
            return 0.0
        return (self.price - self.prev_close) / self.prev_close


class Position(BaseModel):
    """An open position on either market."""

    market: Market
    symbol: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    amount_usd: float = 0.0
    stop_price: float | None = None
    take_profit_price: float | None = None
    # Peak mark since entry — used by crypto trailing stops. Stocks ignore it.
    peak_price: float | None = None
    sector: str = "unknown"
    opened_at: datetime = Field(default_factory=_utcnow)
    score: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def hold_time_hours(self) -> float:
        opened = self.opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return (_utcnow() - opened).total_seconds() / 3600.0

    @property
    def hold_time_minutes(self) -> float:
        return self.hold_time_hours * 60.0

    @property
    def pnl_usd(self) -> float:
        if not self.current_price:
            return 0.0
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0 or not self.current_price:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


class Allocation(BaseModel):
    """Budget split between the two markets."""

    crypto_pct: float = 0.5
    stocks_pct: float = 0.5
    reason: str = ""
    decided_at: datetime = Field(default_factory=_utcnow)

    def normalized(self, crypto_max_pct: float = 1.0, stock_max_pct: float = 1.0) -> "Allocation":
        """Clamp to the configured ceilings, then renormalize to sum to 1."""
        crypto = max(0.0, min(self.crypto_pct, crypto_max_pct))
        stocks = max(0.0, min(self.stocks_pct, stock_max_pct))
        total = crypto + stocks
        if total <= 0:
            crypto, stocks, total = 0.5, 0.5, 1.0
        return Allocation(
            crypto_pct=crypto / total,
            stocks_pct=stocks / total,
            reason=self.reason,
            decided_at=self.decided_at,
        )


class Pulse(BaseModel):
    """Regime read for one market."""

    market: Market
    regime: str = "unknown"
    go_signal: float = 0.0
    risk_appetite: float = 0.5
    notes: str = ""
    fetched_at: datetime = Field(default_factory=_utcnow)


class Decision(BaseModel):
    """Final verdict on one candidate, ready to log."""

    market: Market
    symbol: str
    score: float = 0.0
    buy: bool = False
    reason: str = ""
    agent_scores: dict[str, Any] = Field(default_factory=dict)
    amount_usd: float = 0.0
