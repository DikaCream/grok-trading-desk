"""Portfolio risk manager. One instance governs both markets.

Every limit here is cross-market by design: a memecoin loss eats the same daily
budget an equity loss does, and the open-position cap counts both books. Nothing
in this file asks a model for permission.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date
from typing import Any

from ..models import Allocation, Market, Position

log = logging.getLogger(__name__)


class RiskManager:
    """Budget, exposure and position sizing across crypto and stocks."""

    def __init__(self, config: dict[str, Any]):
        risk = (config or {}).get("risk", {}) or {}
        self.total_budget_usd = float(risk.get("total_budget_usd", 1000.0))
        self.daily_loss_limit_usd = float(risk.get("daily_loss_limit_usd", 200.0))
        self.max_open_total = int(risk.get("max_open_total", 10))
        self.max_open_crypto = int(risk.get("max_open_crypto", 6))
        self.max_open_stocks = int(risk.get("max_open_stocks", 6))
        self.max_per_sector = int(risk.get("max_per_sector", 2))
        self.crypto_max_pct = float(risk.get("crypto_max_pct", 1.0))
        self.stock_max_pct = float(risk.get("stock_max_pct", 1.0))
        self.max_position_pct_of_market = float(risk.get("max_position_pct_of_market", 0.15))
        self.max_position_pct_of_remaining_loss = float(
            risk.get("max_position_pct_of_remaining_loss", 0.25)
        )
        # Crypto-only: cap notional to a fraction of bonding-curve liquidity USD.
        self.max_position_pct_of_liquidity = float(
            risk.get("max_position_pct_of_liquidity", 0.03)
        )

        self.allocation = Allocation()
        self.realized_pnl_today = 0.0
        self.deployed_usd: dict[Market, float] = {Market.CRYPTO: 0.0, Market.STOCKS: 0.0}
        self.session_date = date.today()

    # -- daily bookkeeping --------------------------------------------------------

    def maybe_reset_day(self, today: date | None = None) -> bool:
        """Zero the daily counters when the date rolls over. Returns True if reset."""
        today = today or date.today()
        if today == self.session_date:
            return False
        self.session_date = today
        self.realized_pnl_today = 0.0
        self.deployed_usd = {Market.CRYPTO: 0.0, Market.STOCKS: 0.0}
        log.info("risk: new session %s, daily counters reset", today)
        return True

    def record_fill(self, market: Market, amount_usd: float) -> None:
        self.deployed_usd[market] = self.deployed_usd.get(market, 0.0) + amount_usd

    def record_close(self, market: Market, pnl_usd: float, amount_usd: float = 0.0) -> None:
        self.realized_pnl_today += pnl_usd
        if amount_usd:
            self.deployed_usd[market] = max(0.0, self.deployed_usd.get(market, 0.0) - amount_usd)

    def set_allocation(self, allocation: Allocation) -> Allocation:
        """Store the allocator's split, clamped to the configured ceilings."""
        self.allocation = allocation.normalized(self.crypto_max_pct, self.stock_max_pct)
        return self.allocation

    # -- derived state ------------------------------------------------------------

    def market_budget(self, market: Market) -> float:
        pct = (
            self.allocation.crypto_pct
            if market == Market.CRYPTO
            else self.allocation.stocks_pct
        )
        return self.total_budget_usd * pct

    def remaining_loss_room(self) -> float:
        """How much more we may lose today before the desk shuts."""
        return max(0.0, self.daily_loss_limit_usd + min(0.0, self.realized_pnl_today))

    def daily_loss_breached(self) -> bool:
        return self.realized_pnl_today <= -self.daily_loss_limit_usd

    def remaining_market_budget(self, market: Market) -> float:
        return max(0.0, self.market_budget(market) - self.deployed_usd.get(market, 0.0))

    # -- the gate -----------------------------------------------------------------

    def can_open(
        self,
        market: Market,
        positions: list[Position],
        sector: str = "unknown",
        amount_usd: float = 0.0,
    ) -> tuple[bool, str]:
        """Return (allowed, reason). `reason` is 'ok' when allowed."""
        if self.daily_loss_breached():
            return False, "daily_loss_limit_reached"

        if len(positions) >= self.max_open_total:
            return False, "max_open_total"

        same_market = [p for p in positions if p.market == market]
        if market == Market.CRYPTO and len(same_market) >= self.max_open_crypto:
            return False, "max_open_crypto"
        if market == Market.STOCKS and len(same_market) >= self.max_open_stocks:
            return False, "max_open_stocks"

        if market == Market.STOCKS and sector and sector != "unknown":
            per_sector = Counter(p.sector for p in same_market)
            if per_sector[sector] >= self.max_per_sector:
                return False, "max_per_sector"

        if self.remaining_market_budget(market) <= 0:
            return False, "market_budget_exhausted"

        if amount_usd and amount_usd > self.remaining_market_budget(market) + 1e-9:
            return False, "exceeds_market_budget"

        return True, "ok"

    def position_size(
        self,
        market: Market,
        score: float = 1.0,
        liquidity_usd: float | None = None,
    ) -> float:
        """USD for one new trade, floored at 0.

        Bounded three ways: a share of that market's budget, a share of what is
        left of today's loss allowance, and whatever budget is actually free.
        The score scales linearly between half size and full size.

        Crypto only: when `liquidity_usd` is provided, also cap to
        `max_position_pct_of_liquidity` of curve liquidity (default 3%).
        """
        by_market = self.market_budget(market) * self.max_position_pct_of_market
        by_loss_room = self.remaining_loss_room() * self.max_position_pct_of_remaining_loss
        free = self.remaining_market_budget(market)

        size = min(by_market, by_loss_room, free)
        confidence = max(0.0, min(1.0, float(score)))
        size *= 0.5 + 0.5 * confidence

        if (
            market == Market.CRYPTO
            and liquidity_usd is not None
            and float(liquidity_usd) > 0
            and self.max_position_pct_of_liquidity > 0
        ):
            size = min(size, float(liquidity_usd) * self.max_position_pct_of_liquidity)

        return round(max(0.0, size), 2)

    def snapshot(self, positions: list[Position] | None = None) -> dict[str, Any]:
        positions = positions or []
        return {
            "date": str(self.session_date),
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "remaining_loss_room": round(self.remaining_loss_room(), 2),
            "daily_loss_breached": self.daily_loss_breached(),
            "allocation": {
                "crypto_pct": round(self.allocation.crypto_pct, 4),
                "stocks_pct": round(self.allocation.stocks_pct, 4),
            },
            "budgets": {
                "crypto": round(self.market_budget(Market.CRYPTO), 2),
                "stocks": round(self.market_budget(Market.STOCKS), 2),
            },
            "deployed": {k.value: round(v, 2) for k, v in self.deployed_usd.items()},
            "open_positions": {
                "total": len(positions),
                "crypto": sum(1 for p in positions if p.market == Market.CRYPTO),
                "stocks": sum(1 for p in positions if p.market == Market.STOCKS),
            },
        }
