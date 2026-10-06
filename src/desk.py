"""Grok Trading Desk — the orchestrator.

Six concurrent asyncio loops, one shared risk manager, one event log:

  crypto_loop        continuous, 24/7, driven by the pump.fun WebSocket
  stock_loop         wakes every minute, works only inside the RTH window
  exit_loop          every 4 hours (stocks + shared book)
  crypto_exit_loop   faster LLM exits for crypto (default ~20 min)
  crypto_stops_loop  every ~30s: price poll, staged-entry adds, exit ladder
                     (dump detection, stops, partial TPs, runner trail, time-stop)
  allocator_loop     once a day, resets the crypto/stocks budget split

Run:  python -m src.desk --config config.yaml [--dry-run] [--i-understand-the-risk]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from .base_agent import CostTracker
from .crypto.auditor import Auditor
from .crypto.crypto_checker import CryptoChecker
from .crypto.crypto_executor import CryptoExecutor, LiveTradingDisabled
from .crypto.crypto_pulse import CryptoPulse
from .crypto.crypto_scoring import score_token
from .crypto.dump_detector import dump_config
from .crypto.enricher import OnChainEnricher
from .crypto.entry_scorecard import EntryScorecard
from .crypto.exit_ladder import ExitLadder, ExitSignal
from .crypto.narrative import Narrative
from .crypto.price_poller import CryptoPricePoller
from .crypto.regime import RegimeGate
from .crypto.scout import Scout
from .crypto.staged_entry import StagedEntry
from .crypto.toxic_flow import ToxicFlowFilter, fingerprint
from .models import Allocation, Market, Position
from .shared.allocator import Allocator
from .shared.exit_manager import ExitManager
from .shared.log import EventLog
from .shared.memory import OutcomeMemory
from .shared.risk import RiskManager
from .stocks.analyst import Analyst
from .stocks.insider import Insider
from .stocks.market_pulse import MarketPulse
from .stocks.radar import Radar
from .stocks.screener import Screener
from .stocks.stock_checker import StockChecker
from .stocks.stock_executor import OrderRejected, StockExecutor
from .stocks.stock_scoring import score_stock

log = logging.getLogger("desk")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_hhmm(value: str, default: dtime) -> dtime:
    try:
        hour, minute = (int(part) for part in str(value).split(":", 1))
        return dtime(hour, minute)
    except (TypeError, ValueError):
        return default



def curve_price(token) -> float:
    """Bonding-curve spot from curve reserves. 0 when unavailable."""
    try:
        curve_sol = float(getattr(token, "curve_sol", 0) or 0)
        curve_tokens = float(getattr(token, "curve_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if curve_tokens <= 0 or curve_sol <= 0:
        return 0.0
    return curve_sol / curve_tokens


def crypto_stop_tp_prices(
    entry_price: float,
    take_profit_pct: float = 0.20,
    stop_loss_pct: float = 0.08,
) -> tuple[float | None, float | None]:
    """Mechanical stop and take-profit levels relative to entry."""
    if entry_price <= 0:
        return None, None
    stop = entry_price * (1.0 - float(stop_loss_pct))
    take_profit = entry_price * (1.0 + float(take_profit_pct))
    return stop, take_profit


def check_crypto_stops(position: Position, price: float | None = None) -> str | None:
    """Return 'stop_loss', 'take_profit', or None when price crosses levels.

    Prefers CLOSE at take-profit (no trim). Unit-testable with a mocked price.
    """
    return check_crypto_exits(position, price)


def check_crypto_exits(
    position: Position,
    price: float | None = None,
    *,
    exits_cfg: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> str | None:
    """Mechanical crypto exits: stop, take-profit, trailing stop, time-stop.

    Returns one of: stop_loss, take_profit, trailing_stop, time_stop, or None.
    Updates position.peak_price when a usable mark is supplied.
    """
    if position.market != Market.CRYPTO:
        return None
    cfg = exits_cfg or {}
    px = float(price) if price is not None else float(position.current_price or 0.0)

    # Always advance the peak when we have a mark (trailing depends on it).
    if px > 0:
        peak = float(position.peak_price or position.entry_price or 0.0)
        if px > peak:
            position.peak_price = px
            peak = px
        else:
            position.peak_price = peak if peak > 0 else px

    if px > 0:
        if position.stop_price is not None and px <= float(position.stop_price):
            return "stop_loss"

        # Trailing stop: arm after +N% from entry, then trail M% off peak.
        activate = cfg.get("trailing_stop_activate_pct")
        trail = cfg.get("trailing_stop_pct")
        if (
            activate is not None
            and trail is not None
            and position.entry_price > 0
            and position.peak_price
        ):
            gain = (float(position.peak_price) - position.entry_price) / position.entry_price
            if gain >= float(activate):
                trail_stop = float(position.peak_price) * (1.0 - float(trail))
                if px <= trail_stop:
                    return "trailing_stop"

        if position.take_profit_price is not None and px >= float(position.take_profit_price):
            return "take_profit"

    # Time-stop: max hold for memecoins (checked even without a fresh mark).
    max_hold = cfg.get("max_hold_minutes")
    if max_hold is not None:
        hold_min = position.hold_time_minutes
        if now is not None:
            opened = position.opened_at
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            ref = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
            hold_min = (ref - opened).total_seconds() / 60.0
        if hold_min >= float(max_hold):
            return "time_stop"

    return None


class TradingDesk:
    """Owns every bot, the shared risk state and the four loops."""

    def __init__(self, config: dict[str, Any], dry_run: bool = False, live_ack: bool = False):
        self.config = config
        self.dry_run = dry_run

        self.log = EventLog(config)
        self.risk = RiskManager(config)
        # One tracker across every bot, so spend is a desk number not a per-bot one.
        self.costs = CostTracker()
        self.memory = OutcomeMemory(config, event_log=self.log)

        def agent(cls):
            return cls(config, costs=self.costs)

        # crypto side
        self.enricher = OnChainEnricher(config)
        self.price_poller = CryptoPricePoller(config)
        self.toxic = ToxicFlowFilter(config)
        if self.toxic.enabled:
            self.toxic.seed_from_log(self.log.read())
        self.scout = Scout(
            config, enricher=self.enricher, toxic=self.toxic if self.toxic.enabled else None
        )
        self.scorecard = EntryScorecard(config, sol_usd=self.scout.sol_price_usd)
        self.regime_gate = RegimeGate(config)
        self.staged = StagedEntry(config)
        self.ladder = ExitLadder(config)
        self.dump_cfg = dump_config(config)
        self.auditor = agent(Auditor)
        self.narrative = agent(Narrative)
        self.crypto_pulse = agent(CryptoPulse)
        self.crypto_checker = agent(CryptoChecker)
        self.crypto_executor = CryptoExecutor(config, live_ack=live_ack)

        # stock side
        self.screener = Screener(config)
        self.analyst = agent(Analyst)
        self.radar = agent(Radar)
        self.insider = agent(Insider)
        self.market_pulse = agent(MarketPulse)
        self.stock_checker = agent(StockChecker)
        self.stock_executor = StockExecutor(config, live_ack=live_ack)

        # shared
        self.allocator = agent(Allocator)
        self.exit_manager = agent(ExitManager)

        # Only the agents that decide get history; the analysts describe what is
        # in front of them and should not be anchored by old trades.
        for bot in (self.crypto_checker, self.stock_checker, self.exit_manager, self.allocator):
            bot.memory = self.memory

        self.positions: list[Position] = []
        self.min_go_signal = float((config.get("pulse", {}) or {}).get("min_go_signal", 0.3))
        self.weights = config.get("scoring_weights", {}) or {}
        self.crypto_vetoes = config.get("crypto_vetoes", {}) or {}
        self.exits_cfg = config.get("exits", {}) or {}
        crypto_filter = config.get("crypto_filter", {}) or {}
        self.min_checker_confidence = float(
            crypto_filter.get("min_checker_confidence", 0.6)
        )
        self.last_stock_session: date | None = None
        self._lock = asyncio.Lock()
        self._cost_report_every = int(
            (config.get("logging", {}) or {}).get("cost_report_every", 25)
        )
        self._last_cost_report = 0

    # -- helpers -------------------------------------------------------------------

    def _now_et(self) -> datetime:
        """Current New York time. Falls back to a fixed UTC-4 if tzdata is absent."""
        tz_name = (self.config.get("market_hours", {}) or {}).get("timezone", "America/New_York")
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(tz_name))
        except Exception:  # noqa: BLE001 - missing tzdata must not stop the desk
            return datetime.now(timezone.utc) - timedelta(hours=4)

    def maybe_report_costs(self) -> None:
        """Emit a spend snapshot every N model calls."""
        if self._cost_report_every <= 0:
            return
        if self.costs.calls - self._last_cost_report < self._cost_report_every:
            return
        self._last_cost_report = self.costs.calls
        self.log.write("cost", **self.costs.snapshot())

    def refresh_memory(self) -> int:
        """Re-read the log so the next decision sees the latest outcomes."""
        try:
            return len(self.memory.load())
        except Exception as exc:  # noqa: BLE001 - memory is an enhancement, not a gate
            log.warning("could not refresh outcome memory: %s", exc)
            return 0

    def market_is_open(self, now: datetime | None = None) -> bool:
        hours = self.config.get("market_hours", {}) or {}
        now = now or self._now_et()
        if now.weekday() >= 5:
            return False
        open_at = _parse_hhmm(hours.get("open", "09:35"), dtime(9, 35))
        close_at = _parse_hhmm(hours.get("close", "15:55"), dtime(15, 55))
        return open_at <= now.time() <= close_at

    # -- crypto loop ----------------------------------------------------------------

    async def evaluate_token(self, token) -> dict[str, Any]:
        """Toxic/scorecard prescreen -> regime gate -> audit + narrative -> matrix
        -> entry scorecard -> adversarial check -> staged (probe) entry."""
        symbol = token.symbol or token.mint

        # 0. Pure-code gates first: they are free, the models are not.
        toxic = self.toxic.entry_reason(token)
        if toxic is not None:
            self.log.skip(Market.CRYPTO.value, symbol, toxic, {"creator": token.creator})
            return {"bought": False, "reason": toxic}
        pre = self.scorecard.prescreen(token)
        if pre is not None:
            self.log.skip(Market.CRYPTO.value, symbol, pre,
                          {"factors": self.scorecard.code_factors(token)})
            return {"bought": False, "reason": pre}

        # 1. Regime gate (pulse is cached, so this is cheap after the first call).
        pulse = await self.crypto_pulse.run()
        regime = self.regime_gate.assess(pulse)
        if not regime.allow_entries:
            self.log.skip(Market.CRYPTO.value, symbol, regime.reason, regime.signals)
            return {"bought": False, "reason": regime.reason}

        audit, narrative = await asyncio.gather(
            self.auditor.run(token),
            self.narrative.run(token),
        )

        verdict = score_token(
            token,
            audit,
            narrative,
            pulse,
            weights=self.weights.get("crypto"),
            min_go_signal=self.min_go_signal,
            vetoes=self.crypto_vetoes,
        )
        agent_scores = {
            "audit": audit, "narrative": narrative, "pulse": pulse, "matrix": verdict,
            "regime": regime.to_dict(),
            "citations": self.auditor.last_citations + self.narrative.last_citations,
        }
        self.maybe_report_costs()

        if not verdict["buy"]:
            self.log.skip(Market.CRYPTO.value, symbol, verdict["reason"],
                          {"score": verdict["score"]})
            return {"bought": False, "reason": verdict["reason"]}

        if regime.min_score_add > 0:
            tightened = float(verdict.get("threshold", 0.0)) + regime.min_score_add
            if verdict["score"] < tightened:
                self.log.skip(Market.CRYPTO.value, symbol, "below_regime_threshold",
                              {"score": verdict["score"], "threshold": round(tightened, 4),
                               "regime": regime.state})
                return {"bought": False, "reason": "below_regime_threshold"}

        card = self.scorecard.score(token, narrative, audit, min_score_add=regime.min_score_add)
        agent_scores["scorecard"] = card
        if not card["pass"]:
            self.log.skip(Market.CRYPTO.value, symbol, card["reason"],
                          {"scorecard": card["score"], "factors": card["factors"],
                           "threshold": card["threshold"]})
            return {"bought": False, "reason": card["reason"]}

        check = await self.crypto_checker.run(
            {"token": token.model_dump(mode="json"), "audit": audit,
             "narrative": narrative, "pulse": pulse, "score": verdict, "scorecard": card}
        )
        agent_scores["checker"] = check
        agent_scores["citations"] += self.crypto_checker.last_citations
        self.maybe_report_costs()
        if not check["approve"]:
            self.log.skip(Market.CRYPTO.value, symbol, "checker_rejected",
                          {"kill_reasons": check["kill_reasons"]})
            return {"bought": False, "reason": "checker_rejected"}

        confidence = float(check.get("confidence") or 0.0)
        if confidence < self.min_checker_confidence:
            self.log.skip(
                Market.CRYPTO.value, symbol, "checker_low_confidence",
                {"confidence": confidence, "min": self.min_checker_confidence},
            )
            return {"bought": False, "reason": "checker_low_confidence"}

        size_mult = float(regime.size_mult) * float(card.get("size_mult", 1.0))
        return await self._open_crypto(token, verdict, check, agent_scores, size_mult=size_mult)

    def _crypto_exit_levels(self, entry_price: float) -> tuple[float | None, float | None]:
        """Hard stop from exits.stop_loss_pct. Take-profit is the ladder's runner
        cap when the ladder is on, else the legacy exits.take_profit_pct."""
        stop, take_profit = crypto_stop_tp_prices(
            entry_price,
            take_profit_pct=float(self.exits_cfg.get("take_profit_pct", 0.20)),
            stop_loss_pct=float(self.exits_cfg.get("stop_loss_pct", 0.08)),
        )
        if self.ladder.enabled and stop is not None:
            take_profit = self.ladder.runner_take_profit(entry_price)
        return stop, take_profit

    def _sol_usd(self) -> float:
        return float(self.scout.sol_price_usd or 0.0)

    @staticmethod
    def _token_reserves(token) -> tuple[float, float] | None:
        curve_sol = float(getattr(token, "curve_sol", 0) or 0)
        curve_tokens = float(getattr(token, "curve_tokens", 0) or 0)
        if curve_sol > 0 and curve_tokens > 0:
            return curve_sol, curve_tokens
        return None

    async def _crypto_buy_fill(
        self, mint: str, amount_usd: float, price: float,
        reserves: tuple[float, float] | None, bonding_curve: str = "",
    ) -> dict[str, Any]:
        """Dry-run: simulator only. Otherwise the executor (paper sim or live path)."""
        if self.dry_run:
            fill = self.crypto_executor.simulator.buy(
                mint, amount_usd, price, sol_usd=self._sol_usd(), reserves=reserves
            )
            fill["tx_id"] = "DRY_RUN"
            return fill
        return await self.crypto_executor.buy(
            mint, amount_usd, price=price, sol_usd=self._sol_usd(), reserves=reserves,
            bonding_curve=bonding_curve,
        )

    async def _crypto_sell_fill(
        self, position: Position, fraction: float, price: float
    ) -> dict[str, Any]:
        mint = str(position.meta.get("mint", "") or "")
        reserves = self.scout.flow.reserves(mint) if mint else None
        if self.dry_run:
            fill = self.crypto_executor.simulator.sell(
                mint, float(position.quantity) * fraction, price,
                sol_usd=self._sol_usd(), reserves=reserves,
            )
            fill["tx_id"] = "DRY_RUN"
            return fill
        return await self.crypto_executor.sell(
            mint, fraction, price=price, quantity=float(position.quantity),
            sol_usd=self._sol_usd(), reserves=reserves,
        )

    async def _open_crypto(self, token, verdict, check, agent_scores, size_mult: float = 1.0) -> dict[str, Any]:
        symbol = token.symbol or token.mint
        async with self._lock:
            allowed, reason = self.risk.can_open(Market.CRYPTO, self.positions)
            if not allowed:
                self.log.skip(Market.CRYPTO.value, symbol, reason)
                return {"bought": False, "reason": reason}

            planned = self.risk.position_size(
                Market.CRYPTO,
                score=check["adjusted_score"] or verdict["score"],
                liquidity_usd=float(getattr(token, "liquidity_usd", 0) or 0),
            )
            planned = round(planned * max(0.0, float(size_mult)), 2)
            probe_usd, add_usd = self.staged.split(planned)
            if probe_usd <= 0:
                self.log.skip(Market.CRYPTO.value, symbol, "size_zero",
                              {"size_mult": round(size_mult, 4)})
                return {"bought": False, "reason": "size_zero"}

            raw = token.raw or {}
            bonding_curve = str(raw.get("bondingCurveKey") or raw.get("bonding_curve_key") or "")
            quote = curve_price(token)
            try:
                fill = await self._crypto_buy_fill(
                    token.mint, probe_usd, quote, self._token_reserves(token), bonding_curve
                )
            except (NotImplementedError, LiveTradingDisabled) as exc:
                self.log.skip(Market.CRYPTO.value, symbol, "executor_not_implemented", str(exc))
                return {"bought": False, "reason": "executor_not_implemented"}

            if not fill.get("filled"):
                why = str(fill.get("reason") or "not_filled")
                self.log.skip(Market.CRYPTO.value, symbol, why,
                              {"quote": quote, "impact_pct": fill.get("impact_pct")})
                return {"bought": False, "reason": why}

            entry = float(fill.get("price", 0) or 0)
            qty = float(fill.get("quantity", 0) or 0)
            stop_price, take_profit_price = self._crypto_exit_levels(entry)
            now = datetime.now(timezone.utc)
            meta: dict[str, Any] = {
                "mint": token.mint,
                "tx_id": str(fill.get("tx_id", "")),
                "bonding_curve_key": bonding_curve,
                "liquidity_usd": float(token.liquidity_usd or 0),
                "creator": token.creator,
                "fingerprint": fingerprint(token),
                "quote_price": quote,
                "fees_usd": float(fill.get("fees_usd", 0) or 0),
                "realized_pnl_usd": 0.0,
                "grade": (agent_scores.get("scorecard") or {}).get("grade"),
                "regime": (agent_scores.get("regime") or {}).get("state"),
            }
            if self.ladder.enabled:
                meta["ladder"] = self.ladder.init_state(qty)
            if self.staged.enabled and add_usd > 0:
                meta["staged"] = self.staged.init_state(planned, probe_usd, add_usd, entry, now)

            self.risk.record_fill(Market.CRYPTO, probe_usd)
            self.positions.append(
                Position(
                    market=Market.CRYPTO,
                    symbol=symbol,
                    quantity=qty,
                    entry_price=entry,
                    current_price=entry,
                    amount_usd=probe_usd,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    score=verdict["score"],
                    opened_at=now,
                    meta=meta,
                    peak_price=entry if entry > 0 else None,
                )
            )
            agent_scores["entry"] = {
                "planned_usd": planned, "probe_usd": probe_usd, "add_usd": add_usd,
                "size_mult": round(size_mult, 4), "quote_price": quote, "fill_price": entry,
                "impact_pct": fill.get("impact_pct"), "fees_usd": meta["fees_usd"],
                "mode": "dry_run" if self.dry_run else fill.get("mode", "paper"),
            }
            self.log.buy(Market.CRYPTO.value, symbol, verdict["score"],
                         agent_scores, probe_usd, tx_id=meta["tx_id"])
        await self.scout.track_position(token.mint, token.creator)
        result = {
            "bought": True,
            "amount": probe_usd,
            "planned": planned,
            "add_pending": add_usd if "staged" in meta else 0.0,
            "fill_price": entry,
            "tx_id": meta["tx_id"],
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
        }
        if self.dry_run:
            result["dry_run"] = True
        return result

    async def crypto_loop(self) -> None:
        log.info("crypto loop: streaming pump.fun")
        while True:
            try:
                async for token in self.scout.stream():
                    self.risk.maybe_reset_day()
                    await self.evaluate_token(token)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a loop must not die on one bad token
                log.exception("crypto loop error, restarting in 10s")
                await asyncio.sleep(10)

    # -- stock loop -----------------------------------------------------------------

    async def evaluate_stock(self, stock, pulse: dict[str, Any]) -> dict[str, Any]:
        analyst, radar, insider = await asyncio.gather(
            self.analyst.run(stock),
            self.radar.run(stock),
            self.insider.run(stock),
        )

        verdict = score_stock(
            stock, analyst, radar, insider, pulse,
            weights=self.weights.get("stocks"),
            min_go_signal=self.min_go_signal,
        )
        agent_scores = {
            "analyst": analyst, "radar": radar, "insider": insider,
            "pulse": pulse, "matrix": verdict, "sector": stock.sector,
            "citations": (self.analyst.last_citations + self.radar.last_citations
                          + self.insider.last_citations),
        }
        self.maybe_report_costs()

        if not verdict["buy"]:
            self.log.skip(Market.STOCKS.value, stock.symbol, verdict["reason"],
                          {"score": verdict["score"]})
            return {"bought": False, "reason": verdict["reason"]}

        check = await self.stock_checker.run(
            {"stock": stock.model_dump(mode="json"), "analyst": analyst, "radar": radar,
             "insider": insider, "pulse": pulse, "score": verdict}
        )
        agent_scores["checker"] = check
        agent_scores["citations"] += self.stock_checker.last_citations
        self.maybe_report_costs()
        if not check["approve"]:
            self.log.skip(Market.STOCKS.value, stock.symbol, "checker_rejected",
                          {"kill_reasons": check["kill_reasons"]})
            return {"bought": False, "reason": "checker_rejected"}

        return await self._open_stock(stock, verdict, check, agent_scores)

    async def _open_stock(self, stock, verdict, check, agent_scores) -> dict[str, Any]:
        async with self._lock:
            allowed, reason = self.risk.can_open(Market.STOCKS, self.positions, sector=stock.sector)
            if not allowed:
                self.log.skip(Market.STOCKS.value, stock.symbol, reason)
                return {"bought": False, "reason": reason}

            amount = self.risk.position_size(Market.STOCKS, score=check["adjusted_score"] or verdict["score"])
            if amount <= 0:
                self.log.skip(Market.STOCKS.value, stock.symbol, "size_zero")
                return {"bought": False, "reason": "size_zero"}

            if self.dry_run:
                self.log.buy(Market.STOCKS.value, stock.symbol, verdict["score"],
                             agent_scores, amount, tx_id="DRY_RUN")
                return {"bought": True, "dry_run": True, "amount": amount}

            try:
                fill = await self.stock_executor.buy_bracket(
                    stock.symbol,
                    amount,
                    stock.price,
                    stop_pct=check["suggested_stop_pct"],
                    target_pct=check["suggested_target_pct"],
                )
            except OrderRejected as rejection:
                # PDT blocks and wash-trade refusals are broker policy, not bugs.
                self.log.skip(Market.STOCKS.value, stock.symbol, rejection.reason,
                              rejection.detail)
                return {"bought": False, "reason": rejection.reason}

            if not fill.get("filled"):
                self.log.skip(Market.STOCKS.value, stock.symbol, fill.get("reason", "not_filled"))
                return {"bought": False, "reason": fill.get("reason", "not_filled")}

            self.risk.record_fill(Market.STOCKS, fill["amount_usd"])
            self.positions.append(
                Position(
                    market=Market.STOCKS,
                    symbol=stock.symbol,
                    quantity=fill["qty"],
                    entry_price=fill["entry_price"],
                    current_price=fill["entry_price"],
                    amount_usd=fill["amount_usd"],
                    stop_price=fill["stop_price"],
                    take_profit_price=fill["take_profit_price"],
                    sector=stock.sector,
                    score=verdict["score"],
                    meta={"order_id": fill["order_id"]},
                )
            )
            self.log.buy(Market.STOCKS.value, stock.symbol, verdict["score"],
                         agent_scores, fill["amount_usd"], tx_id=fill["order_id"])
            return {"bought": True, "amount": fill["amount_usd"], "order_id": fill["order_id"]}

    async def run_stock_session(self) -> list[dict[str, Any]]:
        """One pass of the equity workflow: pulse -> screen -> evaluate."""
        self.refresh_memory()
        pulse = await self.market_pulse.run()
        if pulse["go_signal"] < self.min_go_signal:
            self.log.skip(Market.STOCKS.value, "*", "veto_market_paused",
                          {"go_signal": pulse["go_signal"]})
            return []

        candidates = await self.screener.run()
        log.info("stock session: %d candidates cleared the screener", len(candidates))
        return [await self.evaluate_stock(stock, pulse) for stock in candidates]

    async def stock_loop(self, poll_seconds: float = 60.0) -> None:
        log.info("stock loop: polling for the RTH window")
        while True:
            try:
                self.risk.maybe_reset_day()
                today = self._now_et().date()
                if self.market_is_open() and self.last_stock_session != today:
                    self.last_stock_session = today
                    await self.run_stock_session()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("stock loop error")
            await asyncio.sleep(poll_seconds)

    # -- exit loop --------------------------------------------------------------------

    async def refresh_positions(self) -> list[Position]:
        """Merge broker truth into our view. Crypto stays desk-side while the
        executor is a stub."""
        try:
            live_stocks = await self.stock_executor.get_positions()
        except Exception as exc:  # noqa: BLE001 - a broker hiccup must not clear the book
            log.warning("could not refresh stock positions: %s", exc)
            return self.positions

        by_symbol = {p.symbol: p for p in self.positions if p.market == Market.STOCKS}
        merged: list[Position] = [p for p in self.positions if p.market == Market.CRYPTO]
        for live in live_stocks:
            known = by_symbol.get(live.symbol)
            if known is not None:
                known.quantity = live.quantity
                known.current_price = live.current_price
                merged.append(known)
            else:
                merged.append(live)
        self.positions = merged
        return self.positions

    async def manage_position(self, position: Position) -> dict[str, Any]:
        decision = await self.exit_manager.run(position)
        action = decision["action"]
        self.log.action(position.symbol, action, decision["reason"],
                        market=position.market.value, pnl_pct=round(position.pnl_pct, 4))

        if action == "HOLD" or self.dry_run:
            return decision

        if position.market == Market.CRYPTO and action in {"TRIM", "CLOSE"}:
            price = float(position.current_price or 0.0)
            if action == "TRIM":
                await self._trim_crypto(
                    position, ExitSignal("TRIM", "llm_trim", float(decision["trim_fraction"])),
                    price, log_action=False,
                )
            else:
                await self._close_crypto(position, "llm_close", price,
                                         mechanical=False, log_action=False)
            return decision

        executor = (
            self.crypto_executor if position.market == Market.CRYPTO else self.stock_executor
        )
        try:
            if action == "TIGHTEN":
                new_stop = position.current_price * (1 - decision["new_stop_pct"])
                if position.market == Market.STOCKS:
                    await executor.tighten_stop(position.meta.get("order_id", ""), new_stop)
                else:
                    await executor.tighten_stop(position.meta.get("mint", ""), new_stop)
                position.stop_price = new_stop

            elif action == "TRIM":
                fraction = decision["trim_fraction"]
                if position.market == Market.STOCKS:
                    await executor.sell_partial(position.symbol, position.quantity * fraction)
                else:
                    await executor.sell(position.meta.get("mint", ""), fraction)
                position.quantity *= 1 - fraction
                position.amount_usd *= 1 - fraction

            elif action == "CLOSE":
                target = (
                    position.symbol
                    if position.market == Market.STOCKS
                    else position.meta.get("mint", "")
                )
                await executor.close_position(target)
                self.risk.record_close(position.market, position.pnl_usd, position.amount_usd)
                self.log.close(position.market.value, position.symbol,
                               round(position.pnl_usd, 2), round(position.hold_time_hours, 2))
                self.positions = [p for p in self.positions if p is not position]

        except NotImplementedError as exc:
            self.log.skip(position.market.value, position.symbol,
                          "executor_not_implemented", str(exc))
        except OrderRejected as rejection:
            self.log.skip(position.market.value, position.symbol,
                          rejection.reason, rejection.detail)
        except Exception as exc:  # noqa: BLE001
            log.exception("failed to %s %s", action, position.symbol)
            self.log.skip(position.market.value, position.symbol, "action_failed", str(exc))

        return decision

    async def run_exit_pass(self) -> list[dict[str, Any]]:
        self.refresh_memory()
        positions = await self.refresh_positions()
        # Mechanical crypto TP/SL fires before the LLM exit manager.
        mechanical = await self.check_open_crypto_stops()
        positions = await self.refresh_positions()
        log.info("exit pass over %d positions (%d mechanical closes)", len(positions), len(mechanical))
        llm = [await self.manage_position(p) for p in list(positions)]
        return mechanical + llm


    def _crypto_mark_price(self, position: Position, prices: dict[str, float] | None = None) -> float:
        """Best available spot: caller override > live flow tape > scout watch > book."""
        mint = str(position.meta.get("mint", "") or "")
        symbol = position.symbol
        if prices:
            if mint and mint in prices:
                return float(prices[mint])
            if symbol in prices:
                return float(prices[symbol])
        if mint:
            live = self.scout.flow.last_price(mint)
            if live > 0:
                return float(live)
        watch = self.scout.watching.get(mint) if mint else None
        if watch is not None and watch.last_price > 0:
            return float(watch.last_price)
        return float(position.current_price or 0.0)

    def _flow_window(self, position: Position):
        mint = str(position.meta.get("mint", "") or "")
        if not mint or not self.scout.flow.is_tracked(mint):
            return None
        return self.scout.flow.stats(mint, float(self.dump_cfg.get("window_seconds", 90)))

    async def _close_crypto(
        self, position: Position, reason: str, price: float, *,
        mechanical: bool = True, emergency: bool = False, log_action: bool = True,
    ) -> dict[str, Any]:
        """Full exit of a crypto position: sell fill, realized PnL net of fees,
        risk bookkeeping, deployer blacklist on rugs, release the flow tape."""
        if price > 0:
            position.current_price = price
        detail = {
            "mechanical": mechanical,
            "emergency": emergency,
            "trigger": reason,
            "price": price,
            "stop_price": position.stop_price,
            "take_profit_price": position.take_profit_price,
            "pnl_pct": round(position.pnl_pct, 4),
        }
        if log_action:
            self.log.action(position.symbol, "CLOSE", reason, market=position.market.value, **detail)

        fill: dict[str, Any] = {}
        try:
            fill = await self._crypto_sell_fill(position, 1.0, price)
        except (NotImplementedError, LiveTradingDisabled) as exc:
            self.log.skip(position.market.value, position.symbol, "executor_not_implemented", str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("crypto close failed for %s", position.symbol)
            self.log.skip(position.market.value, position.symbol, "action_failed", str(exc))
            return {"action": "CLOSE", "reason": reason, "closed": False, "error": str(exc)}

        fill_price = float(fill.get("price") or price or position.current_price or 0.0)
        leg_pnl = 0.0
        if fill_price > 0 and position.entry_price > 0:
            leg_pnl = (fill_price - position.entry_price) * float(position.quantity)
        leg_pnl -= float(fill.get("fees_usd", 0) or 0)
        leg_pnl -= float(position.meta.get("fees_usd", 0) or 0)  # entry-side costs
        total_pnl = float(position.meta.get("realized_pnl_usd", 0) or 0) + leg_pnl

        self.risk.record_close(position.market, leg_pnl, position.amount_usd)
        creator = str(position.meta.get("creator", "") or "")
        self.log.close(
            position.market.value, position.symbol,
            round(total_pnl, 2), round(position.hold_time_hours, 2),
            mechanical=mechanical, trigger=reason, creator=creator,
            fill_price=fill_price, grade=position.meta.get("grade"),
        )
        if creator and self.toxic.should_blacklist(reason) and self.toxic.add_blacklist(creator, reason):
            self.log.write("blacklist", creator=creator, reason=reason,
                           mint=position.meta.get("mint"), symbol=position.symbol)
        self.positions = [p for p in self.positions if p is not position]
        await self.scout.untrack_position(str(position.meta.get("mint", "") or ""))
        return {"action": "CLOSE", "reason": reason, "closed": True, "price": price,
                "fill_price": fill_price, "pnl": round(total_pnl, 4), "emergency": emergency}

    async def apply_crypto_mechanical_exit(
        self, position: Position, reason: str, price: float, emergency: bool = False
    ) -> dict[str, Any]:
        """CLOSE a crypto position on a mechanical trigger. Books even in dry-run."""
        return await self._close_crypto(position, reason, price, mechanical=True, emergency=emergency)

    async def _trim_crypto(
        self, position: Position, signal: ExitSignal, price: float, *, log_action: bool = True
    ) -> dict[str, Any]:
        """Partial exit (ladder rung or LLM TRIM): realize that slice's PnL."""
        fraction = max(0.0, min(1.0, float(signal.fraction)))
        if fraction <= 0:
            return {"action": "TRIM", "reason": signal.reason, "trimmed": False}
        if price > 0:
            position.current_price = price
        try:
            fill = await self._crypto_sell_fill(position, fraction, price)
        except (NotImplementedError, LiveTradingDisabled) as exc:
            self.log.skip(position.market.value, position.symbol, "executor_not_implemented", str(exc))
            return {"action": "TRIM", "reason": signal.reason, "trimmed": False}
        except Exception as exc:  # noqa: BLE001
            log.exception("crypto trim failed for %s", position.symbol)
            self.log.skip(position.market.value, position.symbol, "action_failed", str(exc))
            return {"action": "TRIM", "reason": signal.reason, "trimmed": False, "error": str(exc)}

        sold_qty = float(position.quantity) * fraction
        fill_price = float(fill.get("price") or price or 0.0)
        leg_pnl = (fill_price - position.entry_price) * sold_qty if position.entry_price > 0 else 0.0
        leg_pnl -= float(fill.get("fees_usd", 0) or 0)
        released = float(position.amount_usd) * fraction
        self.risk.record_close(position.market, leg_pnl, released)
        position.quantity = float(position.quantity) - sold_qty
        position.amount_usd = float(position.amount_usd) - released
        position.meta["realized_pnl_usd"] = float(position.meta.get("realized_pnl_usd", 0) or 0) + leg_pnl
        old_stop = position.stop_price
        self.ladder.apply_trim(position, signal)
        if log_action:
            self.log.action(
                position.symbol, "TRIM", signal.reason, market=position.market.value,
                mechanical=True, fraction=round(fraction, 6), price=price, fill_price=fill_price,
                realized_pnl=round(leg_pnl, 4), stop_before=old_stop, stop_after=position.stop_price,
            )
        return {"action": "TRIM", "reason": signal.reason, "trimmed": True, "fraction": fraction,
                "price": price, "fill_price": fill_price, "realized_pnl": round(leg_pnl, 4),
                "new_stop": position.stop_price}

    async def check_staged_adds(
        self, prices: dict[str, float] | None = None, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Scale-in: fill the 60% add on probe positions that confirmed in time."""
        results: list[dict[str, Any]] = []
        if not self.staged.enabled:
            return results
        now = now or datetime.now(timezone.utc)
        async with self._lock:
            for position in list(self.positions):
                state = position.meta.get("staged")
                if position.market != Market.CRYPTO or not isinstance(state, dict):
                    continue
                if state.get("stage") != "probe":
                    continue
                mint = str(position.meta.get("mint", "") or "")
                price = self._crypto_mark_price(position, prices)
                flow = None
                if mint and self.scout.flow.is_tracked(mint):
                    opened = position.opened_at
                    if opened.tzinfo is None:
                        opened = opened.replace(tzinfo=timezone.utc)
                    flow = self.scout.flow.stats_since(mint, opened.timestamp())
                decision = self.staged.evaluate_add(position, price, flow, now)

                if decision.action == "add":
                    regime = self.regime_gate.last
                    if regime is not None and not regime.allow_entries:
                        decision = type(decision)("expire", "regime_paused", decision.gain)
                    elif self.risk.daily_loss_breached():
                        decision = type(decision)("expire", "daily_loss_limit_reached", decision.gain)

                if decision.action == "expire":
                    self.staged.expire(position, decision.reason)
                    self.log.action(position.symbol, "ADD_EXPIRED", decision.reason,
                                    market=position.market.value, gain=round(decision.gain, 4))
                    results.append({"action": "ADD_EXPIRED", "reason": decision.reason})
                    continue
                if decision.action != "add":
                    continue

                add_usd = min(float(state.get("add_usd", 0) or 0),
                              self.risk.remaining_market_budget(Market.CRYPTO))
                add_usd = round(add_usd, 2)
                if add_usd <= 0:
                    self.staged.expire(position, "market_budget_exhausted")
                    results.append({"action": "ADD_EXPIRED", "reason": "market_budget_exhausted"})
                    continue
                reserves = self.scout.flow.reserves(mint) if mint else None
                try:
                    fill = await self._crypto_buy_fill(
                        mint, add_usd, price, reserves, str(position.meta.get("bonding_curve_key", ""))
                    )
                except (NotImplementedError, LiveTradingDisabled) as exc:
                    self.staged.expire(position, "executor_not_implemented")
                    self.log.skip(position.market.value, position.symbol,
                                  "executor_not_implemented", str(exc))
                    continue
                if not fill.get("filled"):
                    # Slippage too wide this tick: wait for the next one inside the window.
                    self.log.skip(position.market.value, position.symbol,
                                  f"add_{fill.get('reason', 'not_filled')}",
                                  {"impact_pct": fill.get("impact_pct")})
                    continue

                fill_price = float(fill.get("price", 0) or 0)
                old_entry = position.entry_price
                self.staged.apply_add(position, add_usd, fill_price, now)
                self.risk.record_fill(Market.CRYPTO, add_usd)
                position.meta["fees_usd"] = float(position.meta.get("fees_usd", 0) or 0) + float(
                    fill.get("fees_usd", 0) or 0
                )
                position.stop_price, position.take_profit_price = self._crypto_exit_levels(
                    position.entry_price
                )
                self.log.action(
                    position.symbol, "ADD", decision.reason, market=position.market.value,
                    add_usd=add_usd, fill_price=fill_price, gain=round(decision.gain, 4),
                    entry_before=old_entry, entry_after=position.entry_price,
                    stop_price=position.stop_price, flow=flow.to_dict() if flow else None,
                )
                results.append({"action": "ADD", "reason": decision.reason, "add_usd": add_usd,
                                "fill_price": fill_price, "entry_price": position.entry_price})
        return results

    async def check_open_crypto_stops(
        self, prices: dict[str, float] | None = None, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Mechanical exits on every open crypto position. Fires before LLM exits.

        Ladder on: emergency dump -> stop -> crash -> runner cap -> rungs (TRIM)
        -> runner trail -> stale -> time-stop. Ladder off: legacy TP/SL/trail/time.
        """
        results: list[dict[str, Any]] = []
        async with self._lock:
            for position in list(self.positions):
                if position.market != Market.CRYPTO:
                    continue
                price = self._crypto_mark_price(position, prices)
                if price > 0:
                    position.current_price = price
                if self.ladder.enabled:
                    signal = self.ladder.evaluate(
                        position, price if price > 0 else None,
                        flow=self._flow_window(position), now=now,
                    )
                else:
                    trigger = check_crypto_exits(
                        position, price if price > 0 else None, exits_cfg=self.exits_cfg, now=now
                    )
                    signal = ExitSignal("CLOSE", trigger, 1.0) if trigger else ExitSignal()
                if signal.action == "HOLD":
                    continue
                mark = price if price > 0 else float(position.current_price or 0)
                if signal.action == "TRIM":
                    results.append(await self._trim_crypto(position, signal, mark))
                else:
                    results.append(
                        await self.apply_crypto_mechanical_exit(
                            position, signal.reason, mark, emergency=signal.emergency
                        )
                    )
        return results

    async def poll_crypto_marks(self) -> dict[str, float]:
        """Refresh marks for open crypto positions via the price poller. Degrades to {}."""
        crypto_positions = [p for p in self.positions if p.market == Market.CRYPTO]
        if not crypto_positions:
            return {}
        try:
            prices = await self.price_poller.fetch_prices_for_positions(crypto_positions)
        except Exception as exc:  # noqa: BLE001
            log.warning("price poller failed (HOLD): %s", exc)
            return {}
        for position in crypto_positions:
            mint = str(position.meta.get("mint", "") or "")
            px = prices.get(mint) if mint else None
            if px is None:
                px = prices.get(position.symbol)
            if px is not None and float(px) > 0:
                position.current_price = float(px)
                peak = float(position.peak_price or position.entry_price or 0.0)
                if float(px) > peak:
                    position.peak_price = float(px)
        return prices

    async def crypto_stops_loop(self) -> None:
        """Poll marks + mechanical exits (default every 30s). Event-driven LLM on big moves."""
        interval = float(self.exits_cfg.get("mechanical_poll_seconds", 30))
        move_trigger = self.exits_cfg.get("crypto_move_trigger_pct")
        log.info("crypto stops loop: every %.0fs", interval)
        while True:
            await asyncio.sleep(interval)
            try:
                prices = await self.poll_crypto_marks()
                adds = await self.check_staged_adds(prices if prices else None)
                if adds:
                    log.info("staged entry events: %d", len(adds))
                closed = await self.check_open_crypto_stops(prices if prices else None)
                if closed:
                    log.info("mechanical exits fired: %d", len(closed))
                # Event-driven LLM exit for crypto only when price moved hard.
                if move_trigger is not None:
                    threshold = float(move_trigger)
                    for position in list(self.positions):
                        if position.market != Market.CRYPTO:
                            continue
                        if abs(position.pnl_pct) >= threshold:
                            await self.manage_position(position)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("crypto stops loop error")

    async def run_crypto_exit_pass(self) -> list[dict[str, Any]]:
        """LLM exit pass over crypto positions only (faster cadence than stocks)."""
        self.refresh_memory()
        await self.poll_crypto_marks()
        await self.check_staged_adds()
        mechanical = await self.check_open_crypto_stops()
        crypto = [p for p in list(self.positions) if p.market == Market.CRYPTO]
        log.info(
            "crypto exit pass over %d positions (%d mechanical)",
            len(crypto), len(mechanical),
        )
        llm = [await self.manage_position(p) for p in crypto]
        return mechanical + llm

    async def crypto_exit_loop(self) -> None:
        """Faster LLM exits for memecoins (default every 20 minutes)."""
        minutes = float(self.exits_cfg.get("crypto_interval_minutes", 20))
        interval = max(60.0, minutes * 60.0)
        log.info("crypto exit loop: every %.0f min", interval / 60.0)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.run_crypto_exit_pass()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("crypto exit loop error")

    async def exit_loop(self) -> None:
        """Stock-focused LLM exit cadence (crypto has its own faster loop)."""
        interval = float(self.exits_cfg.get("interval_hours", 4)) * 3600
        log.info("exit loop (stocks+shared): every %.1f h", interval / 3600)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.run_exit_pass()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("exit loop error")

    # -- allocator loop -----------------------------------------------------------------

    def weekly_pnl(self) -> dict[str, float]:
        """Realised PnL per market over the trailing seven days, from the log."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        totals = {"crypto": 0.0, "stocks": 0.0}
        for record in self.log.read():
            if record.get("type") != "close":
                continue
            try:
                when = datetime.fromisoformat(record["ts"])
            except (KeyError, ValueError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when < cutoff:
                continue
            market = record.get("market", "")
            if market in totals:
                totals[market] += float(record.get("pnl", 0) or 0)
        return totals

    async def run_allocation(self) -> Allocation:
        self.refresh_memory()
        crypto_pulse, market_pulse = await asyncio.gather(
            self.crypto_pulse.run(), self.market_pulse.run()
        )
        allocation = await self.allocator.allocate(
            crypto_pulse, market_pulse, self.weekly_pnl(), risk=self.config.get("risk")
        )
        applied = self.risk.set_allocation(allocation)
        self.log.allocation(round(applied.crypto_pct, 4), round(applied.stocks_pct, 4),
                            allocation.reason)
        self.log.write("cost", **self.costs.snapshot())
        return applied

    async def allocator_loop(self, interval_seconds: float = 86400.0) -> None:
        log.info("allocator loop: daily")
        while True:
            try:
                await self.run_allocation()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("allocator loop error")
            await asyncio.sleep(interval_seconds)

    # -- entry point ---------------------------------------------------------------------

    async def run(self) -> None:
        log.info(
            "desk starting — dry_run=%s, stock execution=%s, models=%s/%s, live_search=%s",
            self.dry_run,
            "paper" if self.stock_executor.paper else "LIVE",
            self.analyst.model,
            self.stock_checker.model,
            self.analyst.live_search,
        )
        log.info("outcome memory: %d closed trades loaded", self.refresh_memory())
        await asyncio.gather(
            self.crypto_loop(),
            self.stock_loop(),
            self.exit_loop(),
            self.crypto_exit_loop(),
            self.crypto_stops_loop(),
            self.allocator_loop(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grok Trading Desk")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="decide and log, never execute")
    parser.add_argument(
        "--i-understand-the-risk",
        action="store_true",
        dest="live_ack",
        help='required, together with mode: "live" in the config, to leave paper trading',
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    desk = TradingDesk(load_config(args.config), dry_run=args.dry_run, live_ack=args.live_ack)
    try:
        asyncio.run(desk.run())
    except KeyboardInterrupt:
        log.info("desk stopped")


if __name__ == "__main__":
    main()
