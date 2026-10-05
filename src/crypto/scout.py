"""Bot 1 — pump.fun scout. Pure code, no LLM.

Two stages, because a `subscribeNewToken` event does not contain the facts the
desk wants to filter on. It carries the curve reserves, the market cap in SOL,
the deployer's opening buy and the metadata URI — and nothing else. There are no
holders, no buy/sell counts and no age: at creation, age is zero and the holder
count is one, by construction.

So:

  stage 1  `screen_launch`  judge the create event on what it really carries
  stage 2  `Watchlist`      subscribe to that mint's trades, accumulate real
                            buy/sell counts, unique traders and price action for
                            a fixed window, then judge again

Only a token that survives both is worth spending model calls on. Trade
subscriptions are metered (0.01 SOL per 10k messages), so the watchlist is
capacity-bounded and evicts on completion.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections.abc import AsyncIterator
from typing import Any

from ..models import Token
from .flow import FlowTracker
from .toxic_flow import ToxicFlowFilter

log = logging.getLogger(__name__)

LAMPORTS_PER_SOL = 1_000_000_000


def parse_create_event(payload: dict[str, Any]) -> Token:
    """Map a PumpPortal `create` event onto our Token model."""

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return default

    def num(*keys: str) -> float:
        try:
            return float(pick(*keys, default=0) or 0)
        except (TypeError, ValueError):
            return 0.0

    socials = {
        key: str(payload[key])
        for key in ("twitter", "telegram", "website")
        if payload.get(key)
    }
    socials.update(
        {k: str(v) for k, v in (payload.get("socials") or {}).items() if v}
    )

    return Token(
        mint=str(pick("mint", "mintAddress", "ca", default="")),
        symbol=str(pick("symbol", "ticker", default="")),
        name=str(pick("name", default="")),
        creator=str(pick("creator", "traderPublicKey", "dev", default="")),
        curve_sol=num("vSolInBondingCurve"),
        curve_tokens=num("vTokensInBondingCurve"),
        market_cap_sol=num("marketCapSol"),
        # `solAmount` is what the deployer spent on the opening buy; `initialBuy`
        # is the token quantity they received, which is not comparable in SOL.
        dev_initial_buy_sol=num("solAmount"),
        uri=str(pick("uri", default="")),
        pool=str(pick("pool", default="")),
        socials=socials,
        raw=payload,
    )


def launch_reason(token: Token, filt: dict[str, Any]) -> str | None:
    """Stage one. Reject on the facts a create event actually carries."""
    if not token.mint:
        return "no_mint"

    min_curve = filt.get("min_curve_sol")
    if min_curve is not None and token.curve_sol < min_curve:
        return "curve_too_small"

    max_curve = filt.get("max_curve_sol")
    if max_curve is not None and token.curve_sol > max_curve:
        return "curve_too_large"

    min_cap = filt.get("min_market_cap_sol")
    if min_cap is not None and token.market_cap_sol < min_cap:
        return "market_cap_too_small"

    max_cap = filt.get("max_market_cap_sol")
    if max_cap is not None and token.market_cap_sol > max_cap:
        return "market_cap_too_large"

    max_dev_buy = filt.get("max_dev_initial_buy_sol")
    if max_dev_buy is not None and token.dev_initial_buy_sol > max_dev_buy:
        return "dev_initial_buy_too_large"

    if filt.get("require_metadata") and not (token.uri and token.name and token.symbol):
        return "incomplete_metadata"

    if filt.get("require_socials") and not token.socials:
        return "no_socials"

    return None


def filter_reason(token: Token, filt: dict[str, Any]) -> str | None:
    """Stage two. Reject on what the watch window measured.

    Every threshold is optional: a missing config key means the desk does not
    care about that dimension, not that everything fails it.
    """
    if not token.mint:
        return "no_mint"

    min_liq = filt.get("min_liquidity_usd")
    if min_liq is not None and token.liquidity_usd < min_liq:
        return "liquidity_too_low"

    max_liq = filt.get("max_liquidity_usd")
    if max_liq is not None and token.liquidity_usd > max_liq:
        return "liquidity_too_high"

    # Holders come from the on-chain enricher; unique_traders from the watch
    # window. Only enforce min_holders when a real count exists.
    min_holders = filt.get("min_holders")
    if min_holders is not None and token.holders_known and token.holders < min_holders:
        return "too_few_holders"

    min_traders = filt.get("min_unique_traders")
    if min_traders is not None and token.unique_traders < min_traders:
        return "too_few_traders"

    max_top10 = filt.get("max_top10_holder_pct")
    if max_top10 is not None and token.top10_holder_pct > max_top10:
        return "top10_concentration"

    max_dev = filt.get("max_dev_holding_pct")
    if max_dev is not None and token.dev_holding_pct > max_dev:
        return "dev_holding_too_high"

    min_age = filt.get("min_age_seconds")
    if min_age is not None and token.age_seconds < min_age:
        return "too_young"

    max_age = filt.get("max_age_seconds")
    if max_age is not None and token.age_seconds > max_age:
        return "too_old"

    min_buys = filt.get("min_buys")
    if min_buys is not None and token.buys < min_buys:
        return "too_few_buys"

    min_ratio = filt.get("min_buy_sell_ratio")
    if min_ratio is not None and token.buy_sell_ratio < min_ratio:
        return "weak_buy_sell_ratio"

    if token.dev_sold:
        return "dev_sold"

    # Hard requires: unknown must NOT silently pass. Enrich before filtering;
    # if still None after enrichment, reject with mint_unknown / lp_unknown.
    if filt.get("require_mint_revoked"):
        if token.mint_revoked is None:
            return "mint_unknown"
        if token.mint_revoked is False:
            return "mint_not_revoked"

    if filt.get("require_lp_burned"):
        if token.lp_burned is None:
            return "lp_unknown"
        if token.lp_burned is False:
            return "lp_not_burned"

    # Anti-chase: reject tokens that already ran (or dumped) during the watch window.
    # Missing keys mean "no opinion" so old configs keep working unchanged.
    max_pump = filt.get("max_window_pump_pct")
    if max_pump is not None and token.price_change_pct > float(max_pump):
        return "window_pump_too_high"

    min_pump = filt.get("min_window_pump_pct")
    if min_pump is not None and token.price_change_pct < float(min_pump):
        return "window_dump_too_hard"

    return None


def passes_filter(token: Token, filt: dict[str, Any]) -> bool:
    return filter_reason(token, filt) is None


class Watch:
    """One token under observation, accumulating its own trade stats."""

    def __init__(self, token: Token, window_seconds: float, now: float | None = None):
        self.token = token
        self.window_seconds = window_seconds
        self.started_at = now if now is not None else time.monotonic()
        self.traders: set[str] = set()
        self.buys = 0
        self.sells = 0
        self.volume_sol = 0.0
        self.first_price = self._price(token)
        self.last_price = self.first_price
        self.dev_sold = False

    @staticmethod
    def _price(token: Token) -> float:
        if token.curve_tokens <= 0:
            return 0.0
        return token.curve_sol / token.curve_tokens

    def record(self, event: dict[str, Any]) -> None:
        """Fold one trade event into the running stats."""
        side = str(event.get("txType", "")).lower()
        trader = str(event.get("traderPublicKey", ""))

        if side == "buy":
            self.buys += 1
        elif side == "sell":
            self.sells += 1
            if trader and trader == self.token.creator:
                # The deployer selling into their own launch ends the evaluation.
                self.dev_sold = True
        else:
            return

        if trader:
            self.traders.add(trader)

        try:
            self.volume_sol += abs(float(event.get("solAmount", 0) or 0))
        except (TypeError, ValueError):
            pass

        curve_sol = event.get("vSolInBondingCurve")
        curve_tokens = event.get("vTokensInBondingCurve")
        if curve_sol and curve_tokens:
            try:
                self.token.curve_sol = float(curve_sol)
                self.token.curve_tokens = float(curve_tokens)
                self.last_price = float(curve_sol) / float(curve_tokens)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        if event.get("marketCapSol"):
            try:
                self.token.market_cap_sol = float(event["marketCapSol"])
            except (TypeError, ValueError):
                pass

    def expired(self, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        return (now - self.started_at) >= self.window_seconds

    def result(self, sol_usd: float, now: float | None = None) -> Token:
        """The token as it looks at the end of the window."""
        now = now if now is not None else time.monotonic()
        elapsed = now - self.started_at
        change = 0.0
        if self.first_price > 0:
            change = (self.last_price - self.first_price) / self.first_price

        token = self.token.model_copy(
            update={
                "buys": self.buys,
                "sells": self.sells,
                "unique_traders": len(self.traders),
                # holders left for OnChainEnricher — do not proxy unique_traders
                "observed_seconds": round(elapsed, 2),
                "age_seconds": round(elapsed, 2),
                "volume_sol": round(self.volume_sol, 6),
                "price_change_pct": round(change, 4),
                "dev_sold": self.dev_sold,
            }
        )
        return token.priced(sol_usd)


class Scout:
    """Bot 1. Owns the socket, the launch filter and the watchlist."""

    name = "scout"

    def __init__(
        self,
        config: dict[str, Any],
        connect=None,
        enricher=None,
        toxic: ToxicFlowFilter | None = None,
        flow: FlowTracker | None = None,
    ):
        self.config = config or {}
        self.launch_filter = self.config.get("crypto_launch_filter", {}) or {}
        self.filter = self.config.get("crypto_filter", {}) or {}
        self.enricher = enricher
        # Toxic-flow memory sees every create event (filtered or not). Only on
        # when injected or explicitly enabled, so bare configs keep old behaviour.
        if toxic is None and (self.config.get("crypto_toxic", {}) or {}).get("enabled"):
            toxic = ToxicFlowFilter(self.config)
        self.toxic = toxic
        # Post-entry tape for open positions (staged adds + dump detection).
        self.flow = flow or FlowTracker()
        self.reject_counts: dict[str, int] = {}

        pump = self.config.get("pump_fun", {}) or {}
        self.ws_url = pump.get("ws_url", "wss://pumpportal.fun/api/data")
        if pump.get("api_key"):
            self.ws_url = f"{self.ws_url}?api-key={pump['api_key']}"
        self.sol_price_usd = float(pump.get("sol_price_usd", 0) or 0)

        watch = pump.get("watch", {}) or {}
        self.watch_enabled = bool(watch.get("enabled", True))
        self.window_seconds = float(watch.get("window_seconds", 300))
        self.max_concurrent = int(watch.get("max_concurrent", 40))
        self.min_trades_to_score = int(watch.get("min_trades_to_score", 12))

        self._connect = connect
        self.seen: set[str] = set()
        self.watching: dict[str, Watch] = {}
        self._ws: Any = None

    def _connector(self):
        if self._connect is not None:
            return self._connect
        import websockets  # imported lazily so tests never need the dependency

        return websockets.connect

    # -- message handling -----------------------------------------------------------

    def handle_create(self, payload: dict[str, Any]) -> Token | None:
        """Stage one. Returns the token if it is worth watching."""
        token = parse_create_event(payload)
        if not token.mint or token.mint in self.seen:
            return None
        self.seen.add(token.mint)
        if self.toxic is not None:
            self.toxic.observe(token)

        if launch_reason(token, self.launch_filter) is not None:
            return None
        if self.toxic is not None:
            toxic = self.toxic.reason(token)
            if toxic is not None:
                self.reject_counts[toxic] = self.reject_counts.get(toxic, 0) + 1
                log.info("toxic reject %s (%s): %s", token.mint, token.creator, toxic)
                return None
        if len(self.watching) >= self.max_concurrent:
            # Metered subscription: better to miss one than to blow the budget.
            log.debug("watchlist full, skipping %s", token.mint)
            return None
        return token

    def handle_trade(self, payload: dict[str, Any]) -> None:
        """Fold a trade event into whichever watch it belongs to."""
        mint = str(payload.get("mint", ""))
        watch = self.watching.get(mint)
        if watch is not None:
            watch.record(payload)
        self.flow.record(payload)

    async def track_position(self, mint: str, creator: str = "") -> None:
        """Keep (or start) the trade subscription for an open position."""
        if not mint:
            return
        already = self.flow.is_tracked(mint) or mint in self.watching
        self.flow.track(mint, creator)
        if not already:
            await self._subscribe_trades(mint)

    async def untrack_position(self, mint: str) -> None:
        if not mint or not self.flow.is_tracked(mint):
            return
        self.flow.untrack(mint)
        if mint not in self.watching:
            await self._unsubscribe_trades([mint])

    async def mature(self, sol_usd: float | None = None, now: float | None = None) -> list[Token]:
        """Pop every watch whose window has closed, enrich, then filter survivors."""
        sol_usd = self.sol_price_usd if sol_usd is None else sol_usd
        ready: list[Token] = []
        for mint, watch in list(self.watching.items()):
            if not (watch.expired(now) or watch.dev_sold):
                continue
            del self.watching[mint]
            if watch.dev_sold:
                log.info("dropping %s: deployer sold during the window", mint)
                continue
            token = watch.result(sol_usd, now)
            if token.trades < self.min_trades_to_score:
                continue
            if self.enricher is not None:
                try:
                    token = await self.enricher.enrich(token)
                except Exception as exc:  # noqa: BLE001 - enricher must not kill the scout
                    log.warning("enricher failed for %s: %s", mint, exc)
            reason = filter_reason(token, self.filter)
            if reason is None:
                ready.append(token)
            else:
                log.info("stage-two reject %s: %s", mint, reason)
        return ready

    # -- socket -----------------------------------------------------------------------

    async def _subscribe_trades(self, mint: str) -> None:
        if self._ws is None:
            return
        with contextlib.suppress(Exception):
            await self._ws.send(
                json.dumps({"method": "subscribeTokenTrade", "keys": [mint]})
            )

    async def _unsubscribe_trades(self, mints: list[str]) -> None:
        if self._ws is None or not mints:
            return
        with contextlib.suppress(Exception):
            await self._ws.send(
                json.dumps({"method": "unsubscribeTokenTrade", "keys": mints})
            )

    async def stream(self, max_backoff: float = 300.0) -> AsyncIterator[Token]:
        """Yield matured, filtered tokens forever.

        PumpPortal bans clients that reconnect aggressively, so the backoff is
        exponential with jitter and a ceiling — not a flat retry.
        """
        connect = self._connector()
        attempt = 0
        while True:
            try:
                async with connect(self.ws_url) as ws:
                    self._ws = ws
                    attempt = 0
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    log.info("scout subscribed to new tokens")
                    if self.flow.tracked:
                        # Reconnect: open positions must not lose their tape.
                        await ws.send(
                            json.dumps({"method": "subscribeTokenTrade", "keys": self.flow.tracked})
                        )

                    async for message in ws:
                        for token in await self._on_message(message):
                            yield token
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dropped socket is routine
                delay = min(2**attempt, max_backoff) * (0.5 + random.random())
                attempt += 1
                log.warning("scout socket error, reconnecting in %.1fs: %s", delay, exc)
                await asyncio.sleep(delay)
            finally:
                self._ws = None
                self.watching.clear()

    async def _on_message(self, message: str | bytes | dict[str, Any]) -> list[Token]:
        """Route one frame, then harvest anything whose window just closed."""
        payload = self._decode(message)
        if payload is None:
            return []

        tx_type = str(payload.get("txType", "")).lower()
        if tx_type == "create":
            token = self.handle_create(payload)
            if token is not None:
                if not self.watch_enabled:
                    return [token.priced(self.sol_price_usd)]
                self.watching[token.mint] = Watch(token, self.window_seconds)
                await self._subscribe_trades(token.mint)
        elif tx_type in {"buy", "sell"}:
            self.handle_trade(payload)

        before = set(self.watching)
        ready = await self.mature()
        finished = [m for m in before - set(self.watching) if not self.flow.is_tracked(m)]
        await self._unsubscribe_trades(finished)
        return ready

    @staticmethod
    def _decode(message: str | bytes | dict[str, Any]) -> dict[str, Any] | None:
        try:
            payload = message if isinstance(message, dict) else json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None
