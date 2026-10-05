"""Price poller for open crypto positions.

Mechanical TP/SL (and trailing / time-stops) need a live mark even when the
scout is no longer watching the mint. Sources, in order:

  1. Injected `fetch` callable (unit tests)
  2. Optional HTTP JSON endpoint (`exits.price_url_template` with `{mint}`)
  3. Best-effort Solana RPC read of a bonding-curve account (meta key)

Any failure logs and returns no price for that mint — the desk HOLDs rather
than closing on a stale or missing mark.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

FetchPrices = Callable[[list[str]], Awaitable[dict[str, float]]]


class CryptoPricePoller:
    """Fetch mint → spot price (same units as position.entry_price)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        fetch: FetchPrices | None = None,
        rpc: Any | None = None,
    ):
        self.config = config or {}
        exits = self.config.get("exits", {}) or {}
        solana = self.config.get("solana", {}) or {}
        pump = self.config.get("pump_fun", {}) or {}

        self.price_url_template = str(exits.get("price_url_template") or "").strip()
        self.sol_price_usd = float(pump.get("sol_price_usd", 0) or 0)
        self.rpc_url = str(solana.get("rpc_url") or "")
        self._fetch = fetch
        self._rpc = rpc

    async def fetch_prices(self, mints: list[str]) -> dict[str, float]:
        """Return {mint: price}. Missing mints simply omit the key (caller HOLDs)."""
        clean = [m for m in mints if m]
        if not clean:
            return {}

        if self._fetch is not None:
            try:
                return {k: float(v) for k, v in (await self._fetch(clean)).items() if v}
            except Exception as exc:  # noqa: BLE001
                log.warning("price poller: injected fetch failed: %s", exc)
                return {}

        out: dict[str, float] = {}
        if self.price_url_template:
            out.update(await self._http_prices(clean))
        return out

    async def fetch_prices_for_positions(
        self, positions: list[Any], meta_curves: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Resolve prices for desk positions; try RPC bonding curves when meta has keys."""
        mints = []
        for pos in positions:
            mint = str(getattr(pos, "meta", {}).get("mint", "") or "")
            if mint:
                mints.append(mint)
        prices = await self.fetch_prices(mints)

        # Best-effort RPC fill for anything still missing.
        missing = [m for m in mints if m not in prices]
        if missing and (self._rpc is not None or self.rpc_url):
            curves = meta_curves or {}
            for pos in positions:
                mint = str(getattr(pos, "meta", {}).get("mint", "") or "")
                if not mint or mint in prices:
                    continue
                curve = (
                    getattr(pos, "meta", {}).get("bonding_curve_key")
                    or curves.get(mint)
                    or ""
                )
                if not curve:
                    continue
                px = await self._rpc_curve_price(str(curve))
                if px is not None and px > 0:
                    prices[mint] = px
        return prices

    async def _http_prices(self, mints: list[str]) -> dict[str, float]:
        try:
            import httpx
        except ImportError:
            log.warning("price poller: httpx unavailable")
            return {}

        out: dict[str, float] = {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            for mint in mints:
                url = self.price_url_template.format(mint=mint)
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    payload = resp.json()
                    price = _extract_price(payload, mint, self.sol_price_usd)
                    if price is not None and price > 0:
                        out[mint] = price
                except Exception as exc:  # noqa: BLE001
                    log.warning("price poller: HTTP %s failed: %s", mint, exc)
        return out

    async def _rpc_curve_price(self, bonding_curve_key: str) -> float | None:
        """Decode pump.fun-style virtual reserves from a bonding-curve account.

        Layout (after 8-byte discriminator): virtual_token_reserves u64,
        virtual_sol_reserves u64, … Price = sol_reserves / token_reserves
        (raw units — same ratio the scout uses from the WS feed).
        """
        try:
            from solders.pubkey import Pubkey

            client = self._rpc
            if client is None:
                from solana.rpc.async_api import AsyncClient

                client = AsyncClient(self.rpc_url)
                owns = True
            else:
                owns = False
            try:
                info = await client.get_account_info(Pubkey.from_string(bonding_curve_key))
            finally:
                if owns:
                    await client.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("price poller: RPC curve %s failed: %s", bonding_curve_key, exc)
            return None

        value = getattr(info, "value", info)
        if value is None:
            return None
        data = getattr(value, "data", None)
        if isinstance(data, (tuple, list)):
            data = data[0] if data else None
        if data is None:
            return None
        try:
            raw = bytes(data)
        except (TypeError, ValueError):
            return None
        if len(raw) < 24:
            return None
        # Skip 8-byte Anchor discriminator
        try:
            virtual_token = struct.unpack_from("<Q", raw, 8)[0]
            virtual_sol = struct.unpack_from("<Q", raw, 16)[0]
        except struct.error:
            return None
        if virtual_token <= 0 or virtual_sol <= 0:
            return None
        return virtual_sol / virtual_token


def _extract_price(payload: Any, mint: str, sol_usd: float) -> float | None:
    """Best-effort parse of common price JSON shapes into SOL-per-token."""
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if not isinstance(payload, dict):
        return None

    # Direct fields
    for key in ("price", "price_sol", "priceSol", "last_price", "lastPrice"):
        if payload.get(key) is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                pass

    # {mint: price} or {mint: {price: …}}
    node = payload.get(mint) or payload.get("data") or payload.get("pairs")
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        for key in ("price", "price_sol", "priceSol", "priceNative", "priceUsd"):
            if node.get(key) is None:
                continue
            try:
                val = float(node[key])
            except (TypeError, ValueError):
                continue
            if key in {"priceUsd", "price_usd"} and sol_usd > 0:
                return val / sol_usd
            return val
    if isinstance(node, list) and node:
        first = node[0]
        if isinstance(first, dict):
            for key in ("priceNative", "priceUsd", "price"):
                if first.get(key) is None:
                    continue
                try:
                    val = float(first[key])
                except (TypeError, ValueError):
                    continue
                if key == "priceUsd" and sol_usd > 0:
                    return val / sol_usd
                return val
    return None
