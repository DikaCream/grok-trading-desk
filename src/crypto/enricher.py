"""On-chain enricher — fill Token fields the PumpPortal create feed never carries.

Runs after the watch window and before stage-two `filter_reason`. Prefer Solana
RPC via solana/solders; every field is best-effort. Semantics matter:

  None  = unknown (not checked / RPC failed / not discoverable)
  False = checked and negative (mint authority still live, LP not burned, …)
  True  = checked and positive

Hard requires in the stage-two filter must NOT treat None as a pass: when
`require_mint_revoked` is set and mint_revoked is still None after enrichment,
the filter rejects with `mint_unknown` (same for LP → `lp_unknown`).

The RPC client is injectable so unit tests never hit the network.
"""

from __future__ import annotations

import logging
import struct
from typing import Any, Protocol, runtime_checkable

from solders.pubkey import Pubkey

from ..models import Token

log = logging.getLogger(__name__)

TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
# SPL Mint account is 82 bytes; COption<Pubkey> mint_authority sits at offset 0.
_MINT_SIZE = 82
_COPTION_NONE = 0
_COPTION_SOME = 1


@runtime_checkable
class RpcClient(Protocol):
    """Minimal async RPC surface the enricher needs (AsyncClient-compatible)."""

    async def get_account_info(self, pubkey: Pubkey, *args: Any, **kwargs: Any) -> Any: ...

    async def get_token_largest_accounts(self, pubkey: Pubkey, *args: Any, **kwargs: Any) -> Any: ...

    async def get_token_supply(self, pubkey: Pubkey, *args: Any, **kwargs: Any) -> Any: ...

    async def close(self) -> None: ...


def parse_mint_revoked(data: bytes | bytearray | list[int] | None) -> bool | None:
    """True if mint authority is None (revoked), False if set, None if unreadable."""
    if data is None:
        return None
    raw = bytes(data) if not isinstance(data, (bytes, bytearray)) else bytes(data)
    if len(raw) < 4:
        return None
    tag = int.from_bytes(raw[0:4], "little")
    if tag == _COPTION_NONE:
        return True
    if tag == _COPTION_SOME:
        return False
    return None


def parse_mint_supply(data: bytes | bytearray | list[int] | None) -> int | None:
    """Raw token supply (base units) from an SPL Mint account, or None."""
    if data is None:
        return None
    raw = bytes(data) if not isinstance(data, (bytes, bytearray)) else bytes(data)
    if len(raw) < 44:
        return None
    return struct.unpack_from("<Q", raw, 36)[0]


def _account_data(value: Any) -> bytes | None:
    """Pull raw bytes out of a get_account_info response (solders or mock)."""
    if value is None:
        return None
    # solders GetAccountInfoResp → .value is Optional[Account]
    account = getattr(value, "value", value)
    if account is None:
        return None
    data = getattr(account, "data", None)
    if data is None and isinstance(account, dict):
        data = account.get("data")
    if data is None:
        return None
    # solders Account.data is bytes; some encodings wrap (data, encoding)
    if isinstance(data, (tuple, list)) and data:
        data = data[0]
    if isinstance(data, str):
        # base64 — leave unparsed; callers that need it should request base64 decode upstream
        return None
    try:
        return bytes(data)
    except (TypeError, ValueError):
        return None


def _ui_amount(entry: Any) -> float:
    """Token amount as float from a largest-accounts entry (solders or dict)."""
    if entry is None:
        return 0.0
    if isinstance(entry, dict):
        amount = entry.get("ui_amount")
        if amount is None:
            amount = entry.get("uiAmount")
        if amount is not None:
            try:
                return float(amount)
            except (TypeError, ValueError):
                return 0.0
        raw = entry.get("amount")
        decimals = int(entry.get("decimals") or 0)
        try:
            return float(raw) / (10 ** decimals) if raw is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    # solders UiTokenAmount-like
    for attr in ("ui_amount", "ui_amount_string", "amount"):
        val = getattr(entry, attr, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    amount_obj = getattr(entry, "amount", None) or getattr(entry, "ui_token_amount", None)
    if amount_obj is not None and amount_obj is not entry:
        return _ui_amount(amount_obj)
    return 0.0


def _largest_entries(resp: Any) -> list[Any]:
    value = getattr(resp, "value", resp)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value) if value else []


TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


def _entry_address(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("address") or "")
    addr = getattr(entry, "address", None)
    return str(addr) if addr is not None else ""


def curve_token_accounts(token: Token) -> set[str]:
    """Addresses that hold the bonding curve's unsold supply (not real holders).

    Pre-graduation the curve's associated token account holds most of the supply;
    counting it would make top-10 concentration ~80%+ on every launch. Covers
    both SPL Token and Token-2022 ATAs of the curve PDA.
    """
    out: set[str] = set()
    raw = token.raw or {}
    curve = str(raw.get("bondingCurveKey") or raw.get("bonding_curve_key") or "")
    try:
        from solders.token.associated import get_associated_token_address

        mint_pk = Pubkey.from_string(token.mint)
        if not curve:
            from .crypto_executor import derive_bonding_curve

            curve = derive_bonding_curve(token.mint) or ""
        if not curve:
            return out
        curve_pk = Pubkey.from_string(curve)
        out.add(curve)
        for program in (TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID):
            out.add(str(get_associated_token_address(curve_pk, mint_pk, Pubkey.from_string(program))))
        assoc = raw.get("associatedBondingCurve") or raw.get("associated_bonding_curve")
        if assoc:
            out.add(str(assoc))
    except Exception:  # noqa: BLE001 - fake mints in tests
        pass
    return out


def _supply_ui(resp: Any) -> float | None:
    value = getattr(resp, "value", resp)
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("ui_amount", "uiAmount"):
            if value.get(key) is not None:
                try:
                    return float(value[key])
                except (TypeError, ValueError):
                    return None
        return None
    for attr in ("ui_amount", "ui_amount_string"):
        val = getattr(value, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return _ui_amount(value)


def build_mint_account_data(*, authority: Pubkey | None, supply: int = 1_000_000_000) -> bytes:
    """Test helper: minimal SPL Mint account bytes."""
    buf = bytearray(_MINT_SIZE)
    if authority is None:
        struct.pack_into("<I", buf, 0, _COPTION_NONE)
    else:
        struct.pack_into("<I", buf, 0, _COPTION_SOME)
        buf[4:36] = bytes(authority)
    struct.pack_into("<Q", buf, 36, supply)
    buf[44] = 6  # decimals
    struct.pack_into("<I", buf, 45, 1)  # is_initialized
    return bytes(buf)


class OnChainEnricher:
    """Fill holder / authority fields on a Token. Safe to call with a FakeRpc."""

    def __init__(self, config: dict[str, Any] | None = None, rpc: RpcClient | None = None):
        self.config = config or {}
        solana = self.config.get("solana", {}) or {}
        self.rpc_url = str(solana.get("rpc_url") or "https://api.mainnet-beta.solana.com")
        self._rpc = rpc
        self._owns_client = False

    async def _client(self) -> RpcClient:
        if self._rpc is not None:
            return self._rpc
        from solana.rpc.async_api import AsyncClient

        self._rpc = AsyncClient(self.rpc_url)
        self._owns_client = True
        return self._rpc

    async def aclose(self) -> None:
        if self._owns_client and self._rpc is not None:
            with_context = getattr(self._rpc, "close", None)
            if with_context is not None:
                await with_context()
            self._rpc = None
            self._owns_client = False

    async def enrich(self, token: Token) -> Token:
        """Return a copy with on-chain fields filled. Never raises — degrades to None/0."""
        if not token.mint:
            return token

        updates: dict[str, Any] = {}
        try:
            client = await self._client()
            mint_pk = Pubkey.from_string(token.mint)
        except Exception as exc:  # noqa: BLE001
            log.warning("enricher: bad mint %s (%s)", token.mint, exc)
            return token

        # --- mint authority -------------------------------------------------------
        try:
            info = await client.get_account_info(mint_pk)
            data = _account_data(info)
            revoked = parse_mint_revoked(data)
            if revoked is not None:
                updates["mint_revoked"] = revoked
            supply_raw = parse_mint_supply(data)
        except Exception as exc:  # noqa: BLE001
            log.warning("enricher: mint account %s failed: %s", token.mint, exc)
            supply_raw = None

        # --- supply + largest holders --------------------------------------------
        supply_ui: float | None = None
        try:
            supply_resp = await client.get_token_supply(mint_pk)
            supply_ui = _supply_ui(supply_resp)
        except Exception as exc:  # noqa: BLE001
            log.debug("enricher: token supply %s failed: %s", token.mint, exc)

        if supply_ui is None and supply_raw is not None:
            # decimals unknown → treat raw as ui only when we have no better signal
            supply_ui = float(supply_raw)

        try:
            largest = await client.get_token_largest_accounts(mint_pk)
            entries = _largest_entries(largest)
        except Exception as exc:  # noqa: BLE001
            log.warning("enricher: largest accounts %s failed: %s", token.mint, exc)
            entries = []

        returned = len(entries)  # RPC caps the list at 20; judge exactness on this
        excluded = curve_token_accounts(token)
        if excluded:
            entries = [e for e in entries if _entry_address(e) not in excluded]

        amounts = [_ui_amount(getattr(e, "amount", e) if not isinstance(e, dict) else e) for e in entries]
        # solders wraps UiTokenAmount under .amount; also accept entry itself
        if entries and all(a == 0 for a in amounts):
            amounts = [_ui_amount(e) for e in entries]

        if supply_ui and supply_ui > 0 and amounts:
            top10 = sum(sorted(amounts, reverse=True)[:10])
            updates["top10_holder_pct"] = round(min(1.0, top10 / supply_ui), 4)

        # Holder count: exact when < 20 largest accounts returned, else unknown.
        if returned:
            if returned < 20:
                updates["holders"] = len(entries)
                updates["holders_known"] = True
            else:
                # At least 20; exact count needs a heavier index — leave unknown.
                updates["holders_known"] = False

        # --- deployer / creator holding ------------------------------------------
        creator = (token.creator or "").strip()
        if creator and supply_ui and supply_ui > 0:
            try:
                from solders.token.associated import get_associated_token_address

                creator_pk = Pubkey.from_string(creator)
                ata = get_associated_token_address(creator_pk, mint_pk)
                ata_info = await client.get_account_info(ata)
                ata_data = _account_data(ata_info)
                if ata_data is not None and len(ata_data) >= 72:
                    # SPL token account: amount u64 at offset 64
                    raw_amt = struct.unpack_from("<Q", ata_data, 64)[0]
                    # Prefer ui via decimals from mint if we parsed it
                    decimals = 6
                    mint_data = None
                    try:
                        mint_data = _account_data(await client.get_account_info(mint_pk))
                    except Exception:  # noqa: BLE001
                        pass
                    if mint_data is not None and len(mint_data) > 44:
                        decimals = mint_data[44]
                    ui_amt = raw_amt / (10 ** decimals)
                    updates["dev_holding_pct"] = round(min(1.0, ui_amt / supply_ui), 4)
                elif ata_info is not None and getattr(ata_info, "value", ata_info) is None:
                    # ATA missing → deployer holds nothing on-chain (via ATA)
                    updates["dev_holding_pct"] = 0.0
            except Exception as exc:  # noqa: BLE001
                log.debug("enricher: deployer holding %s failed: %s", token.mint, exc)

        # lp_burned: not reliably discoverable for pre-graduation pump.fun curves.
        # Leave as None so require_lp_burned:true → filter rejects with lp_unknown
        # rather than silently passing. Callers may set it via a custom rpc mock.

        if not updates:
            return token
        return token.model_copy(update=updates)
