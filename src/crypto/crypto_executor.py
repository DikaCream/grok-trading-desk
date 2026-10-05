"""Solana memecoin execution — paper-safe by construction.

What is real here:
  * pump.fun bonding-curve quote math (constant product on virtual reserves,
    protocol fee taken on the SOL leg)
  * trade intents: venue choice (bonding curve vs graduated AMM), min-out from
    slippage, compute-budget priority fee, Jito tip, Anchor instruction data
    for curve buy/sell, bonding-curve PDA derivation
  * a dry-run signing path: the wallet key is loaded lazily, used to sign the
    intent's canonical bytes (proves the key + signer work), and the signature
    is returned — nothing is ever sent
  * a paper fill simulator that prices fills at the quote (curve impact when
    reserves are known, configured slippage otherwise) plus fees and network cost

What is deliberately NOT here: transaction assembly and broadcast. Broadcasting
requires ALL of
    config mode: "live"   AND   --i-understand-the-risk   AND   solana.live_broadcast: true
and even then `_broadcast` raises NotImplementedError until the owner wires the
account list against the current pump.fun / PumpSwap IDL and the Jito bundle
submit (see STRATEGY.md → "Mainnet checklist"). Anything short of all three
flags runs as paper.

Units: like the rest of the desk, crypto prices are SOL per token straight off
the curve, and `quantity` is in desk units (amount_usd / fill_price) so that
Position.pnl_usd = (mark - entry) * quantity is a USD number. `tokens` in a fill
is the real token estimate for reference.

Keys: the secret never appears in logs, reprs, intents, fills or `self.config`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import struct
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import Market, Position

log = logging.getLogger(__name__)

LAMPORTS_PER_SOL = 1_000_000_000
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
BASE_FEE_LAMPORTS = 5_000
REDACTED = "***redacted***"


class LiveTradingDisabled(RuntimeError):
    """Raised when something tries to broadcast without every live gate open."""


# --- quote math -----------------------------------------------------------------------

def curve_buy_quote(sol_in: float, v_sol: float, v_tok: float, fee_bps: float = 100) -> dict[str, float]:
    """Spend `sol_in` SOL (gross, fee included) on a constant-product virtual curve."""
    if sol_in <= 0 or v_sol <= 0 or v_tok <= 0:
        return {"tokens_out": 0.0, "avg_price": 0.0, "spot_price": 0.0, "impact_pct": 0.0, "fee_sol": 0.0}
    fee = sol_in * float(fee_bps) / 10_000.0
    net = sol_in - fee
    tokens_out = v_tok - (v_sol * v_tok) / (v_sol + net)
    spot = v_sol / v_tok
    avg = sol_in / tokens_out if tokens_out > 0 else 0.0
    return {
        "tokens_out": tokens_out,
        "avg_price": avg,
        "spot_price": spot,
        "impact_pct": (avg - spot) / spot if spot > 0 else 0.0,
        "fee_sol": fee,
    }


def curve_sell_quote(tokens_in: float, v_sol: float, v_tok: float, fee_bps: float = 100) -> dict[str, float]:
    """Sell `tokens_in` into the curve; fee taken from the SOL out."""
    if tokens_in <= 0 or v_sol <= 0 or v_tok <= 0:
        return {"sol_out": 0.0, "avg_price": 0.0, "spot_price": 0.0, "impact_pct": 0.0, "fee_sol": 0.0}
    gross = v_sol - (v_sol * v_tok) / (v_tok + tokens_in)
    fee = gross * float(fee_bps) / 10_000.0
    net = gross - fee
    spot = v_sol / v_tok
    avg = net / tokens_in
    return {
        "sol_out": net,
        "avg_price": avg,
        "spot_price": spot,
        "impact_pct": (spot - avg) / spot if spot > 0 else 0.0,
        "fee_sol": fee,
    }


# --- instruction plumbing ----------------------------------------------------------------

def anchor_discriminator(name: str) -> bytes:
    return hashlib.sha256(f"global:{name}".encode()).digest()[:8]


BUY_DISCRIMINATOR = anchor_discriminator("buy")    # 66063d1201daebea
SELL_DISCRIMINATOR = anchor_discriminator("sell")  # 33e685a4017f83ad


def encode_curve_buy(token_amount_raw: int, max_sol_cost_lamports: int) -> bytes:
    """pump.fun `buy(amount: u64, max_sol_cost: u64)` instruction data.

    Verify against the current IDL before live use: pump.fun has added
    accounts (creator vault, volume accumulators) over time; the args are stable.
    """
    return BUY_DISCRIMINATOR + struct.pack("<QQ", int(token_amount_raw), int(max_sol_cost_lamports))


def encode_curve_sell(token_amount_raw: int, min_sol_output_lamports: int) -> bytes:
    """pump.fun `sell(amount: u64, min_sol_output: u64)` instruction data."""
    return SELL_DISCRIMINATOR + struct.pack("<QQ", int(token_amount_raw), int(min_sol_output_lamports))


def derive_bonding_curve(mint: str, program_id: str = PUMP_PROGRAM_ID) -> str | None:
    """Bonding-curve PDA: seeds [b"bonding-curve", mint]. None if mint is not base58."""
    try:
        from solders.pubkey import Pubkey

        pda, _bump = Pubkey.find_program_address(
            [b"bonding-curve", bytes(Pubkey.from_string(mint))], Pubkey.from_string(program_id)
        )
        return str(pda)
    except Exception:  # noqa: BLE001 - tests use fake mints like "M1"
        return None


def choose_venue(curve_complete: bool | None) -> str:
    """Bonded (complete) curves trade on the AMM (PumpSwap since 2025, Raydium
    for older migrations, both reachable through Jupiter); otherwise the curve."""
    return "amm" if curve_complete else "pump_curve"


@dataclass
class TradeIntent:
    side: str                       # buy | sell
    mint: str
    venue: str                      # pump_curve | amm
    amount_usd: float = 0.0
    sol_in: float = 0.0             # buy: SOL spent (gross)
    tokens_in: float = 0.0          # sell: tokens sold
    fraction: float = 0.0           # sell: share of holding
    expected_price: float = 0.0     # SOL per token, from the quote
    expected_out: float = 0.0       # tokens (buy) or SOL (sell)
    min_out: float = 0.0            # after slippage tolerance
    slippage_bps: int = 500
    priority_fee_microlamports: int = 0
    compute_unit_limit: int = 120_000
    use_jito: bool = False
    jito_tip_lamports: int = 0
    bonding_curve: str = ""
    instruction_data_hex: str = ""
    notes: list[str] = field(default_factory=list)
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, default=str).encode()


# --- key handling ---------------------------------------------------------------------------

class _SecretKey:
    """Holds the base58 secret. Never printable; keypair built lazily on demand."""

    __slots__ = ("_secret", "_keypair")

    def __init__(self, secret: str | None):
        value = str(secret or "").strip()
        self._secret = "" if (not value or "REPLACE" in value.upper()) else value
        self._keypair = None

    def __repr__(self) -> str:
        return f"<SecretKey {'set' if self._secret else 'unset'} {REDACTED}>"

    __str__ = __repr__

    @property
    def present(self) -> bool:
        return bool(self._secret)

    def keypair(self):
        if self._keypair is None and self._secret:
            from solders.keypair import Keypair

            try:
                self._keypair = Keypair.from_base58_string(self._secret)
            except Exception:  # noqa: BLE001
                # Never include the secret (or the raw exception, which may echo it).
                raise ValueError("wallet_key could not be parsed as a base58 keypair") from None
        return self._keypair


def scrub_config(config: dict[str, Any]) -> dict[str, Any]:
    """Deep copy with secrets replaced, safe to keep on an object or log."""
    clean = copy.deepcopy(config or {})
    sol = clean.get("solana")
    if isinstance(sol, dict) and "wallet_key" in sol:
        sol["wallet_key"] = REDACTED
    return clean


# --- paper fills ------------------------------------------------------------------------------

class PaperFillSimulator:
    """Fill at the quote price, honestly: curve impact, protocol fee, network cost."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        sol = config.get("solana", {}) or {}
        paper = config.get("paper", {}) or {}
        jito = sol.get("jito", {}) or {}
        self.fee_bps = float(paper.get("pump_fee_bps", 100))
        self.default_slippage_bps = float(paper.get("default_slippage_bps", 150))
        self.slippage_bps = int(sol.get("slippage_bps", 500))
        self.compute_units = int(paper.get("compute_unit_limit", 120_000))
        self.priority_fee = int(sol.get("priority_fee_microlamports", 0))
        self.use_jito = bool(jito.get("enabled", False))
        self.tip_lamports = int(jito.get("tip_lamports", 0)) if self.use_jito else 0
        self.sol_usd = float((config.get("pump_fun", {}) or {}).get("sol_price_usd", 0) or 0)

    def network_cost_sol(self) -> float:
        priority = self.priority_fee * self.compute_units / 1_000_000  # lamports
        return (BASE_FEE_LAMPORTS + priority + self.tip_lamports) / LAMPORTS_PER_SOL

    def _reserves_ok(self, reserves: tuple[float, float] | None, sol_usd: float) -> bool:
        return bool(reserves and reserves[0] > 0 and reserves[1] > 0 and sol_usd > 0)

    def buy(
        self, mint: str, amount_usd: float, price: float,
        sol_usd: float | None = None, reserves: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        sol_usd = float(sol_usd or self.sol_usd or 0)
        fees_usd = self.network_cost_sol() * sol_usd if sol_usd > 0 else 0.0
        base = {"mint": mint, "side": "buy", "simulated": True, "amount_usd": round(amount_usd, 6),
                "fees_usd": round(fees_usd, 6), "tx_id": f"PAPER-{uuid.uuid4().hex[:12]}"}
        if amount_usd <= 0 or price <= 0:
            # No quote → no fill. A paper book must not hold positions it cannot price.
            return {**base, "filled": False, "price": 0.0, "quantity": 0.0, "tokens": 0.0,
                    "impact_pct": 0.0, "reason": "no_quote_price" if price <= 0 else "zero_amount"}
        if self._reserves_ok(reserves, sol_usd):
            q = curve_buy_quote(amount_usd / sol_usd, reserves[0], reserves[1], self.fee_bps)
            fill_price, tokens, impact = q["avg_price"], q["tokens_out"], q["impact_pct"]
            # anchor to the caller's mark if reserves are stale vs price
            if q["spot_price"] > 0 and abs(q["spot_price"] - price) / price > 0.02:
                fill_price = price * (1 + impact)
        else:
            impact = self.default_slippage_bps / 10_000.0
            fill_price = price * (1 + impact) * (1 + self.fee_bps / 10_000.0)
            tokens = (amount_usd / sol_usd) / fill_price if sol_usd > 0 else 0.0
        if impact * 10_000 > self.slippage_bps:
            return {**base, "filled": False, "price": fill_price, "quantity": 0.0, "tokens": 0.0,
                    "impact_pct": round(impact, 6), "reason": "slippage_exceeded"}
        return {**base, "filled": True, "price": fill_price, "quantity": amount_usd / fill_price,
                "tokens": tokens, "impact_pct": round(impact, 6), "reason": "ok"}

    def sell(
        self, mint: str, quantity: float, price: float,
        sol_usd: float | None = None, reserves: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """`quantity` in desk units. Proceeds in USD; price is the effective fill."""
        sol_usd = float(sol_usd or self.sol_usd or 0)
        fees_usd = self.network_cost_sol() * sol_usd if sol_usd > 0 else 0.0
        base = {"mint": mint, "side": "sell", "simulated": True, "quantity": quantity,
                "fees_usd": round(fees_usd, 6), "tx_id": f"PAPER-{uuid.uuid4().hex[:12]}"}
        if quantity <= 0 or price <= 0:
            return {**base, "filled": quantity <= 0, "price": float(price or 0), "proceeds_usd": 0.0,
                    "impact_pct": 0.0, "reason": "nothing_to_sell" if quantity <= 0 else "no_quote_price"}
        value_usd = quantity * price
        if self._reserves_ok(reserves, sol_usd):
            tokens = (value_usd / sol_usd) / price
            q = curve_sell_quote(tokens, reserves[0], reserves[1], self.fee_bps)
            impact = q["impact_pct"]
        else:
            impact = self.default_slippage_bps / 10_000.0 + self.fee_bps / 10_000.0
        impact = max(0.0, min(0.99, impact))
        fill_price = price * (1 - impact)
        # Exits are never refused in paper on slippage: a stop must get out.
        return {**base, "filled": True, "price": fill_price, "proceeds_usd": quantity * fill_price,
                "impact_pct": round(impact, 6), "reason": "ok"}


# --- executor ----------------------------------------------------------------------------------

class CryptoExecutor:
    """Same surface the desk already uses: buy / sell / close_position / tighten_stop / get_positions."""

    market = Market.CRYPTO

    def __init__(self, config: dict[str, Any], live_ack: bool = False, simulator: PaperFillSimulator | None = None):
        raw = config or {}
        solana = raw.get("solana", {}) or {}
        self._secret = _SecretKey(solana.get("wallet_key"))
        self.config = scrub_config(raw)
        self.rpc_url = solana.get("rpc_url", "")
        self.jito = solana.get("jito", {}) or {}
        self.slippage_bps = int(solana.get("slippage_bps", 500))
        self.priority_fee = int(solana.get("priority_fee_microlamports", 0))
        self.compute_unit_limit = int((raw.get("paper", {}) or {}).get("compute_unit_limit", 120_000))
        self.sol_usd = float((raw.get("pump_fun", {}) or {}).get("sol_price_usd", 0) or 0)

        wants_live = str(raw.get("mode", "paper")).lower() == "live"
        broadcast_flag = bool(solana.get("live_broadcast", False))
        #: every gate must be open; anything less is paper
        self.live_armed = bool(wants_live and live_ack and broadcast_flag)
        self.paper = not self.live_armed
        if wants_live and not self.live_armed:
            log.warning(
                "crypto executor: live requested but not armed (needs mode=live, "
                "--i-understand-the-risk and solana.live_broadcast=true); staying on paper"
            )
        self.simulator = simulator or PaperFillSimulator(raw)
        self.paper_book: dict[str, float] = {}

    def __repr__(self) -> str:
        return f"<CryptoExecutor paper={self.paper} live_armed={self.live_armed} key={self._secret!r}>"

    # -- intents ------------------------------------------------------------------------------

    def _base_intent(self, side: str, mint: str, curve_complete: bool | None, bonding_curve: str) -> TradeIntent:
        venue = choose_venue(curve_complete)
        notes = []
        if venue == "pump_curve":
            notes.append("pump.fun curve: accounts per current IDL (global, fee_recipient, mint, "
                         "bonding_curve, associated_bonding_curve, user ATA, creator_vault, ...)")
        else:
            notes.append("graduated: route via PumpSwap/Raydium (Jupiter quote+swap API)")
        if self.jito.get("enabled"):
            notes.append("append SystemProgram.transfer(tip) to a getTipAccounts() address; "
                         "submit via block engine sendBundle")
        return TradeIntent(
            side=side, mint=mint, venue=venue, slippage_bps=self.slippage_bps,
            priority_fee_microlamports=self.priority_fee, compute_unit_limit=self.compute_unit_limit,
            use_jito=bool(self.jito.get("enabled")), jito_tip_lamports=int(self.jito.get("tip_lamports", 0) or 0),
            bonding_curve=(bonding_curve or derive_bonding_curve(mint) or "") if venue == "pump_curve" else "",
            notes=notes,
        )

    def build_buy_intent(
        self, mint: str, amount_usd: float, *, price: float = 0.0, sol_usd: float | None = None,
        reserves: tuple[float, float] | None = None, curve_complete: bool | None = None,
        bonding_curve: str = "", token_decimals: int = 6,
    ) -> TradeIntent:
        sol_usd = float(sol_usd or self.sol_usd or 0)
        intent = self._base_intent("buy", mint, curve_complete, bonding_curve)
        intent.amount_usd = float(amount_usd)
        intent.sol_in = amount_usd / sol_usd if sol_usd > 0 else 0.0
        if reserves and reserves[0] > 0 and reserves[1] > 0 and intent.sol_in > 0:
            q = curve_buy_quote(intent.sol_in, reserves[0], reserves[1], self.simulator.fee_bps)
            intent.expected_price, intent.expected_out = q["avg_price"], q["tokens_out"]
        elif price > 0 and intent.sol_in > 0:
            intent.expected_price, intent.expected_out = price, intent.sol_in / price
        intent.min_out = intent.expected_out * (1 - self.slippage_bps / 10_000.0)
        if intent.venue == "pump_curve" and intent.expected_out > 0:
            max_cost = int(intent.sol_in * (1 + self.slippage_bps / 10_000.0) * LAMPORTS_PER_SOL)
            intent.instruction_data_hex = encode_curve_buy(
                int(intent.min_out * 10**token_decimals), max_cost
            ).hex()
        return intent

    def build_sell_intent(
        self, mint: str, fraction: float, *, tokens: float = 0.0, price: float = 0.0,
        reserves: tuple[float, float] | None = None, curve_complete: bool | None = None,
        bonding_curve: str = "", token_decimals: int = 6,
    ) -> TradeIntent:
        intent = self._base_intent("sell", mint, curve_complete, bonding_curve)
        intent.fraction = max(0.0, min(1.0, float(fraction)))
        intent.tokens_in = float(tokens) * intent.fraction
        if reserves and reserves[0] > 0 and reserves[1] > 0 and intent.tokens_in > 0:
            q = curve_sell_quote(intent.tokens_in, reserves[0], reserves[1], self.simulator.fee_bps)
            intent.expected_price, intent.expected_out = q["avg_price"], q["sol_out"]
        elif price > 0:
            intent.expected_price, intent.expected_out = price, intent.tokens_in * price
        intent.min_out = intent.expected_out * (1 - self.slippage_bps / 10_000.0)
        if intent.venue == "pump_curve" and intent.tokens_in > 0:
            intent.instruction_data_hex = encode_curve_sell(
                int(intent.tokens_in * 10**token_decimals), int(intent.min_out * LAMPORTS_PER_SOL)
            ).hex()
        return intent

    # -- dry-run signing -----------------------------------------------------------------------

    def sign_intent(self, intent: TradeIntent) -> dict[str, Any]:
        """Sign the intent's canonical bytes with the wallet. NEVER broadcasts."""
        if not self._secret.present:
            return {"signed": False, "reason": "no_wallet_key", "broadcast": False}
        try:
            keypair = self._secret.keypair()
            signature = keypair.sign_message(intent.canonical_bytes())
            return {"signed": True, "signature": str(signature), "pubkey": str(keypair.pubkey()),
                    "intent_id": intent.intent_id, "broadcast": False}
        except Exception as exc:  # noqa: BLE001
            return {"signed": False, "reason": type(exc).__name__, "broadcast": False}

    async def _broadcast(self, intent: TradeIntent, signed: dict[str, Any]) -> dict[str, Any]:
        if not self.live_armed:
            raise LiveTradingDisabled("broadcast refused: live trading is not armed")
        raise NotImplementedError(
            "live broadcast is not wired: assemble the v0 transaction (compute budget + "
            "swap + Jito tip), sign it, sendBundle, and confirm with a deadline — see "
            "STRATEGY.md mainnet checklist"
        )

    # -- public surface ---------------------------------------------------------------------------

    async def buy(self, mint: str, amount_usd: float, **kwargs: Any) -> dict[str, Any]:
        """Buy `amount_usd` of `mint`. Paper: simulated fill at the quote price.

        kwargs: price, sol_usd, reserves=(v_sol, v_tokens), curve_complete, bonding_curve.
        """
        price = float(kwargs.get("price", 0) or 0)
        reserves = kwargs.get("reserves")
        sol_usd = kwargs.get("sol_usd")
        intent = self.build_buy_intent(
            mint, amount_usd, price=price, sol_usd=sol_usd, reserves=reserves,
            curve_complete=kwargs.get("curve_complete"), bonding_curve=kwargs.get("bonding_curve", ""),
        )
        if self.paper:
            fill = self.simulator.buy(mint, amount_usd, price, sol_usd=sol_usd, reserves=reserves)
            fill.update({"mode": "paper", "intent": intent.to_dict()})
            if fill.get("filled") and fill.get("quantity"):
                self.paper_book[mint] = self.paper_book.get(mint, 0.0) + float(fill["quantity"])
            return fill
        signed = self.sign_intent(intent)
        return await self._broadcast(intent, signed)

    async def sell(self, mint: str, fraction: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Sell `fraction` of the holding. kwargs: price, quantity (desk units), reserves, sol_usd."""
        fraction = max(0.0, min(1.0, float(fraction)))
        price = float(kwargs.get("price", 0) or 0)
        held = float(kwargs.get("quantity") if kwargs.get("quantity") is not None else self.paper_book.get(mint, 0.0))
        qty = held * fraction
        sol_usd = float(kwargs.get("sol_usd") or self.sol_usd or 0)
        tokens = (held * price / sol_usd) / price if (price > 0 and sol_usd > 0) else 0.0
        intent = self.build_sell_intent(
            mint, fraction, tokens=tokens, price=price, reserves=kwargs.get("reserves"),
            curve_complete=kwargs.get("curve_complete"), bonding_curve=kwargs.get("bonding_curve", ""),
        )
        if self.paper:
            fill = self.simulator.sell(mint, qty, price, sol_usd=sol_usd, reserves=kwargs.get("reserves"))
            fill.update({"mode": "paper", "fraction": fraction, "intent": intent.to_dict()})
            if mint in self.paper_book:
                self.paper_book[mint] = max(0.0, self.paper_book[mint] - qty)
                if fraction >= 1.0 or self.paper_book[mint] <= 0:
                    self.paper_book.pop(mint, None)
            return fill
        signed = self.sign_intent(intent)
        return await self._broadcast(intent, signed)

    async def close_position(self, mint: str, **kwargs: Any) -> dict[str, Any]:
        return await self.sell(mint, 1.0, **kwargs)

    async def tighten_stop(self, mint: str, new_stop_price: float) -> dict[str, Any]:
        """No on-chain stop exists on pump.fun; the desk's mechanical loop is the stop."""
        return {"mint": mint, "stop_price": float(new_stop_price), "desk_side": True}

    async def get_positions(self) -> list[Position]:
        """Paper: the desk owns the book. Live: SPL account scan (not wired)."""
        if self.paper:
            return []
        raise NotImplementedError("live wallet position scan is not wired (see STRATEGY.md)")
