# Memecoin strategy improvements (`improve-memecoin-exits`)

Push the crypto book toward top-tier automatic trading. Stocks path untouched
except shared helpers that crypto-only callers use. **No commits on this branch
from this pass** — improvisation only.

## Already in place (kept)

### A) Mechanical take-profit / stop-loss
- On every crypto open (`TradingDesk._open_crypto`), desk sets `stop_price` /
  `take_profit_price` from `exits.stop_loss_pct` (0.08) and
  `exits.take_profit_pct` (0.20).
- `check_crypto_stops` / `check_crypto_exits` return `stop_loss`, `take_profit`,
  `trailing_stop`, `time_stop`, or `None`. Prefer full CLOSE at TP (no trim).
- `crypto_stops_loop` polls every `exits.mechanical_poll_seconds` (30s).

### B) Anti-chase via `price_change_pct`
- Stage-two filter rejects watch-window pumps above
  `crypto_filter.max_window_pump_pct` (0.80) or dumps below
  `min_window_pump_pct` (-0.15). Missing keys = no opinion.

### C) Hard-veto bundled / sniper / insider
- `hard_veto` rejects `bundled_launch`, `sniper_pct > max_sniper_pct` (0.25),
  `insider_pct > max_insider_pct` (0.15). Config under `crypto_vetoes`.

---

## New this pass

### 1) On-chain enricher (`src/crypto/enricher.py`)
- Runs after the watch window and **before** stage-two `filter_reason`.
- Fills (best-effort via Solana RPC / solders): `mint_revoked`,
  `top10_holder_pct`, `dev_holding_pct`, `holders` + `holders_known`.
- `lp_burned` left `None` when not discoverable (pre-grad pump curves).
- **Unknown must not silently pass hard requires:**
  - `require_mint_revoked` + `mint_revoked is None` → `mint_unknown`
  - `require_lp_burned` + `lp_burned is None` → `lp_unknown`
- RPC client injectable (`OnChainEnricher(config, rpc=…)`) for unit tests.
- Wired: `Scout(enricher=…)` → `await mature()` calls `enricher.enrich`.

### 2) Price poller (`src/crypto/price_poller.py`)
- `CryptoPricePoller` refreshes marks for open crypto positions so mechanical
  stops work without a live scout watch.
- Sources: injected `fetch` (tests) → optional `exits.price_url_template` →
  best-effort RPC bonding-curve decode from `meta.bonding_curve_key`.
- `TradingDesk.poll_crypto_marks()` + `crypto_stops_loop` call it each tick.
- RPC/HTTP failure → log and HOLD (empty price map).

### 3) Faster / smarter crypto exits
- `exits.crypto_interval_minutes` (20) → dedicated `crypto_exit_loop` (LLM).
- Stocks stay on `exits.interval_hours` (4) via `exit_loop`.
- Event-driven LLM: `exits.crypto_move_trigger_pct` (0.15) inside stops loop.
- Trailing stop: arm after `trailing_stop_activate_pct` (0.25), trail
  `trailing_stop_pct` (0.10) off `position.peak_price`.
- Time-stop: `exits.max_hold_minutes` (60) → `time_stop` CLOSE.

### 4) Sizing vs liquidity
- `risk.max_position_pct_of_liquidity` (0.03) caps crypto notional to 3% of
  curve `liquidity_usd`. Wired in `RiskManager.position_size(…, liquidity_usd=)`
  and `TradingDesk._open_crypto`. Stocks ignore the kwarg.

### 5) Holders vs unique_traders
- `Watch.result` no longer sets `holders = unique_traders`.
- `Token.holders_known` is True only after enricher fills a real count.
- Stage-two `min_holders` applies only when `holders_known`; `min_unique_traders`
  always applies from the watch window.

### 6) Checker confidence floor
- Crypto path requires `approve` **and**
  `confidence >= crypto_filter.min_checker_confidence` (default 0.6).
- Below floor → skip reason `checker_low_confidence`.

### 7) Config (`config.example.yaml`)
- Memecoin-first defaults: `crypto_max_pct: 0.85`, `stock_max_pct: 0.30`.
- All new knobs documented under `risk`, `crypto_filter`, and `exits`.

### 8) Tests
- `tests/test_enricher_exits.py` — enricher unknown=reject, poller stop trigger,
  time-stop, trailing stop, liquidity cap, confidence floor, scout enrich wire.
- Existing scout/desk tests updated for async `mature` and mint_unknown semantics.

---

## Strategy brain pass (top-tier playbook)

STRATEGY.md has the full playbook. Every threshold below is configurable in
`config.example.yaml`.

### 9) Entry quality scorecard (`src/crypto/entry_scorecard.py`)
- Five factors on 0..1 with documented linear ramps:
  - **velocity:** buys/min 1→8, buy share 0.50→0.75.
  - **holder_quality:** unique/trades 0.15→0.50, unique 12→60, top-10
    0.45→0.20, dev 0.10→0.02, holders 25→250. Unknown concentration scores a
    neutral 0.5.
  - **liquidity:** $5k→$60k, plus turnover.
  - **social_heat:** virality / community / meme, derivative ×0.75, socials bonus.
  - **audit:** safety minus sniper / insider / red flags.
- Composite ≥ **0.55**, plus the regime add. Per-factor **kill floors**
  (0.20 / 0.25 / 0.15 / 0.25 / 0.40). Size grade A/B/C → 1.0 / 0.75 / 0.5×.
- **Prescreen:** the code-only factors run before any model call, so slow or
  thin tokens cost $0. The scorecard verdict is also passed to the checker.

### 10) Regime gate (`src/crypto/regime.py`)
- **normal:** no change.
- **caution** (go < 0.50 or neutral regime): thresholds +0.05, size ×0.60.
- **defensive** (`sol_trend` down, measured SOL ≤ −4% over 60 min, or risk_off):
  thresholds +0.10, size ×0.40.
- **paused:** go < min_go_signal → `veto_market_paused`; weak + dump →
  `regime_paused`.
- The pulse is read first, so a paused regime skips the auditor and narrative
  calls. Pending staged adds expire while paused.

### 11) Staged entry (`src/crypto/staged_entry.py`)
- 40% probe now. The 60% add fires only at **+8%..+30%** over the probe
  **within 10 min**, with flow confirmation (≥ 8 buys, buy/sell ≥ 1.3,
  ≥ 5 buyers), no dev sell, and no rung hit. No tape means no add.
- `TradingDesk.check_staged_adds` (stops loop) fills the add through the
  executor/simulator, then averages entry, re-derives the stop and runner cap,
  and resets the ladder base. Logged as `action: ADD` / `ADD_EXPIRED`.

### 12) Exit ladder + dump detection (`exit_ladder.py`, `dump_detector.py`)
- Order: emergency dump → hard/breakeven stop → catastrophic crash (35%) →
  runner cap (+150%) → rungs **+15% sell 33%**, **+30% sell 33%** (of base) →
  runner trail 12% → stale exit (20 min without +5%) → time-stop 60 min.
- Dump = **flow flip** (sells ≥ 2× buys with ≥ 6 sells, or sell SOL ≥ 2× buy
  SOL) **and** **crash** (≥ 15% off the 90s high or ≥ 25% off peak). A deployer
  sell is an emergency on its own.
- Trims realize PnL per slice (net of simulated slippage and fees) into the risk
  manager. The final `close` record carries total trade PnL plus
  trigger/grade/creator.
- Legacy positions (no `meta["ladder"]`) keep the old TP/trailing plan.
  `exits.ladder.enabled: false` restores legacy desk-wide.
- Default hard stop widened from 8% to **10%** (the probe already pays ~1–2%
  of impact and fees).

### 13) Toxic flow filter (`src/crypto/toxic_flow.py`)
- Scout stage one rejects: serial deployers (> 2 launches / 24h), metadata URI
  reuse, same-deployer name/symbol relaunches, and clones (≥ 3 prior copies in
  180 min). Normalization folds lookalike digits.
- Deployers of tokens closed on `emergency_*` are blacklisted, logged as
  `blacklist` records, and reloaded on restart. The desk re-checks the
  blacklist before any model call.

### 14) Flow tracker (`src/crypto/flow.py`)
- Rolling per-mint tape for open positions: window stats, dev-sell detection,
  latest reserves and price.
- The scout keeps or starts the trade subscription on entry, releases it on
  close, and resubscribes after a reconnect. The tracker's latest price is also
  a mark source.

### 15) Executor, paper-safe (`src/crypto/crypto_executor.py`)
- Curve buy/sell quote math, `TradeIntent` (venue, min-out, priority fee, CU,
  Jito tip, bonding-curve PDA, Anchor `buy`/`sell` instruction data).
- Dry-run signing of the intent (never broadcasts).
- `PaperFillSimulator`: curve impact or fallback slippage, protocol fee,
  network + tip cost. Buys are refused above `slippage_bps`; exits are never
  refused.
- **Three live gates:** `mode: live` + `--i-understand-the-risk` +
  `solana.live_broadcast: true`. Even when armed, `_broadcast` raises
  `NotImplementedError`. The desk logs `executor_not_implemented` and opens
  nothing.
- Paper mode (not only `--dry-run`) now trades the paper book with simulated
  fills. No quote price → `no_quote_price`, no position.
- Secret key kept out of repr, `executor.config` (scrubbed copy), intents,
  fills and logs.

### 16) Enricher fix
- The bonding curve's own token account (curve PDA + SPL and Token-2022 ATAs)
  is excluded from top-10 and holder counts. Before this fix, every
  pre-graduation token measured ~80% top-10 concentration and would have
  failed `max_top10_holder_pct`. Holder exactness is still judged on the raw
  RPC count (< 20).

### 17) Scripts
- `replay.py` / `dashboard.py`: "deployed" includes staged `ADD` fills.

### 18) Tests
- `tests/test_memecoin_strategy.py` (57 tests): scorecard, prescreen, regime
  states + desk wiring, flow tracker, dump detection, exit ladder (rungs, gaps,
  breakeven, runner trail, cap, stale, time, legacy), staged entry (unit + desk
  end-to-end add/expire), trim/close PnL accounting, dump → blacklist →
  restart, toxic rules + scout wiring, curve quote invariant, paper fills,
  executor gates, dry-run signature verification, key hygiene, enricher curve
  exclusion.
- Existing fixtures: `TOKEN` now carries curve reserves (a real token always
  does). The TP/SL desk test asserts the new paper-fill + runner-cap
  semantics, and the liquidity-cap test disables the scorecard to isolate the
  cap.
- **Full suite: 322 passed.**

---

## Remaining gaps
- **Live execution not wired:** transaction assembly against the current
  pump.fun / PumpSwap IDL, `getTipAccounts` + dynamic Jito tip, `sendBundle`
  plus confirmation with a deadline, post-balance fill reconciliation, wallet
  SPL scan. See STRATEGY.md §8.
- **SOL price feed:** `sol_price_usd` is static. `RegimeGate.observe_sol_price`
  is ready but nothing feeds it yet (the regime uses the pulse's `sol_trend`).
- **Graduation:** the curve `complete` flag is not yet read, so venue choice and
  marks assume the curve.
- **Holder count** exact only when < 20 largest accounts (needs DAS/Helius).
  `lp_burned` not detectable pre-graduation.
- **Tuning:** the thresholds are reasoned defaults, not fitted. They need
  ≥ 100 paper trades with the real feed before anyone trusts them.
