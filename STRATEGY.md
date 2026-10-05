# Memecoin playbook

How the crypto book on `grok-trading-desk` picks, sizes, enters, manages and
exits pump.fun memecoins. Each rule below is code, not a prompt. Every
threshold is a config key in `config.example.yaml`, and the defaults listed
here match that file.

> Paper first. Crypto execution simulates fills unless **all three** of these
> are set: `mode: "live"`, the `--i-understand-the-risk` flag, and
> `solana.live_broadcast: true`. Even then, the broadcast step raises
> `NotImplementedError` until the items in the mainnet checklist (end of this
> file) are built.

---

## 0. Pipeline at a glance

```
pump.fun create event
  └─ Stage 1  launch filter (curve size, mcap, dev buy, metadata)       scout.launch_reason
  └─ Stage 1b TOXIC FLOW (serial deployer / URI reuse / clones / blacklist)  toxic_flow.py
  └─ Watch window 300s: buys, sells, unique traders, price change      scout.Watch
  └─ On-chain enricher (mint authority, top-10 without curve, dev %)    enricher.py
  └─ Stage 2  watch filter + anti-chase + hard requires                 scout.filter_reason
desk.evaluate_token
  └─ toxic blacklist + SCORECARD PRESCREEN (code-only factors)          $0, no model calls
  └─ REGIME GATE (crypto_pulse + SOL trend)                             regime.py
  └─ auditor + narrative (LLM, live search)
  └─ weighted matrix + hard vetoes (bundled / sniper / insider / wash)  crypto_scoring.py
  └─ regime-tightened matrix threshold
  └─ ENTRY SCORECARD (5 factors, kill floors, grade)                    entry_scorecard.py
  └─ adversarial checker (approve AND confidence >= 0.6)
  └─ size = risk size × regime mult × grade mult, capped at 3% of liquidity
  └─ STAGED ENTRY: probe 40% now                                        staged_entry.py
crypto_stops_loop (every 30s, plus the live trade tape)
  └─ STAGED ADD: +60% if +8%..+30% with volume within 10 min
  └─ EXIT LADDER: dump → stop → crash → cap → rungs → trail → stale → time   exit_ladder.py
crypto_exit_loop (every 20 min) + event-driven LLM on ±15% moves         exit_manager
```

---

## 1. Toxic flow filter (`src/crypto/toxic_flow.py`)

Most pump.fun losses come from the same few operators running the same play.
The scout sees every create event, including the ones it filters out, so it
keeps a memory of who launched what and how each launch was dressed:

| Reject reason | Rule | Default |
|---|---|---|
| `toxic_deployer_blacklisted` | creator is on the blacklist | auto-added on `emergency_dev_dump`, `emergency_dump`, `emergency_crash` closes; persisted as `blacklist` log records and reloaded at startup |
| `toxic_serial_deployer` | creator launched more than N mints in the window | N = 2 per 1440 min |
| `toxic_uri_reuse` | identical metadata URI already used by a different mint | on |
| `toxic_clone_same_deployer` | same creator relaunches the same normalized name+symbol (relaunch after a rug) | on |
| `toxic_metadata_clone` | at least `max_clones` other mints already used this name+symbol in the window | 3 in 180 min |

Normalization lowercases the text, strips everything except a–z and 0–9, and
folds lookalike digits (0→o, 1→i, 3→e, 4→a, 5→s, 7→t), so `D0GE-W1F` matches
`dogewif`. Memory is bounded and pruned by time.

The scout applies all of these rules at stage one. The desk re-checks the
blacklist before any model call, so a deployer blacklisted *after* its token
entered the watch window still gets rejected.

---

## 2. Entry quality scorecard (`src/crypto/entry_scorecard.py`)

The weighted matrix answers "do the bots like it?". The scorecard answers "is
the tape good enough to pay for?". It scores five factors from 0 to 1. Each
factor is a linear ramp: at or below `lo` it scores 0, at or above `hi` it
scores 1.

| Factor | Weight | Inputs and ramps | Kill floor |
|---|---|---|---|
| **velocity** | 0.25 | 0.7 × buys/min (1 → 8) + 0.3 × buy share of trades (0.50 → 0.75) | 0.20 |
| **holder_quality** | 0.20 | unique-trader ratio = unique / trades (0.15 → 0.50; low means a few wallets churning, i.e. wash or bots), unique traders (12 → 60), top-10 concentration (0.45 bad → 0.20 good), deployer holding (0.10 bad → 0.02 good), real holder count (25 → 250) when the enricher knows it. **Unknown concentration scores a neutral 0.5, never 1.0.** | 0.25 |
| **liquidity** | 0.15 | curve liquidity USD (5k → 60k); with window volume: 0.75 × depth + 0.25 × turnover (volume SOL / curve SOL, 0.3 → 2.0) | 0.15 |
| **social_heat** | 0.20 | narrative: 0.4 virality + 0.3 community + 0.3 meme; × 0.75 if derivative; +0.03 per social link (max +0.09) | 0.25 |
| **audit** | 0.20 | safety − 0.5 sniper% − 0.5 insider% − 0.03 per red flag − 0.3 if bundled | 0.40 |

* **Composite** must be at least `min_entry_score` = **0.55**, plus the regime
  add (see §3).
* **Kill floors:** a single factor below its floor vetoes the entry, however
  good the rest look. A great meme on a dead tape is still a dead tape.
* **Prescreen:** velocity, holder_quality and liquidity need no model, so their
  kill floors run **before** any LLM call. Slow or thin tokens cost $0.
* **Grade → size:** A (≥ 0.75) gets 1.0× planned size, B (≥ 0.65) gets 0.75×,
  C (passes but below B) gets 0.5×.

Why these numbers:

* **8 buys/min** is where a fresh pump.fun curve shows real crowd demand rather
  than the deployer plus a few snipers.
* **Under 15% unique-trader ratio** means the same handful of wallets is
  cycling volume.
* **Top-10 above 45%** is the existing stage-two cap, so the ramp ends there.
* **$5k liquidity** matches stage two's minimum. By $60k the curve absorbs a
  3%-of-liquidity position with little impact.

---

## 3. Regime gate (`src/crypto/regime.py`)

Memecoins are high-beta SOL. The same setup that prints in a hot tape bleeds
out in a cold one. The gate only governs **new** entries; exits are never
gated.

| State | Condition | Effect |
|---|---|---|
| normal | go ≥ 0.50, SOL not dumping | none |
| caution | 0.30 ≤ go < 0.50, or regime `neutral` | matrix and scorecard thresholds +0.05; size × 0.60 |
| defensive | `sol_trend == "down"` or measured SOL move ≤ −4% over 60 min, or regime `risk_off` with go ≥ 0.30 | thresholds +0.10; size × 0.40 |
| paused | go < `pulse.min_go_signal` (0.30) → `veto_market_paused`; weak **and** dumping → `regime_paused` | no new entries; pending staged adds expire |

The pulse is cached for 15 min and is read first, so a paused regime skips the
auditor and narrative calls. `RegimeGate.observe_sol_price()` accepts a
measured SOL feed when one is wired; until then the gate uses the pulse's
`sol_trend` and an optional `sol_change_pct`.

---

## 4. Sizing

```
planned = min(15% of crypto budget, 25% of remaining daily loss room, free budget)
          × (0.5 + 0.5 × checker adjusted_score)
          capped at 3% of curve liquidity USD
          × regime.size_mult × scorecard grade mult
```

---

## 5. Staged entry (`src/crypto/staged_entry.py`)

Paying full size before the tape confirms is how a desk ends up funding the
snipers it was trying to avoid.

* **Probe:** 40% of planned size, filled immediately.
* **Add:** the remaining 60%, filled only if **all** of these hold within
  **Y = 10 min** of the probe:
  * price is at least **X = +8%** above the probe fill;
  * price is at most **+30%** above it (chase guard; a pullback into the band
    can still trigger the add);
  * post-entry flow shows ≥ 8 buys, buy/sell ≥ 1.3, and ≥ 5 distinct buyers;
  * the deployer has not sold, no ladder rung has fired, the regime is not
    paused, and the daily loss limit is not hit.
* **No flow data means no add** (`require_flow: true`). The add is never placed
  blind.
* If the window lapses, the position becomes `probe_only` and is managed by the
  ladder at probe size.
* After an add, the entry price becomes the USD-weighted average. The hard stop
  and runner cap are recomputed from that average, and the ladder's base
  quantity resets.

Post-entry flow comes from `FlowTracker` (`src/crypto/flow.py`). The scout keeps
the PumpPortal trade subscription alive for every open position, and
resubscribes after a reconnect.

---

## 6. Exit ladder (`src/crypto/exit_ladder.py`)

On every mark, the first matching rule wins:

| # | Rule | Trigger | Action |
|---|---|---|---|
| 1 | **Emergency dump** | deployer sold after entry (`emergency_dev_dump`), **or** flow flip **and** price crash (`emergency_dump`) | CLOSE everything, blacklist deployer |
| 2 | Hard / protected stop | price ≤ stop (−10% from average entry; raised to entry + 1% after rung 1) | CLOSE `stop_loss` / `breakeven_stop` |
| 3 | Catastrophic crash | ≥ 35% below the 90s window high (or below the peak when no tape is available) | CLOSE `emergency_crash`, blacklist deployer |
| 4 | Runner cap | price ≥ entry × 2.5 (+150%) | CLOSE `take_profit` |
| 5 | **Rung 1** | +15% | sell 33% of base qty, stop → breakeven |
| 5 | **Rung 2** | +30% | sell another 33% of base qty |
| 6 | Runner trail | after rung 1, price ≤ peak × (1 − 12%) | CLOSE `runner_trail` |
| 7 | Stale exit | 20 min, peak never reached +5%, no rung hit | CLOSE `stale_exit` |
| 8 | Time-stop | 60 min hold | CLOSE `time_stop` |

* If price gaps through several rungs at once, they sell together.
* If what remains after a trim would be dust (≤ 5% of base), the position is
  closed instead.
* **Dump detection** (`src/crypto/dump_detector.py`, 90s window):
  * *Flow flip:* sells ≥ 2× buys with at least 6 sells, **or** sell SOL ≥ 2×
    buy SOL.
  * *Crash:* ≥ 15% below the window high, **or** ≥ 25% below the position peak.
  * Either signal alone is noise (a big seller into strong bids, or a wick on a
    thin book). Both together mean get out.
* Each trim realizes that slice's PnL (net of simulated slippage and fees) into
  the risk manager and releases its capital.
* The final `close` record carries the **total** trade PnL (all partials + the
  remainder − entry costs), so outcome memory learns from whole trades.
* Positions opened before the ladder existed (no `meta["ladder"]`) keep the
  legacy single-TP / trailing plan. Setting `exits.ladder.enabled: false`
  restores the legacy rules desk-wide.
* LLM TRIM/CLOSE decisions on crypto go through the same trim/close helpers, so
  their accounting is identical.

---

## 7. Execution path (`src/crypto/crypto_executor.py`)

| Piece | Status |
|---|---|
| pump.fun curve quote math (constant product on virtual reserves, fee on the SOL leg) | **done**, unit-tested against the invariant |
| Venue choice: curve vs graduated AMM (PumpSwap, or Raydium for older migrations, via Jupiter) | **done** (needs the curve `complete` flag passed in) |
| `TradeIntent`: amounts, expected out, min-out from `slippage_bps`, priority fee, CU limit, Jito tip, bonding-curve PDA, Anchor instruction data (`buy`/`sell` discriminators + u64 args) | **done** |
| Dry-run signing: lazy keypair, signs the intent's canonical bytes, returns `broadcast: False` | **done** |
| Key hygiene: secret never in repr, `executor.config`, intents, fills, or logs (tested with caplog) | **done** |
| Paper fill simulator: fills at the quote with curve impact (or 150 bps when reserves are unknown), 1% protocol fee, base + priority + Jito tip cost; refuses buys whose impact exceeds `slippage_bps`; never refuses an exit | **done** |
| No quote price → no paper position (`no_quote_price`) | **done** |
| Transaction assembly, broadcast, confirmation, wallet scan | **not wired** (see checklist) |

---

## 8. Mainnet checklist (still needed)

1. **Wallet:** a dedicated hot wallet with only the risk budget on it. Load the
   key from an env var or secret manager rather than the YAML file. Add a
   startup check that the wallet's SOL balance covers budget + fees.
2. **RPC:** a paid, low-latency endpoint (Helius, Triton or QuickNode) with
   WebSocket support, ideally colocated near the Jito block engine. The public
   `api.mainnet-beta` endpoint rate-limits and will drop sends.
3. **Account lists against the current IDLs:** pump.fun `buy`/`sell` (global,
   fee_recipient, mint, bonding_curve, associated_bonding_curve, user ATA,
   system/token programs, creator_vault, event authority, plus the volume
   accumulators added in 2025). Detect Token-2022 mints. For graduated tokens,
   use PumpSwap directly or Jupiter's swap API.
4. **Jito:** fetch tip accounts at runtime via `getTipAccounts`. Never hardcode
   them. Size the tip dynamically from the tip-floor API (the current fixed
   100k lamports is a placeholder), and submit through `sendBundle` with
   `getBundleStatuses` polling.
5. **Confirmation and reconciliation:** confirm against a deadline. A bundle
   that never lands is a failure, not a no-op. Read the actual token delta from
   the transaction's post-balances, not from the quote. Reconcile
   `get_positions` (an SPL account scan) with the desk book on startup and
   every exit pass.
6. **Partial fills and failed sells:** retry exits with escalating slippage and
   tip. Never mark a position closed until the sell is confirmed.
7. **SOL price feed:** refresh `sol_price_usd` at runtime (the scout uses a
   static fallback today) and feed `RegimeGate.observe_sol_price`.
8. **Graduation:** pass the curve's `complete` flag to the executor and the
   price poller, so post-migration marks come from the AMM pool.
9. **Holder data:** use a DAS/indexer (e.g. Helius) for exact holder counts
   above 20 accounts. `lp_burned` is not discoverable pre-graduation.
10. **Burn-in:** run at least 2 weeks of paper with the real feed, compare paper
    fills against the on-chain quotes logged in `entry.quote_price`, then go
    live at minimum size.

---

## 9. Tuning loop

* `scripts/replay.py` shows skip reasons (`scorecard_weak_*`, `regime_*`,
  `toxic_*`, `add_window_expired`), actions (`TRIM`, `ADD`, `ADD_EXPIRED`) and
  per-market PnL net of model spend. Deployed capital includes staged adds.
* `close` records carry `trigger`, `grade` and `creator`. Grouping PnL by
  trigger shows whether the stop is too tight (many `stop_loss` closes followed
  by recoveries) or the rungs are too greedy (few `tp_rung_*` hits).
* Change one knob at a time, and judge only on paper runs of at least 100
  trades.
