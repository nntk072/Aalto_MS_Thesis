# Swing Detection & Data Pipeline Fix - Implementation Plan v4

Status: **implemented**. NY-only `TradingEnv` steps over a full-day dataframe;
HTF features shifted to completed bars before ffill.

The "Current Code (v4 audit)" section below is the **pre-implementation**
snapshot. When this spec and the source disagree, the source wins.

Supersedes v3. Design decisions locked: NY-only `TradingEnv` steps over a
full-day dataframe; HTF features shifted to completed bars before ffill.

---

## Architectural Invariant (READ FIRST)

**At environment step t, every observation value must be computable using only
market information that was available no later than the close of the bar
represented by step t.**

This single sentence governs ALL causality requirements:

- pivots, swings, HH/HL/LH/LL
- FVG, SMT, MTF, HTF features
- EMA/RSI/ATR indicators
- session levels
- observation windows

---

## Overview

Fixes for five core issues in the pre-implementation tree:

1. **Data Loading**: Full-day M1 for all feature construction (HTF structure ->
   MTF intermediate -> LTF execution). `filter_session()` must never drop bars
   from that dataset.
2. **Buffer Context Window**: At NY open, the 60-bar observation window must
   contain the same day's pre-NY bars (London/Asia), not the previous NY close.
3. **EOD Close**: Agent-managed overnight with risk guardrails + configurable
   `block_overnight`. Env steps NY bars only; skipped bars are replayed for
   SL/TP/MTM when a position is held.
4. **Plotting**: Full data range (Asia open through NY close + buffer). Falls
   out of (1) once eval consumes pipeline/env bars.
5. **Swing High/Low**: Confirmed, volatility-adaptive detector with strict
   causality. Existing `last_swing_high` / `last_swing_low` remain the
   downstream API via a wrapper.

---

## Current Code (v4 audit)

Facts as of the v4 rewrite. When this spec and the source disagree later, the
source wins and this spec must be updated.

### Issue 1 — pipeline still NY-filters, then features resample that M1

[`quant_rl/data/pipeline.py`](../quant_rl/data/pipeline.py) loads M1, then:

```python
m1_session = filter_session(m1_clean, start=cfg.session.start, end=cfg.session.end)
# M1 cache = NY-only
# HTF: resample(m1_clean) then filter_session again on HTF timestamps
```

[`quant_rl/train/train_rl.py`](../quant_rl/train/train_rl.py) feeds
`data[symbol]["M1"]` into `build_features()`. Every HTF block in
[`quant_rl/features/build.py`](../quant_rl/features/build.py) then
`resample(primary, tf)` from that **already NY-filtered** M1. Pipeline HTF
parquet is unused by training; the damage is the M1 spine.

Bar cache filename `{symbol}_{tf}.parquet` has **no version**. After dropping
the session filter, stale NY-only parquet would be reused unless the name
changes or `force=True`.

[`build_tick_books`](../quant_rl/data/pipeline.py) still passes
`session_start` / `session_end` into [`ticks.py`](../quant_rl/data/ticks.py).
Overnight MTM/SL on skipped bars needs ticks (or bar-spread fallback) on
those timestamps.

**Silent production bug:** [`detect_session_levels()`](../quant_rl/features/structure.py)
masks Asia 01:05–09:00 and London 09:00–16:30. On NY-only M1 both masks are
empty, so `asian_*` / `london_*` are all-NaN in production. Unit tests pass
because they construct synthetic full-day frames
([`tests/test_features/test_structure.py`](../tests/test_features/test_structure.py),
[`tests/test_integration/test_session_levels_integration.py`](../tests/test_integration/test_session_levels_integration.py)).

### Issue 2 — observation window sees previous NY, not London

[`TradingEnv._get_observation`](../quant_rl/envs/trading_env.py):

```python
start_idx = max(0, self.step_idx - self.obs_window)
seq = self._obs_features.iloc[start_idx : self.step_idx]
```

With NY-only storage, NY open of day N is adjacent to late NY of day N-1
(≈17.5 h gap). The 60-bar window is **not** zero-padded after the first
episode bars — it is previous-session NY. v3 understated this.

VAE `pre_ny_data` is only the unused narrative path (`use_narrative: false`).
It is not a substitute for a full-day observation window.

### Issue 3 — overnight flag exists; EOD guards do not

`block_overnight` defaults `true` in [`default.yaml`](../quant_rl/config/default.yaml)
and force-closes when the next bar's `session_id` differs. There is no stale
progress counter, frozen entry ATR, or timestamp max-age.

[`Position`](../quant_rl/backtest/broker.py) fields today: `direction`, `size`,
`entry_price`, `margin_used`, `sl_price`, `tp_price`, `risk_frac`, `rr_ratio`.
No `entry_timestamp`, `entry_atr`, MFE/MAE, or stale state.

[`session_id`](../quant_rl/data/session.py) is **calendar date**, not NY
session. With full-day rows it flips at midnight (Asia), not 23:00. EOD must
not be “calendar `session_id` changed” on an all-bar walk.

v3's gap “23:00 → 01:05 Asian” is wrong for **today** (next stored bar is
next-day 16:30). After the pipeline fix the next *stored* bar is 01:05, but
the next *env step* is next-day 16:30 (NY-only steps).

### Issue 4 — plots inherit NY-only bars

Eval overlays/plots do not call `filter_session()` themselves. They consume
pipeline/env frames. Fixing (1) and keeping full-day bars in the env is the
plot fix. Search every `filter_session` call path when implementing.

### Issue 5 — wick rolling swings, duplicated in SMT

[`structure_levels`](../quant_rl/features/structure.py) / [`smt.py`](../quant_rl/features/smt.py)
use causal rolling max/min on high/low with `smt_swing_period: 5` for every
TF. No ATR filter, no HH/HL/LH/LL, no continuous structure features.

Downstream consumers v3 omitted (must keep working via wrapper):

- [`liquidity.py`](../quant_rl/features/liquidity.py) — `structure_levels` shifted by 1
- MTF structure/BOS in `build.py`
- env SL/TP from `last_swing_high` / `last_swing_low`
- [`chart_overlays.py`](../quant_rl/eval/chart_overlays.py)

### HTF alignment lookahead (v3 invariant, currently violated)

[`align_timeframes`](../quant_rl/data/align.py) ffill by HTF **open** time.
[`tests/test_features/test_htf_alignment.py`](../tests/test_features/test_htf_alignment.py)
locks that in: M1 `t` sees the HTF bar whose open ≤ t, including the still-
forming candle's final OHLC.

v4 **changes** that contract: one HTF period shift before ffill so M1 `t`
only sees a completed higher-TF candle.

Chain F already shifts: [`po3_config.py`](../quant_rl/features/po3_config.py)
`htf_fvg.shift(1)` and `primary_ifvg.shift(1)`. After `align_timeframes`
gains a completed-bar shift, those callers must not be double-shifted.

### Session systems (do not add a third)

| Layer | Role | Keep |
|-------|------|------|
| `cfg.session` 16:30–23:00 broker tz (`Etc/GMT-3`) | Tradable NY window | Yes — eligibility |
| `get_session()` / `detect_session_levels()` | Broker-tz Asia/London/NY levels | Yes — wrap, fix `closed` |
| `features.session_ohlc` CT (`America/Chicago`, `*_ct`) | Opt-in DST-aware levels | Yes — additive only |

Today `get_session()` labels 23:50–01:05 as `ny`. v4 adds explicit `closed`.

Option A (broker-time) for trading eligibility. CT remains feature-only.

### D1 bar boundary

[`resample(..., "D1")`](../quant_rl/data/resample.py) uses pandas `1D`,
`label="left"`, `closed="left"` on the broker tz index. D1 is **broker
midnight** (`Etc/GMT-3`), not UTC midnight and not NY close. Document in the
data contract; do not silently change the rule in this work.

---

## Chosen env contract (NY-only steps)

```
Full-day M1 in bars and features
        |
ny_indices = rows where session == "ny"
        |
TradingEnv.step walks ny_indices only
        |
        +--> obs = features[bar_idx - obs_window : bar_idx]
        |    (contiguous full-day M1; NY 16:30 window is London)
        |
        +--> if overnight hold: replay skipped bars for SL/TP/MTM/age
             before the next NY step
```

Rules:

- Dataframe keeps Asia / London / NY / `closed`.
- Agent actions and PPO `max_episode_steps` count **NY steps only** (~390 M1
  bars/day). `obs_window` still indexes the full-day frame.
- `block_overnight=true`: force-close on the last NY bar of the session
  (detected as a jump in `ny_indices` / session label), same user-visible
  behaviour as today.
- `block_overnight=false`: hold across the gap. Replay skipped bars
  (overnight + Asia + London) for SL/TP/MTM/age **before** the next NY step.
  Fill latency on the last NY bar must not silently fill at next-day 16:30.
- FTMO daily reset: on the NY-session jump, **not** calendar midnight.
- VWAP: calendar-day VWAP on full-day bars (behaviour change vs today's
  NY-only VWAP). Call this out in the cache/version notes; do not invent a
  second NY-only VWAP in this work unless a later prompt asks.
- `session_id` stays calendar date for grouping. EOD / daily reset / entry
  suppression use NY step-index jumps (or `session == "ny"` transitions),
  not “`session_id` changed” on skipped midnight bars.

---

## Phase 0: Write Invariants and Tests FIRST

BEFORE any implementation, define these contracts and write tests.

### Critical Invariants

1. **No-lookahead contract**: Any swing/pivot generated from future candles
   must become visible ONLY at the first candle where confirmation is complete.
2. **Session-data contract**: `filter_session()` must NEVER remove bars from
   the feature construction dataset. It is only a trading eligibility mask
   helper (or is replaced by `session == "ny"`).
3. **Observation-window contract**: At the first NY trading bar, the 60-bar
   window must contain the preceding pre-NY bars (London/Asia of that day),
   not the previous day's NY close.
4. **EOD-state contract**: Position state (entry_price, SL, TP, MFE/MAE,
   progress) must survive the NY-session boundary when
   `block_overnight=false`.
5. **MTF completed-bar contract**: MTF/HTF features at M1 bar t must only use
   COMPLETED higher-timeframe candles (no partial candles).
6. **NY-step contract**: `env.step` advances only NY bars. Skipped bars are
   not agent-visible but are replayed for SL/TP/MTM when a position is open.
7. **Session-label contract**: Broker-tz labels are `asia` | `london` | `ny`
   | `closed`. `closed` covers the broker gap (e.g. 23:50–01:05). CT `*_ct`
   columns stay additive and are not used for tradable masks.

### Mandatory Tests (write before implementation)

v3 tests (keep):

| Test | Description | Priority |
|------|-------------|----------|
| test_pivot_confirmation_has_no_lookahead() | Pivot at bar i only visible at bar i+right | CRITICAL |
| test_swings_only_visible_after_confirmation() | Swings observable only after confirmation | CRITICAL |
| test_ny_open_observation_contains_pre_ny_bars() | 60-bar window at NY open reaches into London | CRITICAL |
| test_session_labels_timezone_correct() | Session labels correct including DST caveat (broker tz has no DST) | HIGH |
| test_resample_uses_full_day_source() | H1/H4/D1 contain Asia/London bars | HIGH |
| test_position_survives_session_boundary() | Position persists across NY sessions when overnight allowed | CRITICAL |
| test_block_overnight_false_no_force_close() | Agent can hold overnight | HIGH |
| test_block_overnight_true_force_close() | Traditional forced close works | HIGH |
| test_eod_age_uses_timestamp() | Age computed from timestamps | MEDIUM |
| test_feature_pipeline_keeps_non_ny_bars() | Pipeline keeps Asia/London bars | CRITICAL |
| test_trading_mask_separate_from_features() | NY mask is separate from features | HIGH |
| test_swing_features_are_causal() | All swing features are causal | CRITICAL |
| test_features_invariant_to_future_data() | Features unchanged when future data mutated | CRITICAL |
| test_mtf_features_use_only_completed_htf_bars() | No partial HTF candles in features | CRITICAL |
| test_observation_window_indices() | Exact bar indices in observation window | HIGH |
| test_eod_episode_continuation() | Episode continues across EOD when block_overnight=false | HIGH |

v4 additions:

| Test | Description | Priority |
|------|-------------|----------|
| test_get_session_closed_label() | 23:50–01:05 is `closed`, not `ny` | CRITICAL |
| test_env_steps_only_ny_bars() | Consecutive step bar times stay inside NY window | CRITICAL |
| test_ny_open_window_not_previous_ny() | First NY bar's window oldest stamp is same-day pre-NY | CRITICAL |
| test_skipped_bars_replay_hits_sl() | Overnight hold: SL in London closes before next NY step | CRITICAL |
| test_fill_latency_last_ny_bar() | Last NY bar does not fill at next-day 16:30 | HIGH |
| test_ftmo_daily_reset_on_ny_jump() | Daily loss resets at NY-session jump, not midnight | HIGH |
| test_align_timeframes_completed_bar() | M1 during 17:00–17:59 H1 does not see that H1's final OHLC | CRITICAL |
| test_po3_not_double_shifted() | Chain F FVG/IFVG lag unchanged after align shift | CRITICAL |
| test_pipeline_cache_bust_on_version() | New bar-cache name is not the unversioned parquet | HIGH |
| test_detect_session_levels_on_pipeline_output() | Asia/London levels non-NaN at NY open | CRITICAL |
| test_ticks_available_outside_ny() | TickBook covers skipped timestamps or fallback is explicit | HIGH |
| test_structure_levels_wrapper_columns() | `last_swing_high` / `last_swing_low` still present | HIGH |

`test_observation_window_indices()` specifics:

- NY opens 16:30, `obs_window=60` M1 bars.
- Oldest observation bar ~15:31, latest = 16:30 (same calendar day).
- Cover first NY bar, second NY bar, first actionable bar (off-by-one prone).

---

## Issue 5: Swing High/Low - Confirmed Volatility-Adaptive Detector

### Current Problem

`quant_rl/features/structure.py` uses rolling max/min with shift.

Issues:

- Uses wicks (high/low) instead of confirmed pivot close
- No ATR filtering — many tiny swings in choppy markets
- No market structure classification (HH/HL/LH/LL)
- No continuous features
- Same parameters for all timeframes (`smt_swing_period: 5`)
- SMT duplicates the detector instead of sharing pivots

### CRITICAL: Pivot Location vs Confirmation Time

Example: bar 3 is pivot high, but the agent cannot know until bar 5 closes:

```
bar:       0 1 2 3 4 5 6
close:     1 2 3 5 3 2 1
                    ^
                  pivot

pivot_location = bar 3
confirmation   = bar 5
```

**Invariant**: Any feature available to the agent must be populated ONLY from
the confirmation bar onward. NEVER at the pivot location bar.

### Pivot API Semantics

```
pivot_high_location  = integer index of the historical pivot bar
pivot_high_price     = pivot price, available from confirmation onward
pivot_high_event     = True exactly on the confirmation bar
```

Downstream:

```python
if pivot_high_event[i]:
    level = pivot_high_price[i]
# Forward-fill level afterward
```

`structure_levels()` stays as a thin wrapper that still returns
`last_swing_high`, `last_swing_low`, `last_swing_high_time`,
`last_swing_low_time` from confirmed, observable swings so liquidity, env
SL/TP, MTF structure, and overlays do not each reimplement the detector.

### Solution: Separated Pivot and Swing Detection

```python
pivots = detect_pivots(df, left=2, right=2)
swings = detect_swings(df, pivots, atr=atr, atr_mult=0.5)
structure = classify_structure(swings)
features = swing_features(df, swings, structure, atr)
```

### New Functions in `quant_rl/features/structure.py`

#### 1. `detect_pivots()` — Confirmed Fractal Pivots

Returns:

- `pivot_high_event`: True at CONFIRMATION bar (not pivot bar)
- `pivot_high_price`: Price of the pivot (available from confirmation onward)
- `pivot_high_location`: Integer index of the actual pivot bar

Mirrored for lows.

#### 2. `detect_swings()` — ATR-Filtered ZigZag Swings

Fully specified state machine:

1. Candidate created: after a confirmed fractal pivot
2. Candidate replaced: if a higher (lower) candidate appears before reversal
3. Reversal threshold: `atr_mult * ATR` sampled at candidate time and FROZEN
4. Price used: close for structural pivot, high/low for liquidity extreme
5. Deterministic ordering: chronological confirmation index, tie-break by price

#### 3. `classify_structure()` — HH/HL/LH/LL

Operates on CONFIRMED swings only:

- `previous_swing_high`: last accepted swing high price
- `previous_swing_low`: last accepted swing low price

Initial state: no previous high → no HH/LH classification.

#### 4. `swing_features()` — Continuous Features for RL

Strict causal rules:

- "last swing" means last swing CONFIRMED and OBSERVABLE by current bar
- No feature may expose information before its confirmation time

#### 5. `TIMEFRAME_CONFIG`

Replaces the single `smt_swing_period` for structure (SMT may keep a period
until it is switched onto shared pivots). Includes M30:

```python
TIMEFRAME_CONFIG = {
    "M1":  {"left": 3, "right": 3, "atr_mult": 0.8, "atr_period": 14},
    "M5":  {"left": 3, "right": 3, "atr_mult": 0.7, "atr_period": 14},
    "M15": {"left": 4, "right": 4, "atr_mult": 0.6, "atr_period": 14},
    "M30": {"left": 4, "right": 4, "atr_mult": 0.55, "atr_period": 14},
    "H1":  {"left": 4, "right": 4, "atr_mult": 0.5, "atr_period": 14},
    "H4":  {"left": 5, "right": 5, "atr_mult": 0.5, "atr_period": 14},
    "D1":  {"left": 5, "right": 5, "atr_mult": 0.5, "atr_period": 14},
}
```

### Close vs Wick: Keep Both

- Structural pivot price = close (for confirmation / HH-HL)
- Liquidity/extreme price = high/low (sweeps, stop hunts)

```
pivot_high_close = 5.0    # structural confirmation
pivot_high_extreme = 5.5  # wick/liquidity extreme
```

[`smt_divergence`](../quant_rl/features/smt.py) must call `detect_pivots`
(or the shared helper) instead of its private `_swing_highs` / `_swing_lows`.

---

## Issue 1: Data Loading — Multi-Timeframe Top-Down Context

### Current Problem

`filter_session()` discards bars outside NY (16:30–23:00). `build_features`
then resamples that M1, so H1/H4/D1 never see Asia/London even though
pipeline comments claim HTF is resampled from `m1_clean` before filtering —
training does not use those HTF caches.

### Top-Down Navigation

```
HTF (H4, D1, H1): Market structure, trend, key levels
  MTF (M15, M5): Intermediate swings, liquidity
    LTF (M1): Entry timing, precise execution
```

The agent does not need raw yesterday in the observation window. Features
carry history (rolling indicators, forward-filled swings, session levels).
The window **does** need same-day pre-NY bars at NY open.

### Solution: All Timeframes Use Full-Day Data

```python
m1_clean = clean(m1_raw, tz=cfg.data.tz)
m1_full = add_session_labels(m1_clean)   # wraps get_session; adds closed
m1_full = add_session_id(m1_full)        # calendar date, unchanged meaning

for tf in cfg.data.timeframes:
    if tf == "M1":
        df = m1_full.copy()
    else:
        tf_bars = resample(m1_full, tf)  # FULL-DAY M1
        df = clean(tf_bars, tz=cfg.data.tz)
        df = add_session_labels(df)
        df = add_session_id(df)
    result[symbol][tf] = df
```

`filter_session()` may remain as a boolean-mask helper. It must not be
applied to the cached feature dataset.

Tick books: stop NY-filtering the stored ticks (or store full-day and let
execution choose). Skipped-bar replay needs quotes outside NY.

### `add_session_labels()` — one broker-tz source of truth

Wrap `get_session()`. Do not duplicate windows. Do not replace CT
`session_ohlc`.

```python
session = add_session_labels(df)
ny_mask = session["session"].eq("ny")
```

Fix `get_session()` so the broker gap is `closed`, not `ny`.

### Single tradable mask

`cfg.session.start` / `end` (broker tz) remain the NY tradable window.
`TradingEnv` builds `ny_indices` from that label/mask.

---

## Issue 2: Buffer Context Window

Path to verify: pipeline → `build_features` → env dataframe → NY `step_idx`
→ `_get_observation()`.

**Invariant**: At first NY trading bar, 60-bar observation contains pre-NY
bars of that day.

Implementation sketch:

```python
self.bars = full_day_bars
self.ny_indices = np.flatnonzero(self.bars["session"].eq("ny"))
# episode cursor over ny_indices; step_idx = ny_indices[ny_pos] (absolute)
start = max(0, self.step_idx - self.obs_window)
seq = self._obs_features.iloc[start : self.step_idx]
```

Episode start: first NY bar that has `obs_window` full-day bars behind it
(typical first session has ~925 pre-NY M1 bars; do not start at
`obs_window` as if the frame were NY-only).

---

## Issue 3: EOD Close — Agent-Managed with Risk Guardrails

### Two-Mode Overnight Handling

| Mode | Config | Behaviour |
|------|--------|-----------|
| Agent-Managed (new default in this work) | `block_overnight: false` | Agent decides. Risk guardrails enforce safety. Skipped bars replayed. |
| Forced Block (backward compat) | `block_overnight: true` | Force-close on last NY bar. No gap hold. |

Default in **current** `default.yaml` is `true`. This work changes the
documented default to `false` only if we accept a train/eval distribution
change; keep `true` as the yaml default unless a later prompt asks to flip
it. The code path for `false` must still be complete and tested.

### EOD State Transition

```
NY session (env steps)
     |
last NY bar / ny_indices jump
     |
if block_overnight:
    CLOSE
else:
    replay skipped bars (SL/TP/MTM/age)
    risk_guard
       |-- CLOSE (thresholds exceeded, or SL/TP during replay)
       |-- HOLD -> next NY session
```

### Position State (survives NY-session boundary)

When `block_overnight=false`, preserve:

```
entry_price
entry_timestamp          # for age
entry_atr                # FROZEN ATR at entry
side, size, sl, tp
best_favorable           # MFE
last_progress_favorable  # last progress checkpoint
stale_counter            # NY steps or replayed bars since last progress — pick one and test it
mfe, mae
```

Extend `Position` or keep sidecar state on `TradingEnv`. Sidecar is
acceptable if `Broker` stays fill-only; document the choice in the PR.

### Stale Counter Algorithm

Measure NEW progress, not historical MFE:

```python
# For long position:
favorable = high - entry_price
new_progress = favorable - last_progress_favorable

if new_progress >= progress_threshold:
    last_progress_favorable = favorable
    stale_counter = 0
else:
    stale_counter += 1
```

Once `best_favorable` is large, old “reset if MFE high” logic would freeze
`stale_counter` at 0 forever. Additional progress beyond the last checkpoint
is required.

During overnight replay, increment stale_counter on replayed bars (or
convert `stale_bars` to a time threshold). Pick one definition, test it,
and do not mix bar counts from NY-only steps with wall-clock hours without
saying so.

### Frozen ATR for Progress Threshold

```python
progress_threshold = progress_threshold_atr * position.entry_atr
```

### Max Age as Universal Risk Guard

Checked **every NY step** (and during skipped-bar replay):

```
Intraday / replay hard guards:
  - max position loss
  - max position age (timestamp-based)

EOD-specific (NY-session boundary only):
  - stale counter
  - final risk assessment
```

```python
age_hours = (current_timestamp - position.entry_timestamp).total_seconds() / 3600
if age_hours > max_age_hours:
    force_close()
```

### Gap and fill-latency

- PnL after a hold uses next available market prices on replayed bars.
- Gap: first skipped bar open vs previous NY close.
- Fill latency: `fill_idx = step_idx + 1 + fill_latency_bars` is a **frame**
  index. On the last NY bar, `step_idx + 1` is the first post-NY bar (01:05
  or 23:01), not next NY. Do not fill overnight entries/exits at next-day
  16:30 by walking `ny_indices` for the fill quote.

### EOD Episode Continuation

When `block_overnight=false`, EOD does NOT terminate the episode. The agent
continues into the next NY session (after replay).

### Config

```yaml
trading:
  block_overnight: false   # yaml default may stay true; this block is the feature surface
  eod_risk:
    max_loss_usd: 500.0
    max_age_hours: 8
    stale_bars: 60
    progress_threshold_atr: 0.25
```

Place keys under `env:` if that matches existing layout (`env.block_overnight`
already exists). Do not invent a parallel `trading:` tree unless yaml already
has one.

### RL Distribution Change Warning

Changing `block_overnight`, observation contents (pre-NY + live Asia/London
levels), HTF lag of one period, and calendar-day VWAP all change environment
dynamics. **Requires retraining.** Golden env traces will break.

[`backtest.engine`](../quant_rl/backtest/engine.py) currently force-closes at
**end of data** (`eod_close`), not at each NY session. Keep that engine
contract consistent with env overnight policy where the engine is used for
the same no-overnight evaluation; do not silently start holding gaps in
one path and flattening in the other.

---

## Issue 4: Plotting — Full Data Range

Invariant: `filter_session()` never removes bars from feature construction
or from the frame passed into overlays.

Grep every call site (`quant_rl/`, `tests/`, eval scripts). After pipeline
fix, [`chart_overlays.py`](../quant_rl/eval/chart_overlays.py) should plot
full-day context around a trade without a second data load path.

---

## HTF completed-bar alignment (Phase 2)

Today:

```python
tf_reindexed = tf_renamed.reindex(result.index, method="ffill")
```

Required: value of the HTF bar labeled at open `T` (left label, interval
`[T, T+period)`) becomes visible on M1 only at `T+period`.

Concrete H1: during 17:00–17:59 M1 must still see the completed 16:00–17:00
candle, not the 17:00–18:00 close/high/low.

Implementation: `tf_df.shift(1)` before ffill **or** shift the index by one
TF period, then ffill. Prefer a single helper in `align_timeframes` so every
Chain A / MTF block inherits the contract.

Audit before merging:

- `build_indicators` HTF blocks
- SMT / structure / liquidity MTF blocks
- FVG ladder via `align_timeframes`
- Chain F `detect_po3_entries` already `shift(1)` — **do not double-shift**

Bump `FEATURE_CACHE_VERSION` (currently `v7-mtf-smt-structure-sweep-fvg-po3-ifvg`).
Rewrite `test_htf_alignment_causal_open_time` into completed-bar semantics
(`test_align_timeframes_completed_bar`). Keep `test_htf_future_invariance`.

---

## Implementation Order

### Phase 0: Contracts and tests first

No production behaviour change except adding failing tests if they are
committed first.

### Phase 1: Session labels + full-day pipeline

`add_session_labels`, `closed` label, stop filtering feature/tick caches,
version bar-cache filenames, prove Asia/London levels non-NaN on pipeline
output.

### Phase 2: Completed-bar `align_timeframes`

Shift, PO3 double-shift audit, feature cache bump, HTF tests.

### Phase 3: Core swing detection (Issue 5)

Pivots / swings / structure / features / `TIMEFRAME_CONFIG`; wrapper
`structure_levels`; SMT shares pivots.

### Phase 4: Trading environment (Issues 2–3)

NY index map, observation window, overnight replay, EOD guards, `Position`
fields, fill-latency on last NY bar, FTMO reset on NY jump. Align
`backtest.engine` overnight policy where it would otherwise diverge.

### Phase 5: Plotting (Issue 4)

Confirm overlays inherit full-day range; remove leftover `filter_session`
on feature/plot frames.

---

## Cache Version

Semantic version encoding:

- `feature_cache_version` / `FEATURE_CACHE_VERSION`: feature schema + HTF
  completed-bar shift + new swing columns
- `bar_cache_version`: full-day vs NY-only parquet (`{symbol}_{tf}_vN.parquet`
  or equivalent; never reuse unversioned `{symbol}_{tf}.parquet`)
- `swing_detector_version`: swing algorithm changes (may be folded into
  feature cache if they always ship together)
- `session_definition_version`: session labels / `closed` (fold into bar
  cache if labels live on the parquet)

Ticks cache similarly busted when NY filter is removed.

---

## Files to Modify (implementation; not this spec pass)

| File | Changes |
|------|---------|
| `quant_rl/data/session.py` | `add_session_labels`; `get_session` → `closed`; `filter_session` mask-only |
| `quant_rl/data/pipeline.py` | Full-day M1/HTF; versioned bar cache; ticks not NY-clipped |
| `quant_rl/data/ticks.py` | Full-day TickBook (or documented fallback) |
| `quant_rl/data/align.py` | Completed-bar shift before ffill |
| `quant_rl/features/structure.py` | New swing API; `structure_levels` wrapper; session label wrap |
| `quant_rl/features/smt.py` | Shared confirmed pivots |
| `quant_rl/features/liquidity.py` | Consume wrapper / confirmed swings (keep +1 shift vs confirmation if still required) |
| `quant_rl/features/build.py` | Full-day primary; per-TF `TIMEFRAME_CONFIG`; `FEATURE_CACHE_VERSION` bump |
| `quant_rl/features/po3_config.py` | Avoid double-shift after align change |
| `quant_rl/envs/trading_env.py` | NY index map; obs window; overnight replay; EOD guards |
| `quant_rl/backtest/broker.py` | Position fields if not env sidecar |
| `quant_rl/backtest/engine.py` | Overnight contract parity where used for the same eval |
| `quant_rl/eval/chart_overlays.py` | New swings; full-day frame |
| `quant_rl/config/default.yaml` | EOD risk keys; cache/version comments; session docs |
| `tests/test_features/test_structure.py` | Pivot/swing/causality tests |
| `tests/test_features/test_htf_alignment.py` | Completed-bar contract |
| `tests/test_session_filter.py` | Labels vs mask; pipeline keeps non-NY |
| `tests/test_envs/test_trading_env.py` (or new) | NY steps, obs window, overnight replay, EOD modes |
| `tests/test_ticks.py` | Non-NY coverage |

---

## Risks

- **Retrain required**: observation contents, live Asia/London levels, HTF
  lag of one period, VWAP calendar-day, optional overnight holds.
- **Golden traces** under `tests/test_envs/` will change.
- **CPU**: `max_episode_steps` still counts NY steps, but skipped-bar replay
  on overnight holds walks Asia+London+gap (~17.5 h of M1) per hold.
- **Cache**: bust both feature parquet and `{symbol}_{tf}.parquet`; leftover
  unversioned files must not be read.
- **Two overnight paths**: env vs `backtest.engine` must not disagree on
  gap holds.

---

## Out of scope (this spec)

- Changing D1 from broker-midnight to NY-close or UTC.
- Replacing CT `session_ohlc` with broker-tz labels.
- Flipping `env.block_overnight` yaml default unless a later prompt says so.
- Geographical (exchange-local) tradable sessions as the eligibility mask.
