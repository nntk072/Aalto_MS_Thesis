# Features Domain

## Primary Files

| File | Role |
|------|------|
| `features/pipeline/builder.py` | Full feature pipeline orchestrator; owns `FEATURE_CACHE_VERSION` |
| `features/pipeline/session_stage.py` | Session OHLC, prior-period, and previous-session feature assembly |
| `features/pipeline/po3_stage.py` | PO3/IFVG state assembly and confirmed-sweep audit output |
| `features/pipeline/structure_stage.py` | Swing, structure, and MTF BOS feature assembly |
| `features/technical/indicators.py` | Technical indicators (EMA, RSI, MACD, ATR, etc.) |
| `features/context/smt.py` | SMT divergence between US100/US500 |
| `features/technical/structure.py` | Structure levels, session levels; wraps swings for SL/TP |
| `features/technical/swings.py` | Fractal pivots and ATR-filtered zigzag swings |
| `features/context/liquidity.py` | Liquidity sweeps and BOS detection |
| `features/po3/fvg_detection.py` | FVG configuration and three-bar detection |
| `features/po3/ifvg_detection.py` | IFVG configuration and confirmation rules |
| `features/po3/trigger_rules.py` | HTF/LTF PO3 entry trigger construction |
| `features/po3/zones.py` | FVG zone lifecycle and plotting records |
| `features/po3/config.py` | Compatibility exports for the former combined PO3 module |
| `features/po3/state.py` | PO3 manipulation/distribution state, IFVG zones |
| `features/pipeline/normalization.py` | Rolling z-score normalization with legacy masked-window and frozen-fold modes |
| `features/context/session_levels.py` | CT-anchored session OHLC, prior period H/L, quadrants |
| `features/context/reachability.py` | Causal take-profit room cap and risk-ratio reachability |
| `features/pipeline/context.py` | Assembles policy context, reachable-R and fold-local feature views |
| `market_data/transforms/align.py` | HTF → M1 alignment (used by features) |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `build_features()` | `build.py` | Full feature matrix construction |
| `FEATURE_CACHE_VERSION` | `build.py` | Cache key; bump on schema change |
| `build_indicators()` | `indicators.py` | Compute all technical indicators |
| `smt_divergence()` | `smt.py` | SMT divergence between symbols |
| `detect_pivots()` | `swings.py` | Detect fractal pivot points |
| `detect_swings()` | `swings.py` | ATR-filtered zigzag swings |
| `detect_bos()` | `liquidity.py` | Break of structure detection |
| `detect_liquidity_sweeps()` | `liquidity.py` | Liquidity sweep detection |
| `structure_levels()` | `structure.py` | Causal swing price levels for SL/TP |
| `detect_session_levels()` | `structure.py` | Broker-tz session-based levels |
| `build_fvg_zones()` | `features/po3/zones.py` | Build FVG zone records |
| `detect_fvg()` | `features/po3/fvg_detection.py` | Detect fair value gaps |
| `detect_po3_entries()` | `features/po3/trigger_rules.py` | Detect PO3 entry signals |
| `build_po3_state()` | `po3_state.py` | PO3 manipulation/distribution state |
| `build_ifvg_zone_features()` | `po3_state.py` | IFVG active zone features |
| `rolling_zscore()` | `normalize.py` | Train-mask normalization; `frozen_fold` holds final training statistics through validation |
| `reachable_r_series()` | `context.py` | Reachable reward room from feature-owned session-range estimates |
| `max_tp_distance()` | `reachability.py` | Causal session-range cap from already closed NY sessions |
| `align_timeframes()` | `align.py` | Forward-fill HTF features onto M1 spine |
| `MTF_RAW_SUFFIXES` | `build.py` | Raw price-level suffixes excluded from model seq |

## Direct Dependencies

- `market_data/loading/pipeline.py` (input data)
- `market_data/transforms/resample.py` (HTF resampling)
- `market_data/transforms/align.py` (HTF alignment)
- `configuration/loader.py` (feature config)
- `market_data/transforms/sessions.py` (session eligibility and session clock)

## Consumers (Callers)

- `simulation/environment/env.py` (consumes feature matrix)
- `training/runner/training.py` (triggers feature build)
- `live/runtime/strategy.py` (rebuilds features for live obs)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_features/test_build_pipeline.py` | Full feature pipeline + cache version |
| `tests/test_features/test_htf_alignment.py` | HTF alignment |
| `tests/test_features/test_pivots_swings.py` | Pivot/swing detection |
| `tests/test_features/test_structure.py` | Structure detection |
| `tests/test_features/test_liquidity.py` | Liquidity sweeps and BOS |
| `tests/test_features/test_po3_config.py` | PO3 FVG detection |
| `tests/test_features/test_po3_state.py` | PO3 state features |
| `tests/test_features/test_session_ohlc.py` | Session OHLC |
| `tests/test_features/test_variants.py` | Feature variants |
| `tests/test_causal_features.py` | Causal feature verification |

## Do NOT Read For Feature Tasks

- `simulation/environment/env.py` (env consumes features but doesn't create them)
- `models/` (models consume features but don't create them)
- `simulation/backtest/` (backtest engine — separate domain)
- `training/` (training — orchestrates, doesn't implement features)
- `integrations/mt5/` (live trading — separate domain)

## Critical Invariants

## Change Map

| Task | Axis | Start file | Contract and focused test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add a feature stage | feature transform | `features/pipeline/builder.py` | feature schema and cache identity; `tests/test_features/test_build_pipeline.py` | one stage plus causal regression | trading and training workflow semantics |
| Change an input representation | input schema | `features/contracts.py` | `InputSchema`/`InputView`; `tests/test_input_view_contract.py` | view contract and consumer adapter | PPO-specific observation coordinates |

## Axis of Change

- **Axis:** market-derived feature and context algorithms.
- **Owner:** `features/`; dataset and session semantics remain owned by `market_data/`.
- **Stable boundary:** feature views and the public `features` API.
- **Expected edit radius:** one feature implementation, its focused test, and the pipeline composer only when stage wiring changes.
- **Must not depend on:** trading execution/order code, training workflows, or model implementations.

1. **Causal only** — no `.shift(-1)` or future reference in any indicator
2. **HTF alignment uses ffill** — `align_timeframes` forward-fills only (never bfill)
3. **Fold fit scope** — `frozen_fold` fits rolling statistics on fold training rows and holds the final training statistics through validation; `legacy_masked_window` remains a separately named compatibility mode
4. **Cache version bump** — `FEATURE_CACHE_VERSION` must change when columns change
5. **Raw price levels excluded from model seq** — `MTF_RAW_SUFFIXES` lists stems to exclude
## Verified Source Paths

- `features/pipeline/builder.py`
- `features/contracts.py`
- `features/input_view.py`
- `features/pipeline/session_stage.py`
- `features/pipeline/po3_stage.py`
- `features/pipeline/structure_stage.py`
- `features/context/reachability.py`
- `features/technical/indicators.py`
- `tests/test_features/test_stage_a_observation.py`
- `features/contracts.py`
