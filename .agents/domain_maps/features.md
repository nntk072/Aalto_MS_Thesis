# Features Domain

## Primary Files

| File | Role | Lines |
|------|------|-------|
| `quant_rl/features/build.py` | Full feature pipeline orchestrator | 631 |
| `quant_rl/features/indicators.py` | Technical indicators (EMA, RSI, MACD, ATR, etc.) | ~200 |
| `quant_rl/features/smt.py` | SMT divergence between US100/US500 | ~80 |
| `quant_rl/features/structure.py` | Swing detection, pivot points, BOS, structure levels | ~250 |
| `quant_rl/features/liquidity.py` | Liquidity sweep detection | ~100 |
| `quant_rl/features/po3_config.py` | PO3 FVG zone detection and configuration | ~150 |
| `quant_rl/features/po3_state.py` | PO3 manipulation/distribution state | ~120 |
| `quant_rl/features/normalize.py` | Rolling z-score normalization (train-only) | 44 |
| `quant_rl/features/session_ohlc.py` | Session OHLC, prior period H/L, range quadrants | ~100 |
| `quant_rl/data/align.py` | HTF → M1 alignment (used by features) | ~100 |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `build_features()` | `build.py` | Full feature matrix construction |
| `build_indicators()` | `indicators.py` | Compute all technical indicators |
| `smt_divergence()` | `smt.py` | SMT divergence between symbols |
| `detect_pivots()` | `structure.py` | Detect pivot points |
| `detect_swings()` | `structure.py` | Detect swing highs/lows |
| `detect_bos()` | `structure.py` | Break of structure detection |
| `structure_levels()` | `structure.py` | Structure level detection |
| `detect_session_levels()` | `structure.py` | Session-based levels |
| `detect_liquidity_sweeps()` | `liquidity.py` | Liquidity sweep detection |
| `build_fvg_zones()` | `po3_config.py` | Build FVG zone features |
| `detect_fvg()` | `po3_config.py` | Detect fair value gaps |
| `detect_po3_entries()` | `po3_config.py` | Detect PO3 entry signals |
| `build_po3_state()` | `po3_state.py` | Build PO3 manipulation/distribution state |
| `build_ifvg_zone_features()` | `po3_state.py` | Build IFVG zone features |
| `rolling_zscore()` | `normalize.py` | Rolling z-score (train_mask for no leakage) |
| `align_timeframes()` | `align.py` | Forward-fill HTF features onto M1 spine |
| `session_ohlc()` | `session_ohlc.py` | Session OHLC values |
| `prior_period_high_low()` | `session_ohlc.py` | Yesterday/last-week H/L |
| `range_quadrants()` | `session_ohlc.py` | Range quadrant features |
| `MTF_RAW_SUFFIXES` | `build.py` | Raw price-level suffixes to exclude from model seq |

## Direct Dependencies

- `quant_rl/data/pipeline.py` (input data)
- `quant_rl/data/resample.py` (HTF resampling)
- `quant_rl/data/align.py` (HTF alignment)
- `quant_rl/config/__init__.py` (feature config)

## Consumers (Callers)

- `quant_rl/envs/trading_env.py` (consumes feature matrix)
- `quant_rl/train/train_rl.py` (triggers feature build)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_features/test_build_pipeline.py` | Full feature pipeline |
| `tests/test_features/test_htf_alignment.py` | HTF alignment |
| `tests/test_features/test_pivots_swings.py` | Pivot/swing detection |
| `tests/test_features/test_structure.py` | Structure detection |
| `tests/test_features/test_liquidity.py` | Liquidity sweeps |
| `tests/test_features/test_po3_config.py` | PO3 FVG detection |
| `tests/test_features/test_po3_state.py` | PO3 state features |
| `tests/test_features/test_session_ohlc.py` | Session OHLC |
| `tests/test_features/test_variants.py` | Feature variants |
| `tests/test_causal_features.py` | Causal feature verification |

## Do NOT Read For Feature Tasks

- `quant_rl/envs/trading_env.py` (env consumes features but doesn't create them)
- `quant_rl/models/` (models consume features but don't create them)
- `quant_rl/backtest/` (backtest engine — separate domain)
- `quant_rl/train/` (training — orchestrates, doesn't implement features)
- `mt5_trading/` (live trading — separate domain)

## Critical Invariants

1. **Causal only** — no `.shift(-1)` or future reference in any indicator
2. **HTF alignment uses ffill** — `align_timeframes` forward-fills only (never bfill)
3. **Train-only normalization** — `rolling_zscore` fits on `train_mask` only
4. **Cache version bump** — `FEATURE_CACHE_VERSION` must change when columns change
5. **Raw price levels excluded from model seq** — `MTF_RAW_SUFFIXES` lists stems to exclude