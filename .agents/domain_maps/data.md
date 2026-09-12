# Data Domain

## Primary Files

| File | Role | Lines |
|------|------|-------|
| `quant_rl/data/pipeline.py` | End-to-end pipeline orchestrator | 152 |
| `quant_rl/data/loader.py` | Load raw CSV bars | ~60 |
| `quant_rl/data/resample.py` | Resample M1 → higher timeframes | ~80 |
| `quant_rl/data/clean.py` | Clean bars (dedup, sort, validate) | ~50 |
| `quant_rl/data/session.py` | Session labels, NY mask, session IDs | 132 |
| `quant_rl/data/split.py` | Train/test date-based split | 85 |
| `quant_rl/data/align.py` | Align HTF features to M1 spine | ~100 |
| `quant_rl/data/ticks.py` | TickBook for accurate fill pricing | ~120 |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `run_pipeline()` | `pipeline.py` | Load, resample, clean, label, cache all symbols/TFs |
| `build_tick_books()` | `pipeline.py` | Build/load TickBook per symbol |
| `load_bars()` | `loader.py` | Load raw CSV into DataFrame |
| `resample()` | `resample.py` | Resample bars to higher timeframe |
| `clean()` | `clean.py` | Dedup, sort, validate bars |
| `add_session_labels()` | `session.py` | Add asia/london/ny/closed column |
| `add_session_id()` | `session.py` | Add unique session_id per date |
| `ny_session_mask()` | `session.py` | Boolean mask for NY window |
| `filter_session()` | `session.py` | Mask-only helper (never drop feature bars) |
| `split_train_test()` | `split.py` | Split bars+features at date boundary |
| `split_bars()` | `split.py` | Split single bar frame |
| `align_timeframes()` | `align.py` | Forward-fill HTF features onto M1 |
| `TickBook` | `ticks.py` | Tick data container with `quote_at()` |

## Direct Dependencies

- `quant_rl/config/__init__.py` (OmegaConf config)
- `pandas`, `omegaconf`

## Consumers (Callers)

- `quant_rl/features/build.py` (uses pipeline output)
- `quant_rl/train/train_rl.py` (uses split + pipeline)
- `quant_rl/train/run_backtest.py` (uses pipeline)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_data_split.py` | `split_train_test()`, `split_bars()` |
| `tests/test_session_filter.py` | `filter_session()`, `ny_session_mask()` |
| `tests/test_ticks.py` | `TickBook`, `build_tick_book()` |
| `tests/test_causal_features.py` | Causal alignment verification |

## Do NOT Read For Data Tasks

- `quant_rl/features/` (downstream — read only if feature logic affects data)
- `quant_rl/envs/` (environment — separate domain)
- `quant_rl/models/` (models — separate domain)
- `quant_rl/backtest/` (backtest — separate domain)
- `quant_rl/train/` (training — separate domain)
- `mt5_trading/` (live trading — separate domain)