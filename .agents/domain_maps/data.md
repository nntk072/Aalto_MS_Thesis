# Data Domain

## Primary Files

| File | Role |
|------|------|
| `quant_rl/data/pipeline.py` | End-to-end pipeline orchestrator; owns `BAR_CACHE_VERSION` |
| `quant_rl/data/loader.py` | Load raw CSV bars |
| `quant_rl/data/resample.py` | Resample M1 → higher timeframes |
| `quant_rl/data/clean.py` | Clean bars (dedup, sort, validate) |
| `quant_rl/data/session.py` | Session labels, NY mask, `get_session()` |
| `quant_rl/data/split.py` | Train/test date-based split |
| `quant_rl/data/align.py` | Align HTF features to M1 spine |
| `quant_rl/data/ticks.py` | TickBook for accurate fill pricing |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `run_pipeline()` | `pipeline.py` | Load, resample, clean, label, cache all symbols/TFs |
| `BAR_CACHE_VERSION` | `pipeline.py` | Bar cache key (`v2-full-day`) |
| `build_tick_books()` | `pipeline.py` | Build/load TickBook per symbol |
| `load_bars()` | `loader.py` | Load raw CSV into DataFrame |
| `resample()` | `resample.py` | Resample bars to higher timeframe |
| `clean()` | `clean.py` | Dedup, sort, validate bars |
| `get_session()` | `session.py` | Broker-tz session label (`asia`/`london`/`ny`/`closed`) |
| `add_session_labels()` | `session.py` | Add session column |
| `add_session_id()` | `session.py` | Add unique session_id per date |
| `ny_session_mask()` | `session.py` | Boolean mask for NY window |
| `filter_session()` | `session.py` | Mask-only helper (never drop feature bars) |
| `split_train_test()` | `split.py` | Split bars+features at date boundary |
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
| `tests/test_session_filter.py` | `filter_session()`, `ny_session_mask()`, cache versioning |
| `tests/test_ticks.py` | `TickBook`, `build_tick_book()` |
| `tests/test_causal_features.py` | Causal alignment verification |

## Do NOT Read For Data Tasks

- `quant_rl/features/` (downstream — read only if feature logic affects data)
- `quant_rl/envs/` (environment — separate domain)
- `quant_rl/models/` (models — separate domain)
- `quant_rl/backtest/` (backtest — separate domain)
- `quant_rl/train/` (training — separate domain)
- `mt5_trading/` (live trading — separate domain)

## Critical Invariants

1. **M1 is the only true source** — higher TFs are resampled from M1 files in `data.m1_files`
2. **Session labels are eligibility flags** — `filter_session` is mask-only, never drops feature bars
3. **Locked OOS split** — `train_end` / `test_start` in `default.yaml` must not be overridden by variants
4. **Cache version bump** — `BAR_CACHE_VERSION` must change when bar schema changes
