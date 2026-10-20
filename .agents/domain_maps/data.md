# Data Domain

## Primary Files

| File | Role |
|------|------|
| `market_data/loading/pipeline.py` | End-to-end pipeline orchestrator; owns `BAR_CACHE_VERSION` |
| `market_data/loading/bars.py` | Load raw CSV bars |
| `market_data/transforms/resample.py` | Resample M1 → higher timeframes |
| `market_data/transforms/cleaning.py` | Clean bars (dedup, sort, validate) |
| `market_data/transforms/sessions.py` | Session labels, NY mask, `get_session()` |
| `market_data/splits/definitions.py` | Train/test date-based split |
| `market_data/transforms/align.py` | Align HTF features to M1 spine |
| `market_data/loading/ticks.py` | TickBook for accurate fill pricing |

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

- `configuration/loader.py` (OmegaConf config)
- `pandas`, `omegaconf`

## Consumers (Callers)

- `features/pipeline/builder.py` (uses pipeline output)
- `training/runner/training.py` (uses split + pipeline)
- `training/runner/backtest.py` (uses pipeline)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_data_split.py` | `split_train_test()`, `split_bars()` |
| `tests/test_session_filter.py` | `filter_session()`, `ny_session_mask()`, cache versioning |
| `tests/test_ticks.py` | `TickBook`, `build_tick_book()` |
| `tests/test_causal_features.py` | Causal alignment verification |

## Do NOT Read For Data Tasks

- `features/` (downstream — read only if feature logic affects data)
- `simulation/environment/` (environment — separate domain)
- `models/` (models — separate domain)
- `simulation/backtest/` (backtest — separate domain)
- `training/` (training — separate domain)
- `integrations/mt5/` (live trading — separate domain)

## Critical Invariants

1. **M1 is the policy sequence, not the only source** — the thesis input uses the supplied M3–D1 bars as native higher-timeframe context; M1 resampling is not a substitute for those source files.
2. **Session labels are eligibility flags** — `filter_session` is mask-only, never drops feature bars
3. **Locked OOS split** — `train_end` / `test_start` in `default.yaml` must not be overridden by variants
4. **Cache version bump** — `BAR_CACHE_VERSION` must change when bar schema changes
## Change Map

| Task | Axis | Start file | Contract and focused test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add a market source | source adapter | `market_data/loading/bars.py` | `DatasetRef` and `MarketEventSchema`; `tests/test_dataset_source_contract.py` | one source and source contract tests | feature algorithms and training workflows |
| Change session or split semantics | dataset policy | `market_data/splits/definitions.py` | partition identity; focused tests under `tests/test_data/` | one transform or split module | model and report code |

## Axis of Change

- **Axis:** market source, event representation, transforms, and split policy.
- **Owner:** `market_data/` owns dataset identity, timestamps, event schema, and partitioning.
- **Stable boundary:** `DatasetRef`, `MarketEventSchema`, and `MarketDataset`.
- **Expected edit radius:** one source, transform, or split implementation with focused data tests.
- **Must not depend on:** training workflows, model internals, or report rendering.

## Verified Source Paths

- `market_data/loading/pipeline.py`
- `market_data/loading/bars.py`
- `market_data/loading/ticks.py`
- `market_data/splits/definitions.py`
- `market_data/transforms/align.py`
- `market_data/contracts.py`
