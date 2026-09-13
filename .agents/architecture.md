# Architecture Map — Aalto_MS_Thesis

## Dependency Graph

```
raw CSV
   ↓
data/loader.py → data/resample.py → data/clean.py → data/session.py
                                                      ↓
                                                data/pipeline.py (caches parquet)
                                                      ↓
                                                data/split.py (train/test split)
                                                      ↓
                    features/build.py ← features/{indicators, smt, structure, swings,
                                                      liquidity, po3_*, session_ohlc, normalize}.py
                                                      ↓
                                                features/ matrix (parquet cache)
                                                      ↓
            envs/trading_env.py ← backtest/{engine, broker, account, costs, guardrails, risk}.py
                                        ↓
                                  models/agent.py ← models/{encoder, vae, base, auxiliary}.py
                                        ↓
                                    train/train_rl.py ← config + eval/rollout + evaluation/metrics
                                        ↓
                                    outputs/ (equity CSV, plots, models)
```

## Data Flow

```
US100.cash_M1_*.csv ──→ load_bars() ──→ resample(M5/M15/H1) ──→ clean()
                                                              ↓
US500.cash_M1_*.csv ──→ load_bars() ──→ smt_divergence() ←─────┘
                                ↓
                        add_session_labels() → add_session_id()
                                ↓
                        run_pipeline() → {symbol}_{tf}_{BAR_CACHE_VERSION}.parquet
                                ↓
                        split_train_test() → train/test split at 2025-12-31
                                ↓
                        build_features() → {symbol}_features_{FEATURE_CACHE_VERSION}.parquet
                                ↓
                        TradingEnv(bars, features) → obs dict → model → action → reward
```

## Domain Boundaries

| Domain | Files | Incoming From | Outgoing To |
|--------|-------|---------------|-------------|
| data | `data/*.py` | raw CSV | features, train |
| features | `features/*.py`, `data/align.py` | data | envs, train |
| envs | `envs/*.py` | features, backtest, models | train, evaluation |
| backtest | `backtest/*.py` | (none) | envs |
| models | `models/*.py` | envs | train, live |
| train | `train/*.py` | all above | evaluation, eval |
| evaluation | `evaluation/*.py` | envs, models | outputs |
| eval | `eval/*.py` | envs, models | outputs |
| live | `live/*.py`, `mt5_trading/` | models | MT5 |
| orchestra | `orchestra/*.py` | (none) | CI verification |

## Cache Versioning

| Constant | File | Current value | Purpose |
|----------|------|---------------|---------|
| `BAR_CACHE_VERSION` | `data/pipeline.py` | `v2-full-day` | Busts stale bar parquet caches |
| `FEATURE_CACHE_VERSION` | `features/build.py` | `v8-full-day-completed-htf-swings` | Busts stale feature caches |

**Rule**: Bump when output schema changes. Use `--force` or delete stale parquet when
changing opt-in feature flags without a version bump.

## Config Hierarchy

```
quant_rl/config/default.yaml     ← base config (all defaults)
    ↓ override (--config or --strategy)
config/idea1_po3_ifvg.yaml       ← PO3 + IFVG strategy overlay
config/idea2_distribution.yaml   ← Distribution strategy overlay
config/features_*_mtf.yaml       ← MTF feature expansion variants
```

All config read via `cfg.<path>` in code. Adding a feature flag? Add YAML key first.
See [docs/config.md](../docs/config.md).

## Temporal Boundaries

| Boundary | Rule |
|----------|------|
| Train/test split | `train_end: "2025-12-31"`, `test_start: "2026-01-01"` (locked) |
| Walk-forward purge | `purge_bars: 60` |
| Walk-forward embargo | `embargo_bars: 20` |
| NY session | `[16:30, 23:00]` broker-tz (eligibility only) |
| Feature warmup | `dropna(how="all")` removes leading NaNs |
