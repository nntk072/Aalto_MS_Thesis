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
                    features/build.py ← features/{indicators, smt, structure, liquidity, po3_*, normalize, session_ohlc}.py
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
                        run_pipeline() → {symbol}_{tf}_v2-full-day.parquet
                                ↓
                        split_train_test() → train/test split at 2025-12-31
                                ↓
                        build_features() → features.parquet
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
| train | `train/*.py` | all above | evaluation |
| evaluation | `evaluation/*.py`, `eval/rollout.py` | envs, models | outputs |
| live | `live/*.py`, `mt5_trading/` | models | MT5 |

## High-Coupling Pairs (from code-review-graph)

| Pair | Edge Count | Implication |
|------|-----------|-------------|
| backtest ↔ tests | 67 | Backtest changes require broad test coverage |
| features ↔ tests | 22 | Feature changes ripple to many tests |
| data ↔ tests | 36 | Data pipeline changes need thorough testing |
| envs ↔ tests | 152 | Env is the most tested module |

## Cache Versioning

| Constant | File | Purpose |
|----------|------|---------|
| `BAR_CACHE_VERSION` | `data/pipeline.py` | Busts stale parquet caches |
| `FEATURE_CACHE_VERSION` | `features/build.py` | Busts stale feature caches |

**Rule**: Bump when output schema changes. Never reuse stale caches silently.

## Config Hierarchy

```
config/default.yaml          ← base config (all defaults)
    ↓ override
config/idea1_po3_ifvg.yaml   ← PO3 + IFVG strategy variant
config/idea2_distribution.yaml ← Distribution strategy variant
```

All config read via `cfg.<path>` in code. Adding a feature flag? Add YAML key first.

## Temporal Boundaries

| Boundary | Rule |
|----------|------|
| Train/test split | `train_end: "2025-12-31"`, `test_start: "2026-01-01"` (configurable) |
| Walk-forward purge | `purge_bars: 60` |
| Walk-forward embargo | `embargo_bars: 20` |
| NY session | `[16:30, 23:00]` broker-tz (eligibility only) |
| Feature warmup | `dropna(how="all")` removes leading NaNs |