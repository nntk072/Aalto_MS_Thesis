# Architecture Map — Aalto_MS_Thesis

> Agent navigation map. The authoritative narrative is [docs/architecture.md](../docs/architecture.md).

## Dependency Graph

```text
config/ + raw market data
        ↓
configuration/ → market_data/ → features/
                              ↓
                    simulation/environment/
                    ↙          ↓          ↘
              trading/      models/    simulation/backtest/
                    ↘          ↓          ↙
                         training/
                             ↓
                         evaluation/
                             ↓
                    artifacts/ + outputs/

integrations/mt5/ → live/          experiments/ → compute/
orchestra/ → repository/CI orchestration
```

## Root Package Ownership

| Package | Owns |
|---|---|
| `configuration/` | config loading, overrides, repository paths |
| `market_data/` | ingestion, transforms, splits, data caches |
| `features/` | technical/context/PO3 features and feature-cache identity |
| `trading/` | account, execution, orders, risk, strategies, baselines |
| `simulation/` | Gym environment contracts/runtime and event-driven backtest |
| `models/` | encoders, policies, agent factory, auxiliary models |
| `training/` | setup, callbacks, runners, training artifacts |
| `evaluation/` | rollout, metrics, reanalysis, reporting, plotting |
| `experiments/` | variant/campaign orchestration and provenance |
| `live/` | broker-neutral live runtime |
| `integrations/mt5/` | MetaTrader-specific adapters and domain integration |
| `compute/` | cluster/environment/Slurm implementation |
| `orchestra/` | engineering workflow orchestration |

## Stable Commands

```text
orchestra
thesis-data
thesis-train
thesis-evaluate
thesis-campaign
thesis-compute
```

Use package APIs or these command surfaces from scripts. `scripts/` contains thin research, maintenance, and compute wrappers rather than reusable library logic.

## Cache Versioning

| Constant | File | Current value |
|---|---|---|
| `BAR_CACHE_VERSION` | `market_data/loading/pipeline.py` | `v3-native-timeframes` |
| `FEATURE_CACHE_VERSION` | `features/pipeline/cache.py` | `v24-native-source-timeframes` |

Cache identity must remain causal and include fit scope/configuration inputs that affect generated content.

## Core Scientific Invariants

1. Train-only normalization and causal feature construction.
2. Locked OOS separation plus purged/embargoed walk-forward where configured.
3. Training, evaluation, and backtest execution/risk semantics remain aligned.
4. Checkpoint, seed, dataset, config, cache, and source provenance remain reproducible.
5. Live broker integrations stay outside simulation/training domain ownership.
