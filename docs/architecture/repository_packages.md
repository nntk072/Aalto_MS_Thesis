# Repository packages and commands

The repository root follows the system's domain boundaries. Shared configuration
feeds market data and causal features; simulation consumes those features and
the trading contracts; training and evaluation build on those APIs; experiments
and live integrations compose them.

```text
configuration → market_data → features ─┬→ simulation ← trading
                                         ├→ training → evaluation
                                         └→ experiments
live → integrations/mt5
compute and orchestra provide execution and engineering interfaces
```

## Package ownership

| Package | Owns |
| --- | --- |
| `configuration/` | Config loading, merging, validation, compatibility, hashing, and repository paths |
| `artifacts/` | Artifact identities, manifests, provenance, and safe writes |
| `market_data/` | Bar and tick loading, cleaning, alignment, splits, and data preparation |
| `features/` | Causal indicators, market structure, PO3, feature building, and normalization |
| `trading/` | Accounts, risk limits, orders, execution costs, policies, and strategies |
| `simulation/` | Environment facade, observations/actions, vectorization, and backtest execution |
| `models/` | PPO/SAC agent construction, encoders, policies, and auxiliary models |
| `training/` | Training orchestration, callbacks, run setup, and training reports |
| `evaluation/` | Rollouts, metrics, reanalysis, reports, and plots |
| `experiments/` | Variant resolution, campaign lifecycle, and experiment orchestration |
| `live/`, `integrations/` | Runtime strategy and external broker/platform adapters |
| `compute/` | Resource discovery and compute-side execution helpers |
| `orchestra/` | Multi-model engineering task pipeline |

Data, configuration assets, caches, outputs, logs, baselines, scripts, and
documentation are not Python packages. Their ownership and retention rules are
described in [`architecture_refactor_plan.md`](../architecture_refactor_plan.md).

## Stable commands

Run commands from the repository root after installing the project environment.

```bash
uv sync
uv run thesis-data prepare
uv run thesis-train train --mvp --seed 42
uv run thesis-train baselines
uv run thesis-train backtest
uv run thesis-evaluate reanalyze --help
uv run thesis-campaign status --help
uv run pytest tests/architecture -q
```

The `thesis-train` command also exposes `lstm-baseline` and `vae`. The Python
module entry points remain useful for development, while scripts under
`scripts/` are launch, maintenance, and research wrappers rather than owners of
reusable domain logic.

## Model and environment contract

`models.factory.agent.build_agent` wires the selected PPO/SAC policy and encoder.
The simulation package owns the Gymnasium interface and action/observation
contracts; the trading package owns account, risk, order, and execution rules.
Training composes these services, while evaluation consumes saved run artifacts
and manifests.

Configuration overrides are passed through the stable training command, for
example:

```bash
uv run thesis-train train --mvp --config config/features/pd_context.yaml
```

The operational architecture map is in [`.agents/architecture.md`](../../.agents/architecture.md).
