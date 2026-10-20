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
| `simulation/` | Native state transitions, Gymnasium adapter, and backtest execution |
| `models/` | Reusable encoders, architectures, and inference components |
| `decisioning/` | Versioned input/action schemas and backend-neutral decision providers |
| `learning/` | Paradigm-neutral fit contracts and supervised/RL learner backends |
| `training/` | Training orchestration, callbacks, run setup, and training reports |
| `evaluation/` | Rollouts, metrics, reanalysis, reports, and plots |
| `experiments/` | Variant resolution, campaign lifecycle, and experiment orchestration |
| `live/`, `integrations/` | Runtime strategy and external broker/platform adapters |
| `compute/` | Resource discovery and compute-side execution helpers |
| `orchestra/` | Multi-model engineering task pipeline |

Data, configuration assets, caches, outputs, logs, baselines, scripts, and
documentation are not Python packages. Their ownership and retention rules are
described in
[`repository_context_local_architecture_plan.md`](repository_context_local_architecture_plan.md).

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

`learning.rl.agent.build_agent` wires the selected PPO/SAC backend and reusable
model encoders. `decisioning/` owns versioned input/action contracts and the
provider boundary. `simulation/` owns state transitions and environment
adapters; `trading/` owns account, risk, order, and execution semantics.
Training composes capabilities through `Learner` contracts. Evaluation and live
runtime consume `DecisionProvider` implementations; the current tree includes
SB3 and rule-based providers plus a supervised Torch reference backend.

Configuration overrides are passed through the stable training command, for
example:

```bash
uv run thesis-train train --mvp --config config/features/pd_context.yaml
```

The operational architecture map is in [`.agents/architecture.md`](../../.agents/architecture.md).
