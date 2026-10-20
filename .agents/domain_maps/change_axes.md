# Repository Change Axes

Use this map to decide whether a change is an implementation addition, a policy
change, an adapter, or a contract change. The owner column is the source of truth
for semantics; neighboring packages should consume its public boundary.

| Axis of change | Semantic owner | Current boundary | Typical edit surface | Must stay independent from |
|---|---|---|---|---|
| Config parsing, merge, schema and hashing | `configuration/` | Resolved `DictConfig` plus config identity helpers | `configuration/` and its config tests | Feature algorithms and training workflows |
| Market source, event representation, sessions and splits | `market_data/` | Loaders, transforms, tick books and split definitions | One source/transform/split module and matching tests | Models, campaign state and report rendering |
| Feature/context algorithms | `features/` | Feature builder, context modules and feature-cache identity | One feature/context implementation; builder only for stage wiring | `trading/`, model implementation and training workflow |
| Input/feature schema | `features/` and `decisioning/` | Feature views plus versioned decision input/action schemas | Schema owner plus compatibility/regression tests | PPO-specific coordinate interpretation |
| Account state and risk policy | `trading/account/`, `trading/risk/`, `trading/policy/` | Account state and policy-specific risk helpers | One policy/state transition module and contract tests | Broker adapters and model family |
| Order construction and target selection | `trading/orders/` | Geometry, target menus and `trading/orders/selection.py` | One order-construction module and fill/order tests | Simulation runtime internals |
| Execution cost, quote and fill semantics | `trading/execution/` plus market quote contracts | Broker, cost model, quote and tick lookup APIs | Execution policy and parity/regression tests | Learner and report renderer |
| Simulation clock and state transitions | `simulation/` | Environment, runtime services and environment factory | One simulation service and golden-trace tests | Concrete model classes and training runner |
| Neural architecture and inference precision | `models/` | Encoders, factory and `models/runtime/precision.py` | One architecture/runtime implementation and model tests | Compute discovery, campaigns and training workflow |
| Learning paradigm and fitting | `learning/` | `Learner`/`FitRequest`/`FitResult` plus PPO/SAC under `learning/rl/` | Learner backend and fit contract tests | Generic simulation, trading and evaluation semantics |
| Decision provider and semantic action | `decisioning/` | `DecisionProvider`, versioned observation/action schemas and SB3 adapter | Provider contract or backend adapter plus contract tests | Training workflow and execution implementation |
| Evaluation cases, metrics and provenance | `evaluation/` | Rollout, reanalysis, metrics and reporting APIs | One metric/provenance/renderer module and focused tests | Training implementation and learner internals |
| Artifact identity and atomic storage mechanics | `artifacts/` | Artifact IDs, manifests, layouts and atomic writers | Generic mechanic in `artifacts/`; meaning stays with producer | Trading or model-specific semantics |
| Resource capability and scheduler adapter | `compute/` | Device and Slurm helpers | Capability/backend adapter and compute tests | Model classes and scientific policy |
| Campaign lifecycle and experiment variants | `experiments/` | Variant and campaign packages | One campaign lifecycle service and campaign tests | Environment internals and feature algorithms |
| Live runtime and external broker/vendor adapters | `live/`, `integrations/` | Broker-neutral runtime with MT5 adapter | Runtime port or one external adapter plus contract tests | Vendor SDK types crossing into scientific packages |
| Workflow composition and developer orchestration | `training/`, `orchestra/` | Training runners and Orchestra pipeline | One use case/phase and integration tests | Leaf-domain ownership of scientific semantics |

## Axis of Change

This repository map is itself the cross-domain routing contract. Update the owning
row whenever a semantic boundary or expected edit surface changes; keep domain
details in that owner's map and verify its source paths through the architecture test.

## Common Change Maps

| Task | Start here | Contract/test | Expected scope | Keep unchanged |
|---|---|---|---|---|
| Add a market-data source | `market_data/loading/` and source adapter | Dataset/event identity; loader and dataset tests | One source adapter and its contract tests | Feature and evaluation algorithms |
| Add a feature/context transform | `features/context/` or `features/technical/` | Feature schema; focused causal feature test | One algorithm, plus builder wiring only if required | Trading execution and model architecture |
| Add an order target policy | `trading/orders/` | Order geometry and fill invariants | One policy plus order/fill regression tests | Simulation action parsing and model family |
| Change simulation transitions | `simulation/environment/` | Action, observation, account and episode contracts | One transition service and golden trace | Learner implementation and report metrics |
| Add an encoder | `models/encoders/` | Observation/input schema and encoder contract | One encoder and model tests | Data source and execution semantics |
| Add a learner backend | `learning/` | `Learner`/`FitRequest`/`FitResult` contracts (workflow wiring remains) | Backend plus contract tests | Decision provider, trading and simulation contracts |
| Add an evaluation metric | `evaluation/metrics/` | Metric input/result contract | Metric module, numeric fixture and report wiring if needed | Training setup and model fitting |
| Add a compute backend | `compute/` | Resource capability and job request | Backend adapter and compute contract tests | Model classes and scientific configs |

## Verified Source Paths

The architecture test validates every current source path in this table. Planned
packages are intentionally excluded until they exist.

| Current source path | Ownership example |
|---|---|
| `configuration/loader.py` | Config resolution |
| `artifacts/atomic.py` | Generic atomic writes |
| `artifacts/contracts.py` | Storage-independent artifact identity and store protocol |
| `artifacts/storage.py` | Filesystem artifact-store adapter |
| `market_data/loading/pipeline.py` | Bar dataset loading |
| `market_data/contracts.py` | Logical dataset and event-schema contract |
| `features/pipeline/builder.py` | Feature stage composition |
| `features/contracts.py` | Feature/raw/learned input-view schema |
| `features/context/reachability.py` | Reachability feature math |
| `trading/orders/entries.py` | Structural order geometry |
| `trading/orders/stops.py` | Structural stop candidates and provenance |
| `trading/orders/targets.py` | Structural target candidate generation |
| `trading/risk/contracts.py` | Risk-policy semantic contract |
| `trading/orders/selection.py` | TP action selection decoding |
| `simulation/environment/factory.py` | Environment construction |
| `decisioning/actions.py` | Versioned action schema |
| `decisioning/observations.py` | Versioned observation schema |
| `decisioning/observation_state.py` | Observation coordinate constants |
| `learning/rl/agent.py` | Current SB3 agent composition |
| `learning/contracts.py` | Paradigm-neutral fit request and result |
| `learning/rl/checkpoint.py` | SB3 checkpoint compatibility adapter |
| `learning/rl/environment_adapter.py` | Sharded SB3 vector-environment adapter |
| `models/runtime/precision.py` | Model-specific autocast policy |
| `training/runner/seed_runtime.py` | Seed training workflow |
| `evaluation/reanalysis/runner.py` | Checkpoint reanalysis workflow |
| `evaluation/reporting/box_table.py` | Shared report table renderer |
| `experiments/campaigns/execution.py` | Campaign execution lifecycle |
| `compute/environment/device.py` | Hardware capability discovery |
| `live/runtime/strategy.py` | Broker-neutral live strategy |
| `decisioning/contracts.py` | Backend-neutral decision and provider contracts |
| `decisioning/adapters/sb3.py` | Existing SB3 policy adapter |
| `evaluation/rollout/provider.py` | Provider-to-environment rollout adapter |
| `evaluation/contracts.py` | Evaluation case and typed metric data |
| `evaluation/reporting/geometry.py` | Order geometry summary renderer |
| `evaluation/reporting/dataset_renderers.py` | Typed report dataset renderers |
| `compute/contracts.py` | Device and execution-backend capabilities |
| `integrations/mt5/domain/trader.py` | MT5-specific adapter |
| `orchestra/pipeline.py` | Engineering workflow composition |
| `tests/architecture/test_dependency_rules.py` | Enforced package dependency rules |

## Implemented Boundary Work

- `learning/` exposes backend-neutral fit contracts and factories for SB3 and
  the supervised Torch reference learner.
- Both rule-based and SB3 providers use `decisioning.DecisionProvider`; the
  supervised reference checkpoint reloads through the same contract.
- Evaluation and live accept providers. Legacy SB3 model arguments remain only
  at explicit compatibility/composition boundaries.
- `models/` contains reusable Torch architectures and has no Gym/SB3 imports;
  SB3 vectorization, callbacks, checkpointing, and policy behavior live under
  `learning/rl/`.
- Structural stop construction now lives in `trading/orders/stops.py`, alongside
  the separately-owned target candidate module.
- Oversized TP-mode tests are separated by decoder, state-machine, execution,
  and action-decoding change surface.

`tests/architecture/test_dependency_rules.py` enforces the package direction
and the supervised backend's independence from Gymnasium, simulation, and RL.
