# Aalto_MS_Thesis — Full Context-Local Architecture Plan

## Goal

The next refactor should optimize the repository for safe local reasoning.
A normal developer or coding-agent change should usually require:

1. one change map,
2. one implementation module,
3. at most one or two boundary contracts,
4. one focused test module,
5. an integration or regression check only when a boundary is crossed.

The goal is not many tiny files. The goal is a small reading radius, explicit ownership, stable contracts, and predictable dependency direction.

## Pre-refactor repository audit

The counts and file sizes in this section are the initial inventory that motivated
the plan. They describe the pre-refactor tree, not the current repository. See
the completion record at the end of this document and the verified source paths
in `.agents/domain_maps/` for the resulting structure.

The current tracked repository contains roughly 928 files, 601 Python files, 316 production Python modules, and 63,658 LOC in production packages. The top-level package split is already substantially improved. The remaining problem is knowledge concentration inside domains and workflows.

Largest or highest-coupling knowledge hubs include:

- training/runner/seed_runtime.py — 970 LOC, 27 local imports across 8 domains.
- evaluation/reanalysis/runner.py — 975 LOC, 18 local imports across 8 domains.
- training/runner/setup_runtime.py — 647 LOC, 19 local imports across 8 domains.
- features/pipeline/builder.py — 966 LOC, about 20 local imports.
- experiments/campaigns/execution.py — 1,881 LOC, 34 definitions.
- orchestra/pipeline.py — 1,088 LOC, 39 definitions.
- trading/orders/entries.py — 973 LOC, 21 definitions.
- models/encoders/core.py — 636 LOC, 51 definitions.
- simulation/environment/execution_runtime.py — 813 LOC.
- simulation/environment/step_runtime.py — 742 LOC.
- simulation/environment/initialization_runtime.py — 742 LOC.
- training/callbacks/core.py — 706 LOC, 60 definitions.
- evaluation/plotting/plots.py — 1,264 LOC.
- evaluation/plotting/plots_interactive.py — 1,107 LOC.
- evaluation/metrics/decision_diversity.py — 823 LOC, 42 definitions.

The audit also found concrete reverse dependencies:

- features calls trading logic for reachability.
- trading calls environment target-decoding logic.
- model encoders depend on environment state constants.
- evaluation reanalysis imports training environment setup.
- compute contains model-specific precision behavior and imports a model class.

These ownership seams should be corrected before broad file splitting.

## Core architecture rule

Use five conceptual layers:

L0 contracts, schemas, immutable value objects
L1 pure domain primitives
L2 domain services
L3 application workflows and use-cases
L4 adapters such as CLI, filesystem, Slurm, MT5, plotting and export

Dependencies normally flow downward. Broad imports are allowed only in explicit composition roots. Leaf modules never import workflows.

## Meta-principle: design around axes of change, not today's nouns

The repository must not be architected around any current implementation concept being permanent.
That includes not only RL, but also CSV files, M1 bars, handcrafted features, PO3/IFVG, FTMO rules,
one-symbol training, one execution engine, one broker, one artifact layout, one scheduler, one plotting
stack, or even the assumption that every experiment has the same pipeline shape.

The durable architectural question is:

> What can plausibly change independently, and what semantic contract should remain stable when it does?

Use **axes of change** as the reason for boundaries. A module deserves a replaceable seam when at least
one of these is true:

- there are already multiple implementations;
- another implementation is scientifically plausible during the thesis/project lifetime;
- it represents external I/O or infrastructure;
- it has an independent schema/version lifecycle;
- it changes for a different reason than its callers;
- it is expensive/risky enough that callers should not know its internals.

Do **not** create an interface merely because something could theoretically change. The objective is
replaceability with low cognitive cost, not a universal plugin framework.

### Change-axis matrix

| Capability / concept | Current assumption | Plausible future replacement | Stable seam should expose |
|---|---|---|---|
| Market source | local MT5/CSV-style exports | Parquet lake, API, DB, Kafka/stream, alternate vendor | dataset/source identity + event schema |
| Market representation | M1 bars + ticks | tick-only, order book, event bars, multi-timeframe native data | time/event semantics, symbol, price/size fields |
| Universe | primarily US100 + secondary context | multi-asset, futures, FX, equities, portfolio | instrument identity + calendar/session metadata |
| Storage | filesystem paths | object store, DB, remote cache | ArtifactRef/DatasetRef, never raw path semantics |
| Splits | fixed train/test + walk-forward | rolling online windows, cross-asset splits, purged CV variants | SplitSpec + immutable partition identity |
| Features | handcrafted indicators/PO3/context | learned embeddings, raw sequence input, feature store, no handcrafted features | FeatureView / InputView schema |
| Representation | TCN/GRU/Transformer encoders | CNN, SSM, foundation model embeddings, no encoder | Encoder/Predictor contract, not framework class |
| Learning | PPO/SAC | supervised DL, offline RL, imitation, self-supervised, no learning | Learner + FitRequest/FitResult |
| Decision maker | SB3 policy | Torch predictor, rule engine, ensemble, remote model service | DecisionProvider -> semantic Decision |
| Objective | DSR/reward shaping | PnL, constrained objective, labels/losses, multi-objective | Objective/Target specification owned by learning backend |
| Strategy semantics | PO3/IFVG overlays | another strategy, pure learned decisioning, portfolio policy | strategy/decision contract, not feature internals |
| Risk | FTMO-style guardrails | broker rules, portfolio VaR, fixed risk, no prop-firm constraints | RiskPolicy + RiskState transition |
| Order construction | structural SL/TP menus | direct price prediction, market/limit policy, portfolio rebalance | OrderIntent / TargetPlan |
| Execution | current broker/cost/fill semantics | alternate venue, L2 simulator, stochastic fills, real broker | ExecutionRequest -> Fill/ExecutionEvent |
| Simulation | Gym trading env | event simulator, vector simulator, historical replay, live shadow | state-transition/evaluation protocol |
| Backtest engine | current event replay + backtrader parity | another engine or exchange replay | BacktestRequest/Result + execution contract |
| Evaluation | trading metrics + reanalysis | forecasting metrics, calibration, portfolio metrics, online monitoring | EvaluationCase -> MetricResult/Report |
| Provenance | current manifests | richer lineage store/DB | versioned artifact lineage envelope |
| Config | OmegaConf/YAML | typed TOML/JSON, generated experiment specs, service config | resolved immutable domain config |
| Compute | Triton/Slurm + local | cloud batch, Kubernetes, workstation-only, other HPC | ResourceProfile/ExecutionBackend |
| Live integration | MT5 | another broker/API, paper broker, exchange gateway | broker-neutral live ports |
| Reporting | matplotlib/plotly/files | notebook/dashboard/web report | typed report data before renderer |
| Orchestration | current campaign/orchestra flows | another scheduler/agent system | workflow state + task/result contracts |

This table is not a mandate to implement all alternatives now. It defines where today's implementation
must avoid leaking assumptions into neighboring packages.

## Capability graph, not one permanent pipeline

The architecture should not encode a single mandatory chain such as:

`market_data -> handcrafted features -> RL environment -> PPO -> trading evaluation`.

That path is one composition. Other valid future compositions include:

```text
bars -> handcrafted features -> rule strategy -> backtest

bars -> raw windows -> supervised model -> DecisionProvider -> execution simulator

ticks/order book -> learned representation -> portfolio policy -> event simulator

offline dataset -> labels -> supervised learner -> forecasting evaluation

market stream -> feature service -> DecisionProvider -> live broker adapter

historical events -> no ML at all -> execution/risk research
```

Therefore the long-term architecture is a graph of **capabilities with typed boundaries**, assembled by
application workflows. Workflows choose which capabilities participate; leaf packages must not assume
that every other capability exists.

Key consequence: `features/` must be bypassable, `simulation/` must be optional for supervised fitting,
`learning/` must be optional for rule-based systems, and `live/` must not require training code.

## Stable semantic contracts

Prefer semantic contracts over library/framework contracts. Cross-domain boundaries should speak in
domain objects and schemas, not `pandas.DataFrame`, `gym.Space`, SB3 models, Torch tensors, filesystem
paths, or Slurm job IDs unless those types are truly part of that boundary.

Candidate stable contracts include:

- `InstrumentId`, `TimeRange`, `MarketEventSchema`, `DatasetRef`, `DatasetManifest`;
- `SplitSpec`, `PartitionRef`;
- `InputSchema`, `FeatureSchema`, `FeatureView`;
- `DecisionContext`, `Decision`, `DecisionProvider`;
- `RiskState`, `RiskDecision`, `RiskPolicy`;
- `OrderIntent`, `TargetPlan`, `ExecutionRequest`, `FillEvent`;
- `AccountSnapshot`, `PositionSnapshot`;
- `SimulationRequest`, `EpisodeResult`, `BacktestResult`;
- `FitRequest`, `FitResult`, `ModelArtifactRef`;
- `EvaluationCase`, `MetricResult`, `ReportDataset`;
- `ArtifactRef`, `ProvenanceRef`, `ResourceProfile`.

These contracts should be small, versioned only when cross-domain compatibility needs it, and owned by
the domain that defines the semantics. Avoid a giant `common/` package that becomes a new knowledge hub.

## Ports and adapters rule

Use ports/adapters at volatile or external boundaries:

- data sources and stores;
- artifact stores;
- execution venues/brokers;
- decision providers;
- learning backends;
- compute/scheduler backends;
- report renderers;
- external integrations.

Do not create a universal plugin registry. Prefer one small factory/registry per capability only when
runtime selection is actually needed. Internal pure algorithms can remain ordinary functions/classes.

Each replaceable capability should define:

1. semantic input/output contract;
2. config contract owned by that capability;
3. implementation identity/version;
4. compatibility policy;
5. focused contract tests;
6. artifact/provenance fields if it affects scientific results.

Callers depend on 1–2; they should not know 3–6 implementation details.

## Unknown-future-change rule

We cannot enumerate every future concept. The architecture therefore needs rules that contain changes that
were not anticipated when the current tree was designed.

When a new concept appears, classify it before placing code:

1. **new implementation of an existing capability** — add behind the existing contract;
2. **new policy within an existing domain** — add a local strategy/policy implementation;
3. **new external adapter** — implement the domain port; do not leak SDK types inward;
4. **new scientific capability** — create a bounded package only when it has its own semantics/lifecycle;
5. **cross-cutting concern** — keep mechanics low-level, but keep semantic interpretation in each owner;
6. **contract-breaking research change** — version the boundary explicitly instead of silently changing meaning.

Before creating a new top-level package, require answers to:

- What semantic concept does it own that no existing package owns?
- What are its stable inputs/outputs?
- Which existing packages may depend on it?
- Which packages must never depend on it?
- Does it have an independent config/schema/artifact lifecycle?
- Can the implementation be replaced without callers changing?

If those answers are weak, it likely belongs inside an existing bounded context rather than at repository root.

### Anti-coupling rules for unknown future concepts

- no generic `AppContext`/`SeedRunContext` carrying the entire repository state into leaf code;
- no whole-repository `DictConfig` passed into algorithms;
- no implicit DataFrame-column conventions across domains without a declared schema;
- no filesystem path used as scientific identity;
- no external SDK object crossing its adapter boundary;
- no environment variable read deep inside scientific logic;
- no package importing a workflow merely to reuse one helper;
- no `utils.py`, `common.py`, `core.py`, or `runtime.py` becoming a dumping ground without one cohesive owner;
- no feature/model/trading semantics encoded only in CLI flags or filenames;
- no artifact consumer inferring meaning from directory names when a manifest/schema can state it explicitly.

These rules are intentionally more important than predicting the next model, feature family or data source.

## Contract evolution policy

Internal functions can evolve freely inside their package. Cross-domain semantic contracts evolve deliberately.

For each public contract change, choose one of:

- backward-compatible additive change;
- explicit version bump with migration adapter;
- new capability contract living beside the old one;
- deliberate retirement after all producers/consumers migrate.

Never overload the same field/name with new semantics merely to preserve a filename or Python signature.
Scientific provenance should record the contract/schema version that affected a result.

Compatibility code belongs at the boundary (loader/adapter/migration), not scattered through core algorithms.

## Future-proof learning principle: RL is one backend, not the architecture

The repository must not encode the assumption that the decision-maker is an RL agent.
PPO/SAC are current implementations. A future supervised DL model, sequence classifier,
return/risk forecaster, imitation learner, offline-RL learner, or rule-based policy must be
able to reuse the same market-data, feature, trading, simulation, artifact, evaluation and
live-runtime boundaries.

Separate four concepts that are currently partially fused:

1. **representation/model architecture** — TCN, GRU, Transformer, MLP, VAE, heads;
2. **learning paradigm** — online RL, offline RL, supervised DL, imitation, self-supervised;
3. **decision runtime** — consume a stable observation/input contract and emit a stable decision;
4. **application workflow** — prepare data, fit, validate, evaluate, persist, campaign execution.

The dependency rule is:

- model architecture must not know PPO, SAC, Gym, campaign runners, or evaluation;
- decision runtime must not know how its model was trained;
- simulation must not know which model family consumes its observations;
- evaluation must evaluate a decision provider, not a PPO-specific object;
- live runtime must load a decision provider, not a training implementation;
- RL-specific Gym/SB3 adaptation belongs in an RL backend, not in generic model code;
- supervised-DL datasets, labels and losses belong in a supervised backend, not in generic features or evaluation.

This is the architectural seam that makes a future RL -> DL replacement local instead of repository-wide.

## Target dependency direction

The scientific code should become acyclic, but not forced into one vertical stack. Think in three groups:

```text
FOUNDATION CAPABILITIES
configuration   artifacts   market_data   compute

SCIENTIFIC CAPABILITIES
features/representation   trading   simulation/backtest   models
decisioning               learning   evaluation

APPLICATION COMPOSITION
training workflows   experiments/campaigns   live   integrations   orchestra
```

Application composition roots may combine several scientific capabilities. Scientific leaf packages should
depend only on the contracts they need, never on an application workflow.

Examples of allowed relationships:

```text
market_data -> features
market_data -> supervised dataset builder
features -> decisioning input assembly
trading contracts -> simulation
trading contracts -> live
simulation contracts -> learning/rl
models -> learning backends
models -> decisioning adapters
decisioning -> evaluation/live
evaluation -> artifacts
training workflows -> learning + decisioning + evaluation
experiments -> training/evaluation/compute
```

Examples that should be structurally impossible:

```text
market_data -> training runner
features -> live broker implementation
trading -> simulation runtime internals
models -> Slurm
evaluation -> training internals
decisioning -> learner implementation
compute -> model class
```

This is an ownership direction, not a requirement that every package imports the package immediately before it.

Additional rules:

- compute owns hardware and scheduler facts and must not import model classes.
- live consumes public feature/trading/decisioning APIs.
- integrations contains external platform adapters and depends on live contracts.
- orchestra remains engineering infrastructure and should not own trading or research semantics.

No generic package should depend on `learning.rl` merely because PPO is the current thesis implementation.

## Stable decision boundary

Introduce a small decision-facing package. The exact name can be `decisioning/`, `decision/`,
or `agents/`; `decisioning/` is used in this plan to avoid implying RL.

Target:

decisioning/
  contracts/
    input.py
    decision.py
    predictor.py
    metadata.py
  runtime/
    provider.py
    batching.py
  adapters/
    sb3.py
    torch.py
    rule_based.py
  registry.py

Core contracts should look conceptually like:

- `DecisionInput` — stable feature/account/context input;
- `Decision` or `ActionIntent` — stable semantic output;
- `DecisionProvider.predict(input) -> Decision`;
- `PredictorMetadata` — input schema, output schema, model family, training paradigm, version;
- `DecisionArtifactRef` — identifier for the persisted provider/model bundle.

Do not make these contracts tensors or Gym spaces unless that is genuinely part of the boundary.
The contract should describe semantics first; adapters convert to NumPy, Torch tensors, SB3
observations, logits or probability vectors internally.

Examples:

- PPO adapter: observation -> SB3 policy -> action coordinates -> `Decision`;
- supervised Transformer adapter: feature tensor -> logits/regression heads -> `Decision`;
- rule-based adapter: feature row -> deterministic `Decision`.

Trading/execution sees the same semantic decision regardless of which provider produced it.

## Learning backend architecture

Add a paradigm-oriented package rather than allowing `training/` or `models/` to accumulate
PPO-specific and future DL-specific code.

Target:

learning/
  contracts/
    learner.py
    fit_request.py
    fit_result.py
    dataset.py
  rl/
    ppo/
      algorithm.py
      policy.py
      rollout.py
      callbacks.py
      config.py
    sac/
      algorithm.py
      policy.py
      replay.py
      config.py
    environment_adapter.py
    action_adapter.py
  supervised/
    datasets/
    labels/
    losses/
    trainers/
    validation/
    config.py
  imitation/
  offline_rl/

Only create future backend directories when they become real work; the boundary should be designed now,
but speculative implementations should not be added.

`Learner` is the workflow-facing abstraction. A training workflow should be able to say conceptually:

prepare_fit_request -> learner.fit(...) -> FitResult -> persist artifact

without knowing whether the learner ran PPO rollouts, supervised mini-batches, behavior cloning,
or another optimization procedure.

`FitResult` should expose backend-neutral facts such as artifact reference, training history,
validation summaries, source/config identity and schema identity. Backend-specific diagnostics
can live under a typed backend metadata field.

## Concrete cycle fixes

### features -> trading

Current feature reachability calls trading.positions.tp_reach.
Move pure reachability math into features/context/reachability.py or another low-level market-context module. Trading may consume the resulting feature or shared primitive, but feature generation must not depend on trading runtime behavior.

### trading -> simulation

Current trading order construction imports simulation.environment.target_decoding.
Move TP selection and fraction decoding to trading/orders/selection.py or a contract-level action-decoding module. Trading must not depend on environment runtime internals.

### simulation -> models

Simulation should expose environment/state-transition contracts and should not import a concrete model.
RL obtains a Gym/SB3-compatible view through `learning/rl/environment_adapter.py`.
Optional representation providers used by observations should depend on a neutral predictor/encoder protocol,
not a concrete VAE class.

### models <-> compute

Move model-specific autocast/precision logic from compute/environment/device.py to models/runtime/precision.py. Compute should return hardware capabilities; models decide how to use them.

### evaluation <-> training

Move shared evaluation environment construction out of training.setup.environments. Put it in simulation/environment/factory.py or evaluation/setup/environment.py. Evaluation must never import training implementation.

## Public API strategy

Each top-level package exposes a deliberately small supported API. Cross-package callers use those APIs or contract modules. They do not import private runtime, runner, mixin, or compatibility internals.

Add architecture tests so a private cross-domain import fails CI.

The most important future-proof public APIs should be backend-neutral:

- `features.build_features(...)`;
- `trading` semantic order/risk contracts;
- `simulation` state-transition/environment contracts;
- `models` reusable architecture constructors;
- `decisioning.load_provider(...)` / `DecisionProvider`;
- `learning.build_learner(...)` / `Learner`;
- `evaluation.evaluate_provider(...)`;
- `training.run_training(...)`.

Avoid public APIs named around one algorithm such as `build_ppo_agent()` unless they live inside
the explicitly RL-specific backend.

## Configuration

Keep configuration as the owner of parsing, merge order, hashing, schema version and compatibility migration.

Stop passing the complete DictConfig deep into leaf algorithms. Each domain resolves a typed semantic config:

- features/config.py
- trading/config.py
- simulation/config.py
- models/config.py
- training/config.py
- evaluation/config.py
- experiments/config.py

A feature detector should receive FeatureConfig rather than knowing PPO, FTMO, training, campaign and evaluation configuration.

## Artifacts

Keep artifacts as low-level mechanics:

- identity
- manifest envelope
- layout
- atomic writes
- validation
- registry
- retention
- promotion

Domain packages retain ownership of artifact meaning.

## Market data

Target:

market_data/
  contracts/
  schemas/
  sources/
  datasets/
  storage/
  loading/
  cleaning/
  resampling/
  alignment/
  sessions/
  splits/
  cache/
  commands/

Split market_data/loading/ticks.py by source parsing, tick-book representation, coverage validation, and execution lookup if those remain independently changeable.

The major future-proofing rule for data is: **logical dataset identity must not equal physical file layout**.

Callers should request a `DatasetRef`/`MarketDataset` with explicit symbol, time range, event schema,
timezone/session semantics and provenance. A source adapter decides whether those bytes came from CSV,
Parquet, an API, a database or a stream.

Similarly, downstream code should not assume the canonical market unit is always an M1 bar. Separate:

- event schema (bar/tick/book/event);
- storage representation;
- logical time/index semantics;
- transforms/resampling;
- dataset partition identity.

Then adding order-book research or a different vendor does not force changes in training/evaluation code.

## Features

Target:

features/
  contracts/
  schema/
  technical/
  context/
  po3/
  stages/
  pipeline/

features/pipeline/builder.py becomes a stage composer. Stage modules own their algorithms. The pipeline owns stage order, schema assembly and cache orchestration only.

Split features/po3/config.py by reason to change:

- fvg.py
- ifvg.py
- entry_trigger.py
- mtf.py
- state.py
- zones.py
- assembly.py

Do not split purely by LOC. Keep cohesive algorithms together.

`features/` should own deterministic or explicitly fitted feature transformations, not the assumption that
every model consumes handcrafted features. The boundary should support three cases:

1. handcrafted feature matrix;
2. raw/near-raw market windows with no handcrafted feature stage;
3. learned representation produced by a model/learning backend.

Use an `InputView`/`FeatureView` schema so training/evaluation can consume an input contract without knowing
whether the columns came from indicators, PO3, embeddings or raw market values.

If a transform has learned state (for example a fitted scaler, PCA, learned tokenizer or feature selector),
persist that state as a versioned artifact. Do not hide fit state inside a global feature builder.

Feature cache identity must be based on semantic inputs and transform identity, not module/file names.

## Trading

Target:

trading/
  contracts/
  account/
  risk/
  execution/
  orders/
    geometry.py
    stops.py
    targets.py
    selection.py
    placement.py
    provenance.py
  positions/
  policy/
  strategies/
  baselines/

Split trading/orders/entries.py so stop generation, target generation, selection, provenance and final placement are independently understandable.

Move geometry_summary to evaluation/reporting if it is analysis-only.

Trading should itself be decomposable by replaceable policy:

- account state/accounting;
- risk policy;
- position sizing;
- order intent construction;
- execution-cost assumptions;
- strategy semantics;
- position lifecycle.

Do not let FTMO constraints become account semantics. FTMO is one `RiskPolicy`/constraint set. A future
portfolio-risk or broker-margin policy should be swappable without rewriting broker/execution or decision code.

Likewise, structural SL/TP is one `TargetPlan` implementation. A future learned exit policy, market-only
policy, limit ladder or rebalance target should still produce a semantic order/execution intent.

## Simulation

Target:

simulation/
  contracts/
  environment/
    core.py
    factory.py
    initialization/
    step/
    execution/
    position/
    risk/
    rewards/
    observations/
    episode/
    diagnostics/
  backtest/

Retire simulation/environment/runtime_dependencies.py as an import bucket. Each service imports only its real dependencies.

Split initialization_runtime.py into market source, account policy, execution config, risk config, reward construction, strategy construction, observation storage, action/observation spaces and episode settings.

Split step_runtime.py into action decoding, constraints, action execution, accounting, entry diagnostics, info construction and termination.

Split execution_runtime.py into fill policy, tick quote resolution, stop execution, target execution, multi-TP execution, discretionary exit and close accounting.

TradingEnv.step should become a composition function over these services.

Simulation must own **state transition and replay mechanics**, not the identity of the learner. Keep separate:

- market replay clock/source;
- execution/fill model;
- account/position state transition;
- risk/constraint application;
- observation/view construction;
- episode/termination semantics;
- optional Gym adapter.

Gymnasium should become an adapter for RL, not the fundamental simulation API. That makes the same simulator
usable by rule-based backtests, supervised-model evaluation, Monte Carlo stress tests and live-shadow replay.

Backtest should consume the same trading/execution contracts where possible, but the repository does not need
one universal engine. What must stay stable is the request/result and execution semantics needed for parity.

## Models

Target:

models/
  contracts/
    encoder.py
    predictor.py
  encoders/
    common.py
    account.py
    tcn.py
    lite_tcn.py
    gru.py
    transformer.py
    mtf.py
  policies/
  auxiliary/
  factory/
    architecture.py
    predictor.py
  runtime/
    precision.py
  compatibility/
    model_state.py

Split models/encoders/core.py by encoder family. Shared sequence-mask and account-embedding helpers stay in common modules. Legacy implementations move to compatibility.

`models/` should own reusable neural components and inference-capable predictor architectures only.
It should not own PPO rollout construction, SAC replay setup, Gym vectorization, or training callbacks.

Split the current models/factory/agent.py responsibilities across the new boundaries:

- encoder/model architecture selection -> `models/factory/`;
- PPO/SAC algorithm construction -> `learning/rl/`;
- vector-environment construction -> `learning/rl/environment_adapter.py`;
- action/observation schema validation -> stable contracts / `decisioning` adapters;
- legacy PPO/action-layout loading -> `decisioning/adapters/sb3.py` plus compatibility;
- precision/autocast -> `models/runtime/precision.py`.

This is intentionally a larger conceptual split than simply making `agent.py` smaller.
It prevents `models/` from remaining an RL package under a generic name.

## Decisioning

`decisioning/` is the stable runtime seam between learned systems and trading/simulation/live consumers.

It owns:

- backend-neutral inference protocol;
- semantic decision output;
- schema validation at load/inference boundaries;
- adapters from SB3/Torch/rule-based implementations;
- provider registry/loading;
- inference metadata and compatibility checks.

It does not own:

- training loops;
- reward computation;
- supervised labels/losses;
- trading execution;
- feature engineering.

A future DL migration should mostly add `learning/supervised/...` plus a Torch decision adapter,
while leaving simulation, trading, evaluation and live consumers on the same `DecisionProvider` interface.

## Learning

`learning/` owns how parameters are fitted.

Current RL code should migrate here over time rather than remain spread across `models/`,
`training/callbacks/`, `simulation/`, and SB3-specific utilities.

The RL backend may depend on simulation contracts because environment interaction is part of RL.
The supervised backend should depend on dataset/feature contracts instead and should not require Gym.

This gives two clean paths:

RL path:
features + trading/simulation contracts -> learning/rl -> model artifact -> decisioning adapter

DL path:
features + labels/datasets -> learning/supervised -> model artifact -> decisioning adapter

Both converge at the same runtime decision contract.

## Training

Training is an application/workflow layer. Broad imports are acceptable only in a few composition roots.
It chooses a learning backend; it does not implement PPO or supervised optimization itself.

Target:

training/
  contracts/
  use_cases/
    prepare_run.py
    fit_seed.py
    evaluate_seed.py
    train_multi_seed.py
    walkforward.py
  setup/
    data.py
    learner.py
    decision_provider.py
  artifacts/
  reporting/

RL-specific callbacks should migrate under `learning/rl/` when they are genuinely tied to SB3/rollouts.
Generic run progress, artifact persistence and experiment-level monitoring may remain under training/reporting.

Split training/runner/seed_runtime.py into typed stages:

- PreparedSeedRun
- TrainedSeedRun
- EvaluatedSeedRun
- PersistedSeedRun

The final seed workflow should read as prepare, fit through selected learner, build/load a decision provider, evaluate, persist.

Shrink SeedRunContext. A context with dozens of fields is a sign that one object carries the whole application state.

Split training/runner/setup_runtime.py into config resolution, dataset/feature preparation, artifact/run identity, and runtime component construction.

Split training/callbacks/core.py into cohesive callback modules.

## Evaluation

Evaluation must be independent of training implementation.

Target:

evaluation/
  contracts/
  checkpoint/
    resolve.py
    load.py
    compatibility.py
    validate.py
  use_cases/
    evaluate_provider.py
    reanalyze.py
    compare_runs.py
  rollout/
  metrics/
  reporting/
  provenance/
  plotting/

Split evaluation/reanalysis/runner.py into request validation, checkpoint resolution/loading, compatibility, dataset reconstruction, feature reconstruction, environment setup, schema validation, execution and artifact writing.

Replace PPO-specific evaluation assumptions with a provider-oriented boundary:

- resolve a persisted decision artifact;
- load it through the appropriate `decisioning` adapter;
- validate input/output schema;
- execute the provider against a simulation/backtest/live-compatible evaluation source;
- compute metrics and persist provenance.

Historical PPO checkpoints remain supported by the SB3 adapter. Future DL checkpoints use a Torch adapter
without changing the evaluation workflow.

Split decision_diversity.py into thresholds, reference classification, statistics, priors and report formatting if these continue to change independently.

Replace catch-all plotting modules with chart-family modules such as equity, drawdown, pnl, distributions, trades, orders, comparisons and PO3.

Evaluation should not assume that every candidate is a trading RL policy. Split evaluation into:

- execution/economic evaluation (equity, drawdown, survival, fills, turnover);
- predictive evaluation (classification/regression/calibration when applicable);
- behavioral/decision diagnostics;
- robustness/statistical analysis;
- reporting/rendering.

An `EvaluationCase` declares what is being evaluated and which metric suites apply. A non-trading forecasting
experiment should not need fake trade records merely to use the evaluation package.

Metric computation should produce typed/tabular metric data first; plotting/report rendering consumes those
results later. This keeps a future renderer/dashboard change out of scientific metric code.

## Experiments

Target:

experiments/
  variants/
  campaigns/
    contracts.py
    state.py
    controller.py
    scheduler.py
    workers.py
    resources.py
    checkpoints.py
    source_snapshot.py
    artifact_validation.py
    resume.py
    reporting.py
  provenance/

experiments/campaigns/execution.py should be split after scientific runtime boundaries are stable. Its current independent responsibilities include state records, input validation, resource probing, CPU masks, Slurm host expansion, worker commands, source snapshots, artifact verification, checkpoint resume, initial checkpoint validation and controller lifecycle.

## Live and integrations

Keep live broker-neutral. Keep MT5/platform behavior in integrations.

Live should reuse the same feature API, trading contracts and `DecisionProvider` API used by offline execution.
It should not care whether the artifact came from PPO, SAC, supervised DL, imitation learning, or a rule-based provider.

Use broker/venue ports so live runtime owns scheduling/state/recovery while integrations own MT5-specific API
calls and data conversions. Replacing MT5 should require a new adapter plus contract tests, not edits throughout
strategy, risk, decisioning and model packages.

## Compute

Compute owns device and scheduler capabilities. It does not know specific encoder/model classes.

Expose neutral capabilities such as `ResourceProfile`, `DeviceCapabilities`, `JobSpec` and `ExecutionBackend`.
Slurm is one adapter. Model/learning code may ask for capabilities (CUDA, bf16, memory) but should not parse Slurm,
hostnames or cluster environment variables directly.

## Orchestra

Lower scientific priority. Split orchestra/pipeline.py into phase engine, invocation, routing, sessions, verification and individual phase implementations. Keep pipeline.py as the state machine/composition root.

## Test architecture

Target:

tests/
  unit/
  contracts/
  integration/
  regression/
  architecture/
  research/

Rules:

- leaf implementation -> focused unit test;
- public boundary -> contract test;
- cross-domain workflow -> integration test;
- scientific invariant -> regression/golden test.

Split giant tests along the same change surfaces as production. A developer should not read a 1,400-line test module to learn one TP-mode rule.

Add **replaceability/architecture scenario tests**. They need not train expensive models; they prove boundaries:

- load the same tiny logical dataset through two source/storage adapters and get the same dataset contract;
- run a pipeline with handcrafted features and a pipeline that bypasses handcrafted features;
- run the same `DecisionProvider` evaluation with a rule provider and an SB3/Torch test provider;
- run the same order/risk contract with two risk-policy implementations;
- run simulation through its native API and through the Gym adapter with equivalent transitions;
- render one metric dataset through two renderers without recomputing scientific metrics;
- persist/read one artifact through filesystem storage and a test in-memory/object-store adapter;
- resolve compute capabilities without importing any model package.

These tests are stronger proof of future-proofing than class/interface presence alone.

## Developer and agent knowledge layer

Every major domain should have a README and CHANGE_MAP, either colocated or generated under .agents/domain_maps.

README contains:

- ownership;
- non-ownership;
- public API;
- dependency direction;
- critical contracts;
- invariants;
- composition roots;
- common tests.

CHANGE_MAP contains one row/section per common task:

- task;
- start file;
- required contract;
- focused test;
- integration/regression check;
- allowed neighboring domain;
- files normally not required;
- invariant that must remain true.

Add a path-validation test so maps cannot silently become stale.

Each CHANGE_MAP should also state the **axis of change**. Example:

```text
Task: add a new market-data vendor
Axis: source adapter
Stable contracts: DatasetRef, MarketEventSchema
Expected new code: market_data/sources/<vendor>.py
Must not change: feature algorithms, training workflows, evaluation metrics

Task: add a new risk regime
Axis: risk policy
Stable contracts: RiskState, RiskDecision
Expected new code: trading/risk/<policy>.py
Must not change: model architecture, broker adapter, market-data loader
```

This teaches both humans and agents whether a change is an implementation addition, contract change, or true
cross-cutting redesign.

## Architecture CI

Extend tests/architecture/test_dependency_rules.py.

Hard rules after migration:

1. production never imports scripts;
2. no legacy package imports;
3. no features -> trading;
4. no trading -> simulation.environment;
5. no simulation -> models;
6. no evaluation -> training;
7. no compute -> models;
8. leaf modules may not import workflows;
9. cross-domain private imports are forbidden;
10. only composition roots may exceed the domain fan-out budget.
11. models may not import Gym/SB3 learning implementations;
12. decisioning may not import training workflows;
13. generic evaluation/live code may not import `learning.rl` or `learning.supervised` internals;
14. supervised-learning modules may not require simulation/Gym unless an explicitly documented task needs it;
15. RL-specific dependencies (SB3, rollout buffers, Gym adapters) remain under `learning/rl` or compatibility adapters.

Soft diagnostics initially:

- production file >500 LOC;
- >15 top-level definitions;
- >8 internal imports;
- >3 top-level domains imported;
- very long function;
- very large state/context dataclass.

LOC is only a review trigger, not an automatic defect.

## Scientific baseline before refactoring

Freeze:

- dataset identities and hashes;
- split boundaries;
- causal feature hashes;
- feature column order;
- normalization fit scope;
- HTF alignment behavior;
- feature-cache identity;
- action schema manifest;
- observation schema manifest;
- stop/target candidate menus;
- multi-TP fractions and selections;
- SL/TP fill and tie policy;
- guardrail state transitions;
- account/equity accounting;
- golden environment traces;
- tick/bar execution parity;
- checkpoint loadability;
- legacy checkpoint migration;
- seed/RNG initialization;
- callback activation;
- deterministic reanalysis;
- frozen metric fixtures;
- run/campaign manifest identities.

Architecture PRs preserve these. Scientific behavior changes are separate PRs.

## Migration waves

### Wave 0 — Freeze behavior and inventory

Create behavior baselines, current dependency inventory and an explicit temporary violations allowlist.

### Wave 1 — Break package cycles

Remove evaluation -> training, trading -> environment runtime, features -> trading, simulation -> models, and compute -> models.

Exit: scientific package graph is acyclic and enforced.

### Wave 2 — Contracts and public APIs

Stabilize dataset, feature schema, order/position/execution, action/observation, decision-provider,
predictor artifact, learner/fit-result, training result, evaluation result and campaign-state contracts.

Create the decisioning/learning seams before moving large amounts of RL implementation code.

### Wave 3 — Training and evaluation workflows

Refactor seed_runtime.py, setup_runtime.py and reanalysis/runner.py against backend-neutral learner/provider contracts.
Separate generic workflow monitoring from RL-specific callbacks.

### Wave 4 — Trading and simulation execution

Refactor order construction and the environment initialization/step/execution/risk/position services.

### Wave 5 — Features

Refactor feature pipeline stages and PO3 subdomain ownership.

### Wave 6 — Models, decisioning and learning backends

Split encoder families and precision ownership. Move PPO/SAC construction, rollout/vector-env logic and
RL-specific callbacks into `learning/rl`. Introduce `decisioning` adapters for current SB3 artifacts.

Exit: adding a supervised DL backend does not require edits to simulation, trading, evaluation or live runtime.

### Wave 7 — Evaluation metrics and plotting

Split decision-diversity and plotting hubs.

### Wave 8 — Campaigns

Split campaigns/execution.py into controller and focused services.

### Wave 9 — Orchestra and compute cleanup

Refactor engineering infrastructure after scientific runtime stabilizes.

### Wave 10 — Test tree and compatibility cleanup

Split oversized tests, remove temporary re-exports, remove resolved allowlist entries, delete obsolete compatibility code only after historical checkpoints are safe, and regenerate domain maps.

## PR sequence

Recommended reviewable sequence:

1. target DAG plus architecture tests;
2. evaluation environment-builder ownership;
3. action/TP decoding ownership;
4. feature reachability ownership;
5. model precision ownership;
6. decision-provider contracts + SB3 adapter around current behavior;
7. learner/fit-result contracts + RL backend boundary;
8. move vector-env/PPO/SAC construction out of generic models;
9. seed workflow stage types;
10. seed prepare/fit/evaluate/persist split;
11. reanalysis artifact/provider/data/execution split;
12. split generic monitoring from RL-specific callbacks;
13. order geometry/stops/targets split;
14. environment initialization services;
15. step services;
16. execution/fill services;
17. feature pipeline stages;
18. PO3 subdomain split;
19. encoder-family/model factory split;
20. evaluation metric/plotting split;
21. campaign execution split;
22. optional supervised-DL reference backend proving replaceability;
23. test-tree and compatibility cleanup.

The supervised-DL reference backend should be small and only added after the contracts are stable.
Its purpose is architectural verification: prove that one can fit a non-RL predictor, wrap it as a
`DecisionProvider`, and run the same evaluation/live-facing path without importing SB3 or the RL environment adapter.

## Definition of done

Navigation:

- repository structure predicts where behavior lives;
- each domain has accurate ownership/change maps.

Dependency:

- scientific package graph is acyclic;
- leaf modules do not import workflows;
- private cross-domain imports fail CI.

Locality:

- normal production change radius <=2–3 modules;
- normal domain radius =1;
- boundary contract radius <=1–2;
- focused test radius <=1–2.

File responsibility:

- large files are cohesive or explicit composition roots;
- no generic runtime/core/builder file owns unrelated semantics;
- compatibility is isolated.

Scientific safety:

- frozen hashes, traces, execution/risk behavior, checkpoints, metrics and provenance remain unchanged through architecture-only PRs.

Public API:

- supported callers import package APIs/contracts;
- internal layout can be reorganized without repository-wide edits.

Learning-paradigm replaceability:

- `models/` contains no mandatory PPO/SAC/Gym dependency;
- generic training workflows operate on `Learner`, not PPO/SAC classes;
- evaluation/live operate on `DecisionProvider`, not SB3 models;
- RL-specific code is isolated under `learning/rl` plus the SB3 compatibility adapter;
- a supervised Torch model can be introduced without modifying trading/simulation semantics;
- model artifacts declare model family, learning paradigm, input schema and decision-output schema.

Capability replaceability:

- a new market-data source/storage backend does not change feature/training/evaluation semantics;
- bar/tick/order-book representations can coexist behind explicit event/input schemas;
- handcrafted features can be bypassed without rewriting training/evaluation infrastructure;
- a new risk policy does not require changes to broker/execution or model code;
- a new execution/fill model does not require changes to strategy/decision-provider internals;
- a new simulator/backtest implementation can consume the same trading contracts;
- a new broker integration stays inside an adapter boundary;
- a new metric/report renderer does not alter scientific metric computation;
- a new compute backend does not require model/scientific imports;
- artifact storage can change without changing artifact semantic identity/provenance;
- at least several replaceability scenario tests prove these statements with tiny fixtures.

Unknown-change resilience:

- adding a genuinely new concept has an explicit classification process: implementation, policy, adapter,
  capability, cross-cutting mechanic, or versioned contract change;
- no whole-repository context/config/schema object must be understood to add one local capability;
- framework/SDK/storage types do not leak across unrelated domains;
- new bounded contexts can be introduced without reorganizing existing package internals.

## Original implementation sequence

This was the implementation sequence for the refactor. It is retained as a
record of the intended scope and is completed by the implementation record at
the end of this document.

1. freeze all current scientific parity/provenance gates;
2. create the repository-wide change-axis/volatility map and assign one owner to every semantic concept;
3. define and test the capability-level package DAG plus composition-root allowlist;
4. remove the five currently observed reverse dependencies/cycles;
5. define the first stable cross-domain contracts: dataset/event, input/feature schema, decision,
   risk, order/execution, artifact/provenance, and resource capability;
6. add small replaceability scenario tests for data source, feature bypass, risk policy, simulator/Gym adapter,
   decision provider, artifact storage and compute capability;
7. create CHANGE_MAP files for the main scientific/infrastructure domains, including axis-of-change metadata;
8. refactor training/runner/seed_runtime.py and evaluation/reanalysis/runner.py around typed stage results and
   backend-neutral contracts;
9. isolate PPO/SAC as one learning backend and wrap current behavior behind DecisionProvider/Learner contracts;
10. then continue the domain-by-domain splits in the migration waves, never mixing scientific behavior changes
    with architecture-only moves.

The architecture is complete when the filesystem and contracts answer:

- who owns this behavior;
- what may depend on it;
- what its stable input/output contract is;
- which file should be edited;
- which test proves it;
- which invariant must not change;
- which surrounding files do not need to be read.

## Completed in the context-local refactor

The current implementation supplies the backend-neutral learner and provider
contracts, SB3 and supervised Torch learner backends, provider-based training,
walk-forward, reanalysis, and live composition paths, plus the replaceability
scenario tests described above. Structural stops and targets have separate
owners. TP-mode tests are grouped by decoder, state-machine, execution, and
action-decoding change surfaces. Historical checkpoint import paths remain
available through explicit compatibility modules.

The soft file-size diagnostics remain review prompts: the remaining large
workflow modules are composition roots with typed stage boundaries, rather
than leaf modules that own unrelated scientific semantics.
