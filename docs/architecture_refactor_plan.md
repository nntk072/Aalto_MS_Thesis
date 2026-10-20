# Repository Architecture Refactor Plan

## Goal

The repository itself is the architecture boundary. The target is a set of first-class top-level Python packages with clear responsibilities, explicit public APIs, and predictable dependency direction.

The refactor is not complete when large files are merely split into smaller files. It is complete when a developer can look at the repository root, choose the correct package, open that package's `__init__.py`, and understand:

- what the package owns;
- which symbols are safe to import;
- which neighboring package owns the next concern;
- which dependency directions are allowed;
- where tests for that package live.

All importable code packages and subpackages must contain `__init__.py`. Asset, evidence, generated-output, and command-only directories are not Python packages and should not receive placeholder `__init__.py` files.

The sequence is:

**repository inventory → package boundaries → public contracts → characterization tests → incremental migration → compatibility verification → removal of legacy layout**

Behavioral equivalence is the primary constraint. Architecture work must not silently alter feature causality, execution, risk, rewards, training, checkpoint compatibility, evaluation results, or campaign reproducibility.

## Repository categories

The root should clearly separate six kinds of content.

### Domain and application packages

These are importable Python packages with `__init__.py` and stable public APIs:

- `configuration/`
- `artifacts/`
- `market_data/`
- `features/`
- `trading/`
- `simulation/`
- `models/`
- `training/`
- `evaluation/`
- `experiments/`
- `live/`
- `integrations/`
- `compute/`
- `orchestra/`

### Configuration assets

`config/` contains YAML, CSV schedules, experiment declarations, and campaign configuration. It remains an asset directory rather than a Python package.

Python code that loads, validates, merges, hashes, and resolves shared configuration belongs to the top-level `configuration/` package. Domain packages own only domain-specific interpretation/validation layered on the resolved configuration.

### Data and generated artifacts

- `data/` stores raw and prepared market data.
- `cache/` stores disposable caches.
- `outputs/` stores run outputs and generated evidence.
- `logs/` stores runtime logs.
- `baseline/` stores baseline evidence and audit records.

These are not Python packages.

### Executable wrappers

`scripts/` contains command wrappers only. Reusable Python logic must live in one of the top-level packages.

A script should normally do only this:

1. parse command-line arguments;
2. call a package API;
3. convert failures to exit codes;
4. print or persist the result.

Tests and production packages should not import reusable logic from `scripts/`.

### Engineering and CI tooling

- `tools/` contains developer command wrappers.
- `ci-scripts/` contains CI entry scripts.
- `.github/` contains hosted CI configuration.
- `.agents/` contains agent guidance and architecture/domain maps.

These are repository tooling, not trading-domain packages.

### Documentation and tests

- `docs/` documents architecture, research protocol, operations, and evidence.
- `tests/` mirrors the top-level package architecture.
- package-level import and dependency tests live under `tests/architecture/`.

## Repository-root ownership matrix

The repository architecture includes code, declarative inputs, immutable evidence, disposable generated state, human documentation, and execution wrappers. A directory being non-importable does not make it architecture-free.

| Root area              | Architectural role                                                         | Owning logic                                                          | Write policy                                                                 | Source-of-truth rule                                        |
| ---------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `config/`            | Declarative experiment/runtime inputs                                      | `configuration` plus domain validators                              | Human-authored or generated only through explicit config-generation commands | Versioned config plus resolved snapshot determine a run     |
| `data/`              | Market-data assets and curated partitions                                  | `market_data`                                                       | Data ingestion/preparation APIs only; scripts wrap them                      | Raw source identity and dataset manifest are authoritative  |
| `cache/`             | Disposable acceleration artifacts                                          | Owning producer package plus generic artifact primitives              | Atomic, identity-keyed, safe to rebuild                                      | Never the only copy of research evidence                    |
| `outputs/`           | Immutable run, campaign, evaluation, CI, and research artifacts            | `artifacts` plus the package that defines each artifact's semantics | Structured run/artifact writers                                              | Manifest and provenance identify every result               |
| `logs/`              | Local/global runtime diagnostics that are not already colocated with a run | Runtime owner; shared conventions from `artifacts`                  | Append/rotate; no domain state stored only in logs                           | Logs support diagnosis, not scientific truth                |
| `baseline/`          | Curated, reviewed reference evidence                                       | Evaluation/verification code plus promotion workflow                  | Promote from validated outputs; do not write casually during runs            | Baseline manifest points to source artifacts and validation |
| `scripts/`           | Human/Slurm command entry points                                           | Owning top-level package                                              | Thin wrappers only                                                           | No reusable domain logic                                    |
| `tests/`             | Verification architecture                                                  | Mirrors package and lifecycle ownership                               | Test fixtures/results use isolated temp paths                                | Tests specify contracts and invariants                      |
| `docs/`              | Human architecture/research/operations knowledge                           | Relevant package/process owner                                        | Human-authored docs plus explicitly generated reports                        | Docs cite artifact IDs/manifests for numerical claims       |
| `tools/`             | Developer workstation helpers                                              | Engineering/toolchain layer                                           | Thin wrappers around standard tools/package APIs                             | No experiment or trading semantics                          |
| `ci-scripts/`        | CI command adapters                                                        | Engineering/toolchain layer                                           | Thin, deterministic wrappers                                                 | CI behavior remains inspectable from workflow + script      |
| `.github/workflows/` | Hosted CI orchestration                                                    | Engineering/toolchain layer                                           | Declarative workflow definitions                                             | Calls `ci-scripts/` or standard package commands          |

This ownership matrix is part of the target architecture. Refactoring is incomplete if Python packages are clean while generated data, reports, caches, and commands still write arbitrary files with undocumented identities.

## Configuration architecture

`config/` is a declarative asset tree. Parsing, merging, defaults, schema validation, compatibility rules, override handling, and resolved-config hashing belong to a new low-level `configuration/` package.

Target asset organization:

```text
config/
├── base/                 # canonical runtime defaults and shared fragments
├── features/             # feature-family profiles
├── policies/             # trading-policy profiles when useful
├── experiments/          # experiment/variant declarations
├── campaigns/            # concrete campaign specifications
├── data/                 # symbols, sessions, source/partition declarations
├── models/               # model/architecture presets if they remain declarative
└── schemas/              # optional machine-readable config schemas
```

Migration should preserve existing config paths until callers move. Existing root YAML files may remain compatibility assets during the transition.

`configuration/` owns:

- deterministic merge order;
- unknown-key rejection and typo safety;
- explicit schema/version identifiers;
- path resolution relative to repository/runtime roots;
- environment/CLI override normalization;
- canonical serialization for hashing;
- resolved configuration snapshots stored with every run;
- compatibility migration when a config schema changes.

Training, evaluation, experiments, live runtime, and compute code consume a resolved configuration object. They must not each implement their own YAML merge semantics.

A resolved run configuration is immutable after run creation. Reanalysis loads the saved snapshot first and may compare it with current source configuration, but it must never silently reinterpret a historical run using current defaults.

## Data architecture

`data/` is storage, while `market_data/` owns all logic that creates, validates, transforms, partitions, and identifies datasets.

The current tree contains full-period exports plus `data/training/`, `data/testing/`, examples, inferred session data, and derived arrays. The target separates lifecycle state explicitly:

```text
data/
├── raw/                  # source exports copied/linked without semantic transformation
├── curated/              # cleaned/aligned canonical market datasets
├── partitions/           # immutable train/validation/test or walk-forward views/manifests
├── reference/            # schedules/levels/reference assets derived from approved sources
├── examples/             # tiny tracked developer fixtures
└── manifests/            # dataset identities, hashes, ranges, source lineage, schema versions
```

Large physical files may stay in their existing locations during migration; the important first step is to give them a typed dataset identity and a manifest rather than moving terabytes for cosmetic reasons.

Every dataset-producing operation records at least:

- dataset ID and schema version;
- source file identities/hashes;
- symbol and timeframe;
- timezone/session semantics;
- inclusive/exclusive timestamp bounds;
- cleaning/resampling/alignment version;
- row count and key quality statistics;
- partition definition and leakage-sensitive boundaries;
- producer code/source revision when relevant.

Training and evaluation should accept a dataset identity or resolved dataset object rather than reconstructing meaning from a filename.

## Cache architecture

`cache/` is disposable and reproducible. It may improve runtime but must never become hidden scientific state.

Target namespaces should follow the producer and artifact type, for example:

```text
cache/
├── market_data/          # bar/tick normalization and lookup caches
├── features/             # feature matrices and higher-timeframe auxiliaries
├── simulation/           # shared environment bundles / memory-mapped arrays
├── evaluation/           # safe-to-recompute derived evaluation intermediates
└── manifests/            # optional cache inventory/index, never required for correctness
```

A cache key must be derived from all semantic inputs that can change its contents: source-data identity, relevant resolved config, schema/version, feature code/version or explicit algorithm version, partition identity, and any execution/session semantics that affect values.

Cache writers must use temporary files/directories followed by atomic promotion where practical. Partial caches are invalid. A cache entry must be validated before reuse. A stale or missing cache causes regeneration, not behavior change.

Feature caches, tick caches, and shared environment bundles keep their domain-specific identity logic in `features`, `market_data`, and `simulation`; generic path/manifest/atomic-write primitives may come from `artifacts`.

## Generated-artifact architecture

Generation logic is part of the architecture. A generated file is not adequately designed merely because its destination directory is documented.

Introduce a low-level `artifacts/` package for cross-cutting artifact mechanics:

```text
artifacts/
├── __init__.py
├── identity.py           # ArtifactId, content/source identity helpers
├── layout.py             # canonical run/campaign/evaluation path layouts
├── manifest.py           # common provenance envelope and schema/version fields
├── atomic.py             # atomic JSON/text/binary/directory promotion helpers
├── registry.py           # artifact-kind metadata and validation dispatch
├── retention.py          # disposable/retained/promotable classifications
└── promotion.py          # validated output -> curated baseline/evidence transitions
```

Exact filenames may adjust, but these responsibilities need one owner.

`artifacts` owns storage/provenance mechanics, not domain content. For example, `training.artifacts` defines what a training manifest means, `evaluation.reporting` defines evaluation tables, and `experiments.campaigns` defines campaign state. They call common artifact primitives instead of independently inventing directories and JSON-write patterns.

Every generated artifact type must define:

1. producer package/API;
2. consumer package/API;
3. artifact kind and schema version;
4. semantic identity inputs;
5. canonical location/layout;
6. overwrite/immutability policy;
7. atomicity/completion marker policy;
8. validation before consumption;
9. retention class: disposable, run-retained, evidence-retained, or baseline-promotable;
10. provenance links to parent artifacts;
11. regeneration command/API;
12. compatibility policy for older artifact versions.

Direct ad-hoc `mkdir` + `write_text`/CSV/JSON/Parquet output remains acceptable only inside the owning writer implementation. Callers should request an artifact from an API rather than assemble path conventions themselves.

### Artifact lifecycle

Use an explicit lifecycle instead of treating generated files as anonymous filesystem state:

```text
DECLARED INPUT
    ↓ resolve + validate
RESOLVED CONFIG / DATASET IDENTITY
    ↓ generate
STAGING ARTIFACT
    ↓ validate + atomically promote
COMPLETE IMMUTABLE ARTIFACT
    ↓ consume / derive
DERIVED ARTIFACT
    ↓ verification gate
CURATED EVIDENCE / BASELINE
    ↓ cite
REPORT / THESIS / DEFENSE OUTPUT
```

Failed or interrupted generation leaves a staging/incomplete state that consumers reject. Completion must be represented structurally by an atomic promotion or explicit completion manifest, not inferred from “file exists”.

### Provenance graph

Artifacts form a directed provenance graph. A model checkpoint should identify the resolved config, dataset/partition, feature schema/cache identity, action/observation schema, seed, source revision, and parent checkpoint if resumed. Evaluation artifacts identify the checkpoint and evaluation dataset. Campaign summaries identify their child runs. Baselines identify the validated source artifacts from which they were promoted.

This graph must be traversable without reading free-form logs.

## Outputs architecture

`outputs/` is the materialized artifact store for experiment execution. It should be organized by artifact semantics rather than by whichever script happened to create a folder.

Target logical namespaces:

```text
outputs/
├── runs/                 # individual training runs
├── campaigns/            # campaign controllers, child-run refs, scheduler evidence
├── evaluation/           # checkpoint reanalysis and OOS evaluation packs
├── research/             # research audits and one-off validated analyses
├── ci/                   # CI/gate evidence intended to be retained locally
├── reports/              # generated report/figure packs
└── agent/                # agent execution state only when intentionally retained
```

Existing names such as campaign, CI-gate, data-refresh, thesis-validation, and agent-state trees can migrate gradually. Do not rename active campaign paths in place if that would break reproducibility.

A canonical training run layout should include a top-level manifest and typed children, for example:

```text
outputs/runs/<run-id>/
├── manifest.json
├── config/
│   ├── resolved.yaml
│   └── identity.json
├── model/
├── training/
├── evaluation/
├── diagnostics/
├── logs/
└── COMPLETE
```

Campaign and evaluation layouts should likewise have explicit manifests instead of relying on filename conventions.

Run IDs are immutable identities. A resumed run either records lineage to its parent checkpoint/run or uses a clearly defined continuation identity; it must not silently overwrite the provenance of the original run.

## Logs architecture

Logs have two scopes:

- run-scoped logs belong inside the corresponding run/campaign/evaluation artifact tree;
- root `logs/` is reserved for local/global runtime services that have no containing artifact, such as local live/paper runtime logging.

Structured event/state needed for metrics or reproducibility belongs in typed JSON/CSV/Parquet artifacts, not only in text logs. Text logs may be rotated or archived without changing scientific results.

All logging setup should accept a run/artifact context carrying run ID, campaign ID, seed, worker/job identity, and relevant source/config identity. This avoids detached log files whose experiment cannot be reconstructed.

## Baseline architecture

`baseline/` is curated evidence, not another output directory.

Baseline creation is a promotion workflow:

```text
validated output artifact
→ comparison/acceptance gate
→ baseline manifest
→ curated baseline files
```

A baseline manifest records source artifact IDs/paths, source revision, config/data/schema identities, validation commands, acceptance result, creation time, and the specific metrics/traces being frozen.

Normal training/evaluation runs never write directly into `baseline/`. A dedicated promotion API/command performs the copy/link plus manifest creation after validation. Updating a baseline is an explicit research decision visible in review.

## Scripts architecture

The existing script tree is large enough to require its own architecture. Script categories should map to the packages they invoke rather than becoming alternate implementations.

Target command groups:

```text
scripts/
├── data/                 # ingestion/preparation/cache commands -> market_data/features
├── train/                # training commands -> training
├── eval/                 # evaluation/report commands -> evaluation
├── experiments/          # campaign/variant commands -> experiments
├── compute/              # Slurm/allocation/runtime wrappers -> compute
├── research/             # reproducible analysis entry points -> package APIs
├── maintenance/          # migrations/cleanup/repair commands
└── tests/                # convenience launchers only; canonical tests live in tests/
```

Shell scripts remain appropriate for Slurm allocation, environment activation, and process supervision, but business logic, campaign state transitions, data rules, and artifact schemas move into Python packages.

A script may choose arguments and destinations; it may not define a second version of the underlying algorithm.

## Tests architecture

Tests mirror both code ownership and lifecycle contracts.

In addition to package-oriented tests, add:

```text
tests/
├── architecture/         # imports, dependency rules, root ownership rules
├── contracts/            # config/data/artifact schema compatibility
├── fixtures/             # tiny deterministic tracked data/config/artifact fixtures
├── lifecycle/            # cache invalidation, atomic completion, provenance, promotion
└── ...                   # package-mirrored tests already defined below
```

Tests that generate files use `tmp_path` or an isolated test artifact root. They must not mutate repository `data/`, `cache/`, `outputs/`, `logs/`, or `baseline/` during ordinary CI.

Golden artifacts are intentionally small, versioned fixtures with an owning test and schema version. Large historical outputs are not test fixtures.

## Documentation architecture

`docs/` should distinguish stable architecture, decisions, operating procedures, research protocol, training guidance, and thesis evidence.

Target logical groups:

```text
docs/
├── architecture/         # package + repository lifecycle architecture
├── decisions/            # ADR-style durable design decisions
├── operations/           # Triton, deployment, environment, recovery runbooks
├── training/             # training/evaluation/campaign procedures
├── research/             # experiment protocol, reproducibility, methodology
├── thesis/               # thesis-specific evidence interpretation and handoffs
└── generated/            # optional reproducible generated docs, if truly needed
```

Generated numerical reports should normally live in `outputs/` and be linked/cited from docs. If a generated document is committed under `docs/generated/`, its generator, source artifact IDs, and regeneration command must be recorded.

Documentation may describe an artifact; it must not become the only location of machine-readable evidence.

## Tools and CI architecture

`tools/` is the canonical developer/CI engineering command surface. It owns repository checks such as `check`, `test`, `build`, `clean`, bootstrap helpers, and graph/dependency inspection. These wrappers may compose standard tools and package/test commands but must not encode experiment semantics.

`.github/workflows/` defines hosted-runner scheduling, permissions, caching, concurrency, timeouts, and artifact upload. It invokes the same named gates available locally rather than redefining Ruff, mypy, pytest, architecture checks, or build semantics inline.

`ci-scripts/` is transitional. The current directory duplicates commands already expressible through `tools/` and GitHub Actions. During Phase 9, migrate all generic wrappers to `tools/check` or direct package commands. Keep a CI-only script only when it performs irreducibly platform-specific glue; if no such script remains, delete `ci-scripts/` entirely.

Target dependency direction:

```text
.github/workflows ───────┐
local developer shell ──┼──> tools/check + stable console commands
Docker/nightly runner ──┘                 │
                                          ├──> tests / build tools
                                          └──> public package APIs

optional platform-only ci-scripts ───────> tools/check
```

Production packages must never import `scripts/`, `tools/`, workflow code, or CI adapters. CI-generated evidence uses the same artifact conventions as local gates where retention matters; ephemeral runner logs may remain CI-platform output.

## Command and CI execution architecture

Scripts, console commands, Docker, Slurm, developer tools, and GitHub Actions are one execution architecture. They may select arguments, resources, environment variables, scheduling, and output locations, but they must converge on the same package APIs. None of these outer layers may contain a second implementation of feature logic, trading rules, risk equations, training setup, evaluation semantics, campaign state transitions, provenance rules, or artifact schemas.

Preferred direction:

    domain/application packages
        -> stable public APIs and package-owned CLI entry functions
        -> stable console commands
        -> scripts and developer-tool wrappers
        -> Docker / Slurm / GitHub Actions orchestration

### Stable commands and script lifecycle

Frequently used commands such as training, evaluation/reanalysis, campaign execution/status, and data preparation should have stable console entry points backed by package-owned CLI functions. Internal module moves must not force users or Slurm launchers to learn new deep module paths.

Keep scripts for research-specific, maintenance, and platform-launch work when a permanent console command adds no value.

Target script groups:

    scripts/
      data/
      train/
      eval/
      experiments/
      compute/slurm/
      compute/local/
      research/
      maintenance/
      tests/

The current matrix and Triton groups should be migrated by responsibility. Campaign behavior belongs to experiments. Slurm/resource behavior belongs to compute. Shell files that remain should be thin platform wrappers.

Every migrated script is classified as stable wrapper, temporary compatibility wrapper, frozen research record, or obsolete. Before removing or renaming one, search campaign specifications, Docker Compose, workflows, tests, docs, run manifests, and saved launch commands.

A Python script is too large for the scripts layer when it owns reusable state machines, dataclasses, checkpoint selection, result aggregation, hashing/provenance, dataset rules, artifact schemas, or substantial reusable helpers. A shell script is too large when it owns scientific calculations, JSON state mutation, retry-state decisions, campaign reconciliation, or long business-logic branches.

### Current CI audit

The current CI has useful coverage, but its command architecture is duplicated:

- GitHub workflows, ci-scripts, tools, Make targets, Docker Compose, and shell pipelines encode overlapping lint/type/test/build commands.
- Most ci-scripts are not called by the GitHub workflows, so local and hosted definitions can drift.
- the ordinary CI workflow executes the full test suite through several environment paths, including uv/venv and Docker, with substantial duplicate coverage;
- local tools/check and hosted type checking currently select different source scopes;
- documentation checks cover only a small root-level subset, and one formatting path can silently skip when its tool is absent;
- some Docker helper checks return success when Docker is unavailable, which is fine for optional convenience but invalid for a required gate;
- Docker Compose still contains legacy deep-module commands and historical script paths that must migrate with the package architecture;
- ordinary test containers mount repository data/cache/output/log roots even though normal CI should use isolated temporary roots;
- unit, integration, and slow markers are not yet a complete scheduling contract;
- GitHub CPU runners cannot validate Triton/GH200/Slurm behavior;
- release/image publication should remain only if a real runtime or reproducibility consumer needs it.

These are explicit refactor targets.

### One canonical gate vocabulary

Define each validation property once and invoke it everywhere through the same named gate. Recommended names:

    format
    lint
    types
    architecture
    contracts
    unit
    integration
    parity
    build
    docker-smoke
    training-smoke
    gpu
    triton
    full

The preferred local surface is tools/check with a gate argument. GitHub Actions should invoke the same gate semantics. ci-scripts should contain only hosted-runner/platform glue that cannot reasonably live in tools. If no such glue remains, remove ci-scripts rather than preserving duplicate wrappers.

A required gate may not silently pass because its required tool or capability is missing.

### Validation tiers

Tier A: fast developer and pull-request correctness.

- dependency/lock consistency;
- Ruff formatting and lint;
- mypy over every first-class package;
- architecture/public-import/dependency rules;
- config, artifact, lifecycle, and repository-hygiene contracts;
- fast unit tests;
- wheel/package build and import smoke.

Tier A must not depend on external market data, Docker, GPUs, or Triton.

Tier B: deterministic CPU integration and parity.

- integration tests using controlled tracked fixtures;
- environment/backtest execution parity;
- action/observation/checkpoint compatibility;
- data/feature causality and partition contracts practical on CPU;
- historical config/artifact read compatibility;
- selected command/script smoke tests in isolated temporary roots.

Do not run the identical full test collection repeatedly unless each execution validates a distinct environment contract.

Tier C: scheduled exhaustive validation.

- slow tests;
- larger deterministic integration/backtest suites;
- practical seed reproducibility checks;
- larger reanalysis/fixture checks;
- Docker build/runtime smoke if Docker remains supported.

Nightly is not a dumping ground for flaky tests.

Tier D: Triton/GPU/research-infrastructure validation.

Use Triton/GH200 when changes touch GPU execution, vectorized environments, shared bundles, training orchestration, checkpoint serialization, Slurm/campaign infrastructure, or resource-sensitive behavior. Include the applicable current-source provenance check, GPU import/device smoke, direct/bundle equivalence, short PPO/SAC/recurrent smoke training, checkpoint save/load/reanalysis, worker/resource geometry, campaign dry-run/recovery, and one short training-to-evaluation-to-manifest path.

Tier D results are structured gate artifacts. Long jobs use the existing external watcher rather than repeated agent polling.

### Test scheduling contract

Normalize markers so package location describes ownership and markers describe execution requirements:

- unit: deterministic, isolated, fast;
- integration: crosses package or I/O boundaries using controlled fixtures;
- slow: expensive but platform-neutral;
- gpu: requires CUDA-capable execution;
- triton: requires Triton/Slurm/filesystem assumptions;
- research: validates a larger research/evidence protocol.

A test may have more than one marker. Unmarked tests should become a deliberate small exception rather than an accidental majority.

### Local/hosted parity and CI isolation

A developer must be able to reproduce a hosted failure with the same repository command, for example:

    ./tools/check fast
    ./tools/check unit
    ./tools/check integration
    ./tools/check full

GitHub workflow YAML should mainly provide runner setup, caching, permissions, concurrency, timeouts, and artifact upload. It should not independently redefine test/lint/type semantics.

Every ordinary CI job receives isolated temporary data, cache, output, log, and artifact roots. CI must not consume whatever happens to exist in a developer or research checkout. Tests needing data use small tracked fixtures or explicitly provisioned versioned test datasets.

### Workflow responsibilities

ci.yml owns required pull-request gates: Tier A plus Tier B, implemented as 9 semantic validation jobs plus one `ci-success` aggregator as defined in the Target normal GitHub CI graph.

nightly.yml owns Tier C.

GPU/Triton validation is a documented Tier D command or dispatch flow and must not be represented as validated by a CPU runner.

release.yml remains only if container/package publication is an intentional supported deliverable. For this research repository, if there is no deployment/reproducibility consumer, replace registry publication with build/install/reproducibility checks.

Use least-privilege workflow permissions, concurrency cancellation for superseded pull-request commits, explicit timeouts, lockfile-keyed dependency caches, and BuildKit/GitHub cache primitives for Docker rather than caching the Docker daemon directory as an ordinary file tree.

### Docker, Make, tools, and ci-scripts

Docker is an execution environment, not a second application architecture. Compose services invoke stable package/console commands. Separate runtime/dev/test/GPU images only when each validates a real supported property.

The Makefile may own dependency export/lock maintenance and concise aliases. It must not become another independent quality-gate implementation.

tools owns human-friendly repository engineering commands and the canonical gate surface.

ci-scripts contains only hosted-platform glue. Wrappers that merely repeat raw Ruff, mypy, or pytest commands should delegate to the canonical gate or be removed.

### CI acceptance criteria

CI architecture is complete when:

1. each required gate has one canonical definition and owner;
2. local and hosted executions share gate semantics;
3. redundant full-suite executions are removed unless they test distinct environment contracts;
4. every first-class root package is formatted, linted, typed, packaged, and import-tested as applicable;
5. architecture/config/artifact/lifecycle rules are continuously enforced;
6. CPU CI uses isolated data/artifact roots;
7. Triton/GPU behavior has an explicit Tier D gate;
8. Docker and release automation contain only current supported commands;
9. no required gate silently skips due to missing capability;
10. the final architecture migration passes Tier A, Tier B, applicable Tier D smoke validation, and the full repository gate before the single final commit.

## Repository state and retention policy

Every root area needs an explicit version-control/retention policy:

- source packages, config assets, scripts, tests, docs, tools, CI definitions, and curated baseline manifests are version controlled;
- raw/large data is normally external or ignored, with tracked manifests/examples where useful;
- caches are ignored and disposable;
- normal outputs/logs are ignored, with selected evidence promoted to a curated tracked location only through an explicit process;
- virtual environments and tool caches are ignored;
- generated files committed to source control require a declared generator and regeneration check.

The `.gitignore` rules are therefore part of repository architecture and should be tested/reviewed alongside directory ownership. They must not accidentally ignore importable source packages or tracked examples.

## Shared-code and duplication architecture

The repository should deliberately reuse common behavior, but it should not create a catch-all `utils/`, `common.py`, or `helpers.py` dumping ground. Shared functions need semantic ownership just like larger modules.

Use this promotion ladder:

```text
private helper in one module
        ↓ reused by sibling modules
subpackage-owned helper/module
        ↓ reused across subpackages in one domain
package public API
        ↓ truly domain-neutral and reused across several top-level packages
small foundational package with a precise responsibility
```

Do not promote a helper only because two functions have the same name or similar-looking code. Consolidation is correct only when inputs, outputs, error behavior, determinism, units, numerical tolerances, side effects, and compatibility requirements are the same.

### Shared primitive ownership

The current repository already contains several repeated helper patterns. Their target ownership should be explicit:

| Repeated concern                                    | Canonical owner                                                                                    | Examples of consumers                                    |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| file/content SHA-256, artifact fingerprints         | `artifacts.identity`                                                                             | campaigns, thesis evidence, evaluation manifests, audits |
| atomic JSON/text/binary writes and safe replacement | `artifacts.atomic`                                                                               | training, evaluation, experiments, research generators   |
| artifact layout/path construction                   | `artifacts.layout`                                                                               | runs, campaigns, evaluation packs, CI evidence           |
| canonical manifest/provenance envelope              | `artifacts.manifest`                                                                             | training/evaluation/experiments/baseline promotion       |
| canonical config serialization/hash                 | `configuration.hashing`                                                                          | training, campaigns, evaluation reanalysis               |
| repository/config path resolution                   | `configuration.paths` when config-related; owning package otherwise                              | CLI/config/data resolution                               |
| CUDA/device and host-resource discovery             | `compute.environment`                                                                            | training setup, benchmarks, Slurm execution              |
| position sizing and risk capacity                   | `trading.risk`                                                                                   | simulation, evaluation, live/MT5 adapters                |
| action/observation schema conversion                | `simulation.contracts`                                                                           | models, training, evaluation                             |
| feature schema/cache identity                       | `features.pipeline`                                                                              | training, evaluation, cache builders                     |
| dataset hashing/partition identity                  | `market_data`                                                                                    | training, evaluation, feature cache identity             |
| plot/report semantic transforms                     | `evaluation.plotting.transforms` / `evaluation.reporting`                                      | static and interactive renderers                         |
| terminal/dashboard formatting                       | keep close to reporting/dashboard owners unless several unrelated packages need the exact renderer | training dashboard, evaluation diagnostics               |

The existing generic utility package should disappear as responsibilities migrate. For example, device/resource logic belongs to `compute.environment`, while terminal box-table rendering belongs with the reporting/dashboard layer that consumes it unless broader reuse proves necessary.

### No generic utility package by default

Do not add a top-level `utils/` merely to make imports shorter. A generic utility package tends to accumulate unrelated behavior, obscure ownership, and create dependency cycles.

A new foundational package is justified only when all of these are true:

1. the behavior is domain-neutral;
2. at least three independent top-level packages need it, or two packages need it and neither can reasonably own it;
3. the API can be described by one precise responsibility;
4. it has no upward dependency on trading, simulation, training, evaluation, experiments, or scripts;
5. moving it does not merge subtly different semantics;
6. tests can prove compatibility for every migrated caller.

`configuration` and `artifacts` already satisfy this role for most repository-wide concerns. Prefer extending those focused foundations over inventing `utils`.

### Duplication-removal workflow

For each candidate duplication:

1. locate all implementations and call sites;
2. compare signatures, accepted types, units, defaults, exceptions, side effects, ordering, determinism, and numerical behavior;
3. classify the behavior by semantic owner;
4. add characterization tests for each existing implementation when behavior is not already protected;
5. implement one canonical function behind the owning package API;
6. migrate callers one group at a time;
7. keep a compatibility shim only when historical imports or checkpoints require it;
8. run result-equivalence tests on representative real inputs;
9. delete the duplicate implementation after all callers migrate;
10. add an architecture test or lint rule when the duplicated pattern is important enough to prevent from returning.

### Result-equivalence requirement

When replacing several implementations with one shared function, "same result" includes more than the return value.

For pure deterministic functions, equivalence should cover:

- exact equality for discrete/state/schema/path outputs;
- bitwise equality where practical for hashes/serialization;
- configured `rtol`/`atol` for floating-point calculations;
- stable ordering for dictionaries, columns, events, and manifests where ordering is part of identity;
- identical exception types for invalid input when callers depend on them.

For stateful or I/O helpers, also verify:

- files produced and their schema;
- atomic/completion behavior;
- overwrite rules;
- side effects and logging contracts;
- deterministic provenance/identity fields;
- failure and recovery behavior.

A shared implementation must not silently change research results merely because its API looks cleaner.

### Duplicate-function inventory as an implementation aid

Before moving each package, the implementation agent should run an AST/search-based duplicate inventory for that package and its callers. The inventory is a review aid, not an automatic refactoring list. Generic names such as `main`, `reset`, `forward`, or `to_dict` usually represent different contracts and should not be merged automatically.

High-value current candidates include repeated SHA-256 helpers, JSON readers/writers, artifact writers, path/layout construction, resource/device resolution, position-sizing logic, and semantic plot transforms. Strategy methods with the same name should instead be consolidated through explicit interfaces/base classes only where their contracts are intentionally identical.

## Semantic naming and function-shape architecture

Architecture refactoring must improve names and function boundaries as well as file placement. A repository can remain difficult to understand after a directory split if modules, classes, functions, variables, phases, and generated artifacts keep vague or numbered names.

Names must communicate domain intent. Avoid placeholders such as `setting1`, `setting2`, `function1`, `function2`, `part1`, `part2`, `step1`, `step2`, `process_data`, `handle_item`, `do_work`, `misc`, or `common` when a more specific concept exists.

Numbering is acceptable only when the number is itself part of the domain or protocol, for example `tp1`, `tp2`, `tp3`, `M1`, `H1`, a documented migration phase, schema version `v3`, or `seed42`. It must not substitute for a missing concept name.

### Naming rules

Use names that state the action and the domain object. Prefer `approve_trade_risk()` over `process_risk()`, `compute_daily_loss_headroom()` over `calculate_value2()`, `apply_partial_take_profit()` over `handle_tp_part2()`, `resolve_repeated_payout_transition()` over `process_cycle()`, `build_observation_vector()` over `prepare_data()`, and `write_campaign_manifest()` over `save_json()` when campaign semantics matter.

### Function-boundary rules

Do not preserve a large function by splitting it into meaningless numbered fragments such as `_step_part1()`, `_step_part2()`, or `_process_section_3()`. Each extracted function must represent one independently understandable responsibility with a meaningful input/output contract.

For orchestration code, decomposition should follow domain stages. For example, the simulation step flow should decompose around concepts such as `advance_market()`, `update_open_position()`, `apply_exit_rules()`, `evaluate_account_constraints()`, `resolve_payout_transition()`, `decode_policy_action()`, `evaluate_entry_intent()`, `execute_accepted_order()`, `update_reward_and_account_state()`, `build_observation_and_info()`, and `resolve_episode_end()`.

These exact names may adjust to the final implementation, but the extracted responsibilities must remain semantic rather than ordinal.

### Similar ideas require one vocabulary

When several modules describe the same concept, choose one canonical term and use it consistently across code, configuration, logs, metrics, tests, and documentation. Important examples include `requested_risk` / `approved_risk` / `executed_risk`, `daily_loss_headroom` versus `daily_loss_breach`, `account_floor` / `operating_equity` / `realized_balance` / `payout`, `dataset` / `partition` / `feature_schema` / `cache_identity`, `run` / `attempt` / `checkpoint` / `campaign` / `evaluation`, `terminated` versus `truncated`, and `entry_intent` / `order` / `fill` / `position` / `exit`.

Do not keep parallel synonyms such as `job/run/trial` or `risk_cap/risk_budget/risk_limit` unless they intentionally represent different concepts and the distinction is documented.

### Rename when ownership changes

Moving code to a new package is an opportunity to correct misleading names. If an old name reflects its former container instead of its actual responsibility, rename it during migration and keep a compatibility alias only when external callers, historical imports, checkpoints, serialized metadata, or persisted artifact fields require one.

Renames must be behavior-preserving. Update imports, config keys where migration is intentional, tests, documentation, and artifact schemas together. Do not silently rename persisted fields used by historical analysis.

### Public API naming

Public APIs exposed through `__init__.py` must use stable domain names. Avoid temporary migration names such as `new_*`, `legacy2_*`, `refactor_*`, `tmp_*`, or arbitrary `v2_*` names unless versioning is part of the supported contract.

### Naming audit during refactor

For every moved responsibility, review the module/package name, public class/function names, key state variables and dataclass fields, config keys and schema fields, metric/log/event names, generated artifact filenames and manifest fields, test names/fixtures, and documentation terminology.

The goal is a consistent conceptual model that lets a developer follow one idea through configuration, runtime code, artifacts, tests, and reports without translating between arbitrary aliases.

## Current structural problems

The issue is broader than file length, but size exposes areas where responsibilities have accumulated.

| Priority | Current area                                  | Approx. size | Architectural issue                                                                                                                                                               |
| -------- | --------------------------------------------- | -----------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0       | legacy `envs/trading_env.py`                |  5,369 lines | One environment class owns action decoding, entry, execution, exits, risk, account lifecycle, guardrails, sessions, rewards, observations, payout cycles, logging, and debug flow |
| P0       | legacy `train/train_rl.py`                  |  2,905 lines | One entry point owns config resolution, data setup, features, environment/model factories, callbacks, training, evaluation, manifests, and aggregation                            |
| P0       | legacy `features/build.py`                  |  1,551 lines | Feature facade and many feature families, alignment rules, normalization, schema, and cache identity are mixed                                                                    |
| P1       | `experiments/campaigns/execution.py`     |  1,873 lines | Reusable campaign state, checkpoint recovery, validation, workers, scheduler concerns, and CLI handling are mixed inside a script                                                 |
| P1       | legacy `eval/plots.py`                      |  1,263 lines | Data shaping and renderer-specific plotting are mixed                                                                                                                             |
| P1       | `scripts/research/thesis/evidence/verify_numerical_report.py` |  1,239 lines | Multiple report-validation domains and CLI orchestration are combined                                                                                                             |
| P1       | legacy `eval/plots_interactive.py`          |  1,106 lines | Duplicates semantic plot preparation and mixes it with Plotly rendering                                                                                                           |
| P1       | `orchestra/pipeline.py`                     |  1,088 lines | Pipeline phase definition, execution, recovery, and orchestration live in one class                                                                                               |
| P1       | legacy `eval/eval_run.py`                   |    977 lines | Checkpoint loading, provenance, data reconstruction, schema validation, rollout, metrics, and reporting are mixed                                                                 |
| P1       | legacy `backtest/entry_levels.py`           |    973 lines | Entry menus, stop/target candidate construction, repair, and order creation are coupled                                                                                           |
| P1       | legacy `backtest/engine.py`                 |    967 lines | Event loop, execution, accounting, entries, exits, and result assembly are concentrated in one function                                                                           |
| P2       | legacy `features/po3_config.py`             |    991 lines | Detection, configuration, zone construction, and plotting helpers share one module                                                                                                |
| P2       | legacy `models/agent.py`                    |    644 lines | Model factory, algorithm selection, and compatibility validation are coupled                                                                                                      |
| P2       | legacy `data/ticks.py`                      |    645 lines | Tick storage, quote lookup, construction, and execution-facing helpers are approaching a useful split                                                                             |

These hotspots are migration inputs. The target architecture is defined by repository-level package ownership below.

## Target repository layout

```text
Aalto_MS_Thesis/
├── configuration/
│   ├── __init__.py
│   ├── loader.py
│   ├── merge.py
│   ├── schema.py
│   ├── hashing.py
│   ├── paths.py
│   └── compatibility.py
│
├── artifacts/
│   ├── __init__.py
│   ├── identity.py
│   ├── layout.py
│   ├── manifest.py
│   ├── atomic.py
│   ├── registry.py
│   ├── retention.py
│   └── promotion.py
│
├── market_data/
│   ├── __init__.py
│   ├── loading/
│   │   ├── __init__.py
│   │   ├── bars.py
│   │   ├── ticks.py
│   │   └── datasets.py
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── align.py
│   │   ├── resample.py
│   │   ├── cleaning.py
│   │   └── sessions.py
│   ├── splits/
│   │   ├── __init__.py
│   │   ├── definitions.py
│   │   └── walkforward.py
│   └── cache/
│       ├── __init__.py
│       ├── identity.py
│       └── storage.py
│
├── features/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── context.py
│   │   ├── normalization.py
│   │   ├── schema.py
│   │   └── cache.py
│   ├── technical/
│   │   ├── __init__.py
│   │   ├── indicators.py
│   │   ├── swings.py
│   │   └── structure.py
│   ├── po3/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── fvg.py
│   │   ├── ifvg.py
│   │   └── entries.py
│   └── context/
│       ├── __init__.py
│       ├── liquidity.py
│       ├── session_levels.py
│       ├── premium_discount.py
│       └── policy_market.py
│
├── trading/
│   ├── __init__.py
│   ├── account/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── accounting.py
│   │   └── limits.py
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── legacy.py
│   │   └── repeated_payout.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── budget.py
│   │   ├── sizing.py
│   │   └── execution_reserve.py
│   ├── orders/
│   │   ├── __init__.py
│   │   ├── intents.py
│   │   ├── entries.py
│   │   ├── stops.py
│   │   └── targets.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── costs.py
│   │   ├── quotes.py
│   │   ├── fills.py
│   │   └── latency.py
│   ├── positions/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── exits.py
│   │   ├── multi_tp.py
│   │   └── trailing.py
│   └── strategies/
│       ├── __init__.py
│       ├── base.py
│       ├── baseline.py
│       ├── po3_ifvg.py
│       └── distribution.py
│
├── simulation/
│   ├── __init__.py
│   ├── contracts/
│   │   ├── __init__.py
│   │   ├── actions.py
│   │   ├── observations.py
│   │   └── results.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── env.py
│   │   ├── runtime.py
│   │   ├── reset.py
│   │   ├── step.py
│   │   ├── observations.py
│   │   ├── rewards.py
│   │   ├── sessions.py
│   │   └── diagnostics.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── entries.py
│   │   ├── exits.py
│   │   ├── accounting.py
│   │   └── results.py
│   └── parity/
│       ├── __init__.py
│       ├── traces.py
│       └── assertions.py
│
├── models/
│   ├── __init__.py
│   ├── factory/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── checkpoints.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── tcn.py
│   │   ├── gru.py
│   │   ├── transformer.py
│   │   └── mtf.py
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── ppo.py
│   │   ├── recurrent.py
│   │   └── sac.py
│   └── auxiliary/
│       ├── __init__.py
│       ├── vae.py
│       └── heads.py
│
├── training/
│   ├── __init__.py
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── resolution.py
│   │   ├── strategy.py
│   │   └── policy.py
│   ├── setup/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── environments.py
│   │   ├── models.py
│   │   └── callbacks.py
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   └── training.py
│   ├── callbacks/
│   │   ├── __init__.py
│   │   ├── diagnostics.py
│   │   ├── risk.py
│   │   └── checkpoints.py
│   └── artifacts/
│       ├── __init__.py
│       ├── manifests.py
│       ├── metrics.py
│       └── aggregation.py
│
├── evaluation/
│   ├── __init__.py
│   ├── rollout/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   └── episode.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── performance.py
│   │   ├── distributions.py
│   │   ├── calibration.py
│   │   ├── bootstrap.py
│   │   └── temporal.py
│   ├── reanalysis/
│   │   ├── __init__.py
│   │   ├── inputs.py
│   │   ├── provenance.py
│   │   ├── data_rebuild.py
│   │   ├── schemas.py
│   │   └── runner.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── export.py
│   │   ├── events.py
│   │   ├── diagnostics.py
│   │   └── manifests.py
│   └── plotting/
│       ├── __init__.py
│       ├── transforms.py
│       ├── equity.py
│       ├── orders.py
│       ├── signals.py
│       ├── static.py
│       └── interactive.py
│
├── experiments/
│   ├── __init__.py
│   ├── variants/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── resolution.py
│   ├── campaigns/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── state.py
│   │   ├── validation.py
│   │   ├── checkpoints.py
│   │   ├── workers.py
│   │   ├── execution.py
│   │   └── reporting.py
│   └── provenance/
│       ├── __init__.py
│       ├── source.py
│       └── artifacts.py
│
├── live/
│   ├── __init__.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── strategy.py
│   │   ├── risk.py
│   │   └── execution.py
│   └── paper/
│       ├── __init__.py
│       └── logger.py
│
├── integrations/
│   ├── __init__.py
│   └── mt5/
│       ├── __init__.py
│       ├── connection.py
│       ├── data.py
│       ├── trader.py
│       ├── risk.py
│       └── robot.py
│
├── compute/
│   ├── __init__.py
│   ├── slurm/
│   │   ├── __init__.py
│   │   ├── resources.py
│   │   ├── jobs.py
│   │   ├── sessions.py
│   │   └── campaign.py
│   └── environment/
│       ├── __init__.py
│       ├── runtime.py
│       └── identity.py
│
├── orchestra/
│   ├── __init__.py
│   ├── phases/
│   │   ├── __init__.py
│   │   ├── definitions.py
│   │   └── runner.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── invocations.py
│   │   └── subprocesses.py
│   ├── recovery/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── resume.py
│   └── reporting/
│       ├── __init__.py
│       ├── health.py
│       └── status.py
│
├── config/                      # YAML/CSV assets, not a package
├── data/                        # market-data assets, not a package
├── cache/                       # generated cache, not a package
├── outputs/                     # generated experiment outputs, not a package
├── logs/                        # generated logs, not a package
├── baseline/                    # baseline evidence, not a package
├── scripts/                     # research, maintenance, and platform wrappers only
├── tests/
├── docs/
├── tools/
├── ci-scripts/                  # transitional; delete if no platform-only glue remains
└── pyproject.toml
```

The exact module filenames can adjust during migration when source-level dependencies reveal a cleaner boundary. The top-level package responsibilities and dependency direction should remain stable.

<!-- EXECUTION_BLUEPRINT_START -->
## Target execution architecture layouts

The target repository layout above describes ownership of code and state. The diagrams below describe **how humans, CI, Docker, Slurm/Triton, scripts, and package APIs execute that architecture**.

### Target scripts layout

Stable product/research operations should not require users to know deep Python module paths. Frequent supported operations move to package-owned CLI functions exposed through `pyproject.toml`. The steady-state `scripts/` tree is intentionally smaller:

```text
scripts/
├── compute/
│   └── slurm/                 # allocation, tmux/srun, login-node-safe wrappers only
├── research/
│   ├── audits/                # reproducible research/protocol audits
│   ├── benchmarks/            # performance/resource experiments
│   ├── campaigns/             # research-only campaign analyses, not controllers
│   ├── strategy_checks/       # isolated thesis/strategy diagnostics
│   ├── thesis/
│   │   ├── evaluation/        # thesis-specific rescoring/figures
│   │   └── evidence/          # ledgers, indexes, report verification
│   └── frozen-launches/       # only launchers required to replay named historical evidence
├── maintenance/
│   ├── data-and-cache/        # explicit rebuild/repair operations
│   ├── artifacts/             # artifact migration/repair/promotion support
│   └── migrations/            # one-time schema/layout migrations with expiry criteria
└── compatibility/             # temporary forwarding wrappers; empty and removed in Phase 11
```

Steady state deliberately has no generic `matrix/`, `utility/`, `docker/`, `tests/`, `setup/`, or reusable `train/`/`eval/` implementation trees. Their reusable behavior moves to packages, engineering behavior moves to `tools/`, platform behavior moves under `scripts/compute/slurm/`, and declarative variant lists move to `config/`.

### Stable console-command surface

The exact executable prefix may be adjusted once during Phase 1, but the plan requires stable commands with these responsibilities:

```text
thesis-data       -> market_data.cli       # discover, prepare, validate, identify datasets
thesis-train      -> training.cli          # train/resume/benchmark supported training modes
thesis-evaluate   -> evaluation.cli        # evaluate, reanalyze, report, verify result packages
thesis-campaign   -> experiments.cli       # run, resume, status, summarize campaigns
thesis-compute    -> compute.cli           # platform/resource inspection; no trading semantics
```

A command may have subcommands. The important contract is that Docker, Slurm wrappers, tests, docs, and users call these stable surfaces rather than deep implementation modules. Historical/frozen launchers may keep their original command text as evidence, but current launchers migrate.

### Current normal GitHub CI graph

The current `.github/workflows/ci.yml` contains **15 jobs**. It mixes semantic checks with environment mechanisms and runs substantially overlapping test/build paths.

```text
PUSH / PULL REQUEST
│
├─ build-dev-image ─────────────> docker-release-build ─────> ut-docker
│
├─ check-docker-tools                         (independent/orphan verification)
│
├─ check-venv                                 (independent environment verification)
│
├─ check-python-requirements ─┬──────────────┬──────────────┬──────────────┐
│                             │              │              │              │
└─ check-uv ──────────────────┼─> check-docs │              │              │
                              ├─> code-formatting           │              │
                              ├─> uv-build ────────────────┬┘              │
                              ├─> venv-build ──────────────┼──> ut-venv    │
                              ├─> json-check               │               │
                              └─> markdown-formatting      └──> ut-uv      │
```

Current job inventory:

1. `build-dev-image`
2. `check-docker-tools`
3. `check-python-requirements`
4. `check-uv`
5. `check-venv`
6. `check-docs`
7. `code-formatting`
8. `json-check`
9. `markdown-formatting`
10. `docker-release-build`
11. `uv-build`
12. `venv-build`
13. `ut-venv`
14. `ut-docker`
15. `ut-uv`

The graph has no final semantic aggregator and no explicit architecture/contracts/parity node. `ut-uv`, `ut-venv`, and `ut-docker` largely prove the same test property through different mechanisms; that duplication is only justified if those environments are intentionally supported compatibility contracts.

### Target normal GitHub CI graph

Normal `ci.yml` should contain **9 validation jobs plus one final aggregator**. Each validation node proves a repository property; tools/environments are implementation details inside that node.

```text
                           PUSH / PULL REQUEST
                                   │
                                   ▼
                           dependency-lock
                                   │
           ┌───────────────────────┼────────────────────────┐
           │           │           │           │            │
           ▼           ▼           ▼           ▼            ▼
        quality   architecture  contracts      unit      build-smoke
   format/lint/       imports/    config/       fast       wheel +
      types         boundaries    artifact      tests      install/import
           │           │           │           │            │
           │           └───────────┼───────────┘            │
           │                       │                        │
           │              ┌────────┴────────┐               │
           │              ▼                 ▼               │
           │         integration          parity            │
           │              │                 │               │
           │              └────────┬────────┘               │
           │                       ▼                        │
           │                training-smoke <────────────────┘
           │                       │
           └───────────────────────┼─────────────────────────
                                   ▼
                              ci-success
```

Target job contracts:

| Job | Required property | Canonical local gate | Must not require |
|---|---|---|---|
| `dependency-lock` | lockfile/pin/project metadata consistency | `tools/check deps` | external data, Docker, GPU |
| `quality` | Ruff format + Ruff lint + mypy across every first-class package | `tools/check quality` | Docker/GPU |
| `architecture` | public imports, dependency direction, naming, script/entry-point ownership | `tools/check architecture` | research data |
| `contracts` | config/artifact/cache/lifecycle/schema/repository-hygiene contracts | `tools/check contracts` | mutable repository outputs |
| `unit` | deterministic isolated unit tests | `tools/check unit` | network, external datasets, GPU |
| `build-smoke` | wheel build, clean install/import, console-entry discovery | `tools/check build` | Docker/GPU |
| `integration` | deterministic cross-package/I/O behavior on fixtures | `tools/check integration` | mutable research checkout state |
| `parity` | execution/data/schema equivalence contracts | `tools/check parity` | GPU unless explicitly marked |
| `training-smoke` | tiny CPU training -> checkpoint -> evaluation/artifact roundtrip | `tools/check training-smoke` | thesis-scale data/runtime |
| `ci-success` | required-job aggregation for branch protection | no separate logic | new checks of its own |

`quality`, `architecture`, `contracts`, `unit`, and `build-smoke` may run in parallel after dependency consistency. `integration` and `parity` begin only after the structural/unit prerequisites they rely on. `training-smoke` is the final behavioral CPU gate. Branch protection targets only `ci-success`, while that aggregator depends on every required semantic gate.

### Current-to-target CI migration

| Current job(s) | Target | Decision |
|---|---|---|
| `check-python-requirements`, `check-uv`, relevant `check-venv` behavior | `dependency-lock` + `build-smoke` | Collapse environment setup into deterministic lock/build properties. |
| `code-formatting` | `quality` | Keep Ruff + mypy semantics; expand mypy to all first-class packages. |
| `check-docs`, `json-check`, `markdown-formatting` | `contracts` or dedicated repository checks inside it | Validate all owned docs/config schemas; required format tools may not silently skip. |
| `uv-build`, `venv-build` | `build-smoke` | Build once, install into a clean environment, import/test console entry points. |
| `ut-uv`, `ut-venv` | `unit` | One canonical CPU unit suite. Keep a second environment only if it is a declared support contract. |
| `build-dev-image`, `docker-release-build`, `ut-docker`, `check-docker-tools` | Tier C `docker-smoke` | Docker proves image/runtime compatibility, not a second full unit suite. |
| missing today | `architecture` | Add import/dependency/naming/execution-surface rules. |
| missing today | `contracts` | Add configuration/artifact/cache/lifecycle/repository contracts. |
| missing today | `integration`, `parity`, `training-smoke` | Promote existing relevant tests into explicit semantic gates. |

### Validation tier graph

```text
source change
    │
    ├── Tier A: fast CPU / every PR
    │      dependency-lock -> quality + architecture + contracts + unit + build-smoke
    │
    ├── Tier B: deterministic CPU / normal CI
    │      integration + parity -> training-smoke
    │
    ├── Tier C: scheduled/manual exhaustive
    │      slow + research-scale deterministic checks + docker-smoke + larger reanalysis
    │
    └── Tier D: Triton/GPU/Slurm
           gpu -> short accelerator training -> checkpoint/evaluation roundtrip
               -> resource/vector-env checks -> campaign dry-run/resume/recovery
               -> structured Triton gate artifact
```

Tier D is not represented as passed by GitHub-hosted CPU runners. It is executed on Triton/GPU infrastructure through `tools/check gpu` / `tools/check triton` plus thin Slurm wrappers. A future GitHub workflow may dispatch it only if a secure, intentional runner/bridge is introduced; the architecture does not assume one.

### Target CI / engineering layout

```text
.github/
└── workflows/
    ├── ci.yml                  # Tier A + Tier B; 9 validation jobs + ci-success
    ├── nightly.yml             # Tier C slow/research/docker-smoke
    └── release.yml             # only if publishing remains an intentional deliverable

tools/
├── check                       # canonical named-gate dispatcher
├── test                        # targeted pytest convenience; delegates marker semantics
├── build                       # clean wheel/install/import verification
├── clean                       # safe generated-state cleanup
├── bootstrap                   # developer environment setup if still required
└── ...                         # graph/lint helper wrappers with one responsibility

ci-scripts/                     # transitional only
└── <platform-only glue>        # keep only if GitHub cannot express it cleanly; otherwise delete dir

pyproject.toml
├── project.scripts             # stable package-owned console commands
├── pytest markers              # unit/integration/slow/gpu/triton/research
├── Ruff scope                  # every first-class package + supported scripts/tests
└── mypy scope                  # every first-class package
```

The Makefile remains for lock export/maintenance and short aliases only. Docker Compose invokes stable commands. It does not own an independent test/training architecture.

### CLI/scripts/tools/Docker/Slurm/CI -> package API execution flow

```text
                         HUMAN / AUTOMATION
                                 │
         ┌──────────────┬────────┼──────────┬───────────────┐
         ▼              ▼        ▼          ▼               ▼
 stable console CLI   research  tools/check Docker       Slurm/Triton
                      scripts                 │               wrappers
         │              │        │           │               │
         └──────────────┴────────┴───────────┴───────────────┘
                                 │
                                 ▼
                        PUBLIC PACKAGE APIs
                                 │
      ┌──────────────┬───────────┼────────────┬──────────────┐
      ▼              ▼           ▼            ▼              ▼
 market_data/features trading simulation   training      evaluation
      │              │           │            │              │
      └──────────────┴───────────┴────────────┴──────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
                experiments                 live
                    │                         │
                    ▼                         ▼
                  compute                integrations
                    │
                    ▼
          canonical artifact lifecycle
```

GitHub Actions is an outer scheduler for `tools/check`; Docker is an environment; Slurm is a resource scheduler. None owns domain algorithms.

### Migration phase dependency/order graph

The numbered phases remain the review/acceptance order, but the dependency graph explains what can be prepared in parallel without violating ownership:

```text
Phase 0  freeze contracts / inventory
   │
   ▼
Phase 1  configuration + artifacts + root API foundations
   │
   ├──────────────> Phase 4  market_data + features ───────┐
   │                                                       │
   ▼                                                       │
Phase 2  trading ─────> Phase 3 simulation ────────────────┤
                                                           ▼
                                                    Phase 5 models + training
                                                           │
                                                           ▼
                                                    Phase 6 evaluation
                                                           │
                                                           ▼
                                                    Phase 7 experiments + compute
                                                           │
Phase 2 trading ───────────────┐                            │
Phase 5 models/training ───────┼──> Phase 8 live/integrations
                               │                            │
                               └──────────────┬─────────────┘
                                              ▼
                                Phase 9 Orchestra + execution surfaces
                                              │
                                              ▼
                                Phase 10 repository-state normalization
                                              │
                                              ▼
                                Phase 11 remove compatibility + final verify
```

No phase may delete a legacy path before all producers, consumers, saved launch surfaces, tests, and historical-artifact readers that depend on it have migrated or received an explicit compatibility adapter.

<!-- EXECUTION_BLUEPRINT_END -->

## Package ownership

### `configuration`

Owns shared configuration loading, schema/version handling, deterministic merge/override resolution, validation, canonical serialization/hashing, and historical compatibility. It consumes assets from `config/` but is itself importable code.

It must remain independent of training/evaluation orchestration so the same resolved configuration semantics are usable by training, evaluation, experiments, live, and tooling.

### `artifacts`

Owns cross-cutting generated-artifact identity, layouts, manifests, atomic writes/promotions, retention metadata, and provenance links.

It does not define training metrics, evaluation statistics, feature values, or campaign semantics; those remain with their domain packages.

### `market_data`

Owns bar/tick ingestion, market dataset identity, resampling/alignment, session labeling, split definitions, and bar/tick cache identity.

Does not own indicators, strategy semantics, risk, or model training.

### `features`

Owns causal feature construction, technical indicators, swings/structure, PO3/FVG/IFVG feature state, liquidity/session context, train-scoped normalization, feature schemas, and feature-cache identity.

Does not own order execution, trading account state, or model fitting.

### `trading`

Owns pure trading-domain rules and state: account state/accounting, selectable trading policies, risk budgets, sizing, orders, stops/targets, execution-cost/fill models, position state/management, and strategy semantics.

It must be usable by simulation, evaluation, and live paths without importing training code.

### `simulation`

Owns offline runtime mechanics: Gymnasium environment, action/observation/result contracts, reset/step sequencing, reward-input orchestration, sessions, standalone backtesting, and environment/backtest parity.

It depends on `trading`, `market_data`, and `features`.

### `models`

Owns model factories, encoders, PPO/SAC/recurrent policies, checkpoints, and auxiliary neural components.

### `training`

Owns model-fitting composition: configuration resolution, data/feature/environment/model setup, callbacks, one-seed training, checkpoint writing, manifests, and aggregation.

### `evaluation`

Is the single canonical evaluation package. It owns rollout, metrics, statistical analysis, checkpoint reanalysis, provenance, export, and plotting.

### `experiments`

Owns variant resolution, campaign configuration/state, checkpoint discovery/resume, validation, worker construction, provenance, and campaign reporting.

### `live`

Owns broker-neutral live/paper trading runtime.

### `integrations`

Owns external-system adapters. MT5-specific I/O, connections, data adapters, and order routing live under `integrations.mt5`.

### `compute`

Owns reusable compute-platform mechanics: Slurm resources/jobs/sessions, runtime environment identity, and campaign compute dispatch.

### `orchestra`

Remains a separate top-level package for coding-agent orchestration and does not import thesis trading semantics.

## Public import model

The repository should teach its own architecture through imports.

Typical imports should look like:

```python
from market_data import TickBook, load_market_data
from features import build_features
from trading import AccountState, TradingPolicy
from trading.risk import RiskBudget, approve_risk
from simulation import TradingEnv
from simulation.backtest import run_backtest
from models import build_agent, load_checkpoint
from training import run_training
from evaluation import calculate_metrics, evaluate_model
from experiments import resolve_variant
from experiments.campaigns import run_campaign
from integrations.mt5 import MT5Connection, MT5Trader
```

A caller should not need to know which implementation file contains a stable domain contract.

## `__init__.py` contract

Every importable package and subpackage must have an intentional `__init__.py`.

Example:

```python
# trading/risk/__init__.py
"""Risk approval and position-sizing contracts."""

from .budget import RiskBudget, approve_risk
from .sizing import PositionSize, compute_position_size

__all__ = [
    "PositionSize",
    "RiskBudget",
    "approve_risk",
    "compute_position_size",
]
```

Rules:

1. Include a one- or two-sentence ownership docstring.
2. Re-export only stable symbols intended for callers.
3. Define `__all__` when it clarifies the supported API.
4. Do not place implementation logic in `__init__.py`.
5. Do not trigger model loading, file I/O, data loading, logging setup, or runtime initialization from package import.
6. Internal modules may import sibling implementation modules directly.
7. Cross-package callers should prefer the owning package API.
8. Deep imports are allowed only for explicitly public specialized contracts.
9. Package import tests must verify the supported surface.

## Dependency direction

Preferred dependency flow:

`configuration` and `artifacts` are foundational infrastructure. Domain packages may depend on them, but they must not depend upward on training, evaluation, experiments, live, or scripts.

```text
configuration ──────────────────────────────────────────────┐
artifacts ──────────────────────────────────────────────────┤
                                                            │
market_data ──────┐
                  ├──> features
                  │
                  └──────────────┐
                                 v
trading ───────────────────> simulation
                                 │
simulation contracts ───────> models
                                 │
market_data/features/trading ─> training
                                 │
models/simulation/trading ────> evaluation
                                 │
training/evaluation ──────────> experiments
                                 │
trading/models ───────────────> live
trading/live ─────────────────> integrations

compute <──── experiments/scripts
orchestra     independent engineering toolchain
```

Hard rules:

- `configuration` imports no training/evaluation/experiment/live orchestration.
- `artifacts` imports no domain-specific training/evaluation/experiment implementations.
- `market_data` never imports `features`, `trading`, `training`, or `evaluation`.
- `features` never imports `training`, `evaluation`, or live/broker code.
- `trading` never imports `training`, `evaluation`, `experiments`, or scripts.
- `simulation` never imports `training`, `evaluation`, or scripts.
- `models` may depend on simulation contracts but not on training orchestration.
- `training` composes upstream packages but must not become the owner of domain equations.
- `evaluation` consumes trained models and simulation/trading contracts; it never performs model fitting.
- `experiments` orchestrates runs but contains no trading equations or feature calculations.
- `live` is broker-neutral.
- `integrations` adapts external systems and must not duplicate trading policy/risk equations.
- production packages never import from `scripts`.
- reusable code never lives only inside `scripts`.
- `orchestra` remains outside the research dependency graph.

## Trading environment decomposition

The current environment class is the most urgent migration because it combines almost the entire simulator.

### `simulation.environment`

Owns the Gymnasium facade, constructor wiring, reset, top-level step ordering, observation/reward coordination, and info/result assembly.

Target: a comprehensible facade/runtime, with no single orchestration function above roughly 250 lines unless there is a strong reason.

### `simulation.contracts`

Owns action schemas, observation schemas, result/event interfaces, and compatibility metadata shared with models/checkpoints.

### `trading.risk`

Extract sizing equity, requested/approved risk, daily/account capacity, execution reserve, lot sizing, and risk diagnostics.

### `trading.positions`

Extract SL/TP evaluation, multi-TP partial closes, breakeven, trailing stop, TP progression, and position exits.

### `trading.orders`

Extract structural entry candidates, stop/target menus, selection repair, and order intents.

### `trading.execution`

Own deterministic fill/cost semantics shared by environment, backtest, and live adapters where behavior is truly identical.

After extraction, environment step sequencing should read as explicit stages:

```text
advance market
→ update open position
→ apply exits
→ enforce account constraints
→ process payout transition
→ decode action
→ evaluate entry
→ execute accepted order
→ update reward/account
→ build observation/info
→ resolve termination/truncation
```

## Training decomposition

The training entry point becomes a thin composition root.

Move responsibilities into:

- `training.configuration` for training-specific option interpretation only; shared YAML loading/merging/schema/hash behavior belongs to root `configuration`;
- `training.setup.data`;
- `training.setup.environments`;
- `training.setup.models`;
- `training.setup.callbacks`;
- `training.runner`;
- `training.artifacts`.

The CLI wrapper under `scripts/train/` calls `training.run_training()`.

## Feature-pipeline decomposition

`features` exposes one stable facade:

```python
from features import build_features
```

Internally:

```text
base indicators
→ session/context
→ higher-timeframe alignment
→ structure/liquidity
→ PO3/FVG/IFVG
→ strategy state
→ premium/discount context
→ train-scoped normalization
→ schema/order validation
→ cache persistence
```

Every stage documents required inputs, produced columns, configuration gate, decision-time availability, and cache-identity inputs.

## Backtest and environment parity

The backtest loop belongs under `simulation.backtest`.

Shared deterministic mechanics belong in `trading`. Orchestration can remain separate where lifecycle semantics differ.

Parity requirements include identical cost model, stop/TP semantics, multi-TP accounting, risk budget where configured identically, tick/bar fill contract, and explicit tests for intentional differences.

A dedicated `simulation.parity` package owns fixed traces and assertions.

## Evaluation unification

All reusable evaluation code moves into the single top-level `evaluation/` package.

Subpackages:

- `rollout`;
- `metrics`;
- `reanalysis`;
- `reporting`;
- `plotting`.

Static and interactive plots share semantic data transforms, not renderer-specific drawing code.

Old evaluation paths remain compatibility shims only during migration.

## Campaign and experiment architecture

Reusable matrix/thesis campaign logic moves into `experiments`.

`experiments.variants` owns variant resolution and registry.

`experiments.campaigns` owns campaign lifecycle, checkpoint mapping, resume logic, run validation, worker commands, and campaign summaries.

`experiments.provenance` owns source/artifact identity and reproducibility records.

Slurm mechanics belong to `compute.slurm`, not experiment-domain code.

## Live and MT5 architecture

Broker-neutral live logic moves into `live`.

Broker-specific implementation moves into `integrations.mt5`.

Boundary:

```text
trading policy/risk
        ↓
live runtime
        ↓
integration adapter
        ↓
external terminal/broker
```

## Script migration architecture

Scripts are not libraries.

Examples:

```text
scripts/train/run_training.py
    -> training.run_training()

scripts/eval/reanalyze.py
    -> evaluation.reanalysis.reanalyze_checkpoint()

experiments/campaigns/execution.py
    -> experiments.campaigns.run_campaign()

experiments/campaigns/status.py
    -> experiments.campaigns reporting + compute.slurm status
```

Do not add `__init__.py` throughout `scripts/` to legitimize reusable imports. Move reusable logic into a top-level package.

Steady-state script acceptance:

- no reusable campaign/data/training/evaluation implementation remains under `scripts/`;
- no generic `matrix/`, `utility/`, `docker/`, `tests/`, or `setup/` category survives solely for historical convenience;
- every retained research script declares the package APIs/artifacts it consumes and produces;
- every retained Slurm shell wrapper contains resource/environment orchestration only;
- every compatibility wrapper has an expiry condition and is removed in Phase 11;
- declarative variant/resource lists live under `config/`, not `scripts/`.

<!-- SCRIPT_LEDGER_START -->
## Current script migration ledger

The pre-refactor audit found **152 files under `scripts/`** excluding `__pycache__`. After migration, **116 files remain** under the steady-state `scripts/compute/`, `scripts/maintenance/`, and `scripts/research/` groups. The ledger records each retained wrapper's owner and disposition; the moved package commands remain listed where their compatibility lifecycle matters.

| Current path | Target class | Target owner/location | Migration action |
|---|---|---|---|
| `scripts/research/benchmarks/measure_env_memory.py` | research command | `scripts/research/benchmarks/` | Keep thin; extract reusable benchmark/resource logic to owning packages. |
| `scripts/research/benchmarks/measure_ppo_throughput.py` | research command | `scripts/research/benchmarks/` | Keep thin; extract reusable benchmark/resource logic to owning packages. |
| `scripts/research/benchmarks/measure_shared_ppo_memory.py` | research command | `scripts/research/benchmarks/` | Keep thin; extract reusable benchmark/resource logic to owning packages. |
| `scripts/research/benchmarks/parse_stages_bench.py` | research command | `scripts/research/benchmarks/` | Keep thin; extract reusable benchmark/resource logic to owning packages. |
| `scripts/research/benchmarks/run_fps_probe.sh` | research command | `scripts/research/benchmarks/` | Keep thin; extract reusable benchmark/resource logic to owning packages. |
| `scripts/research/benchmarks/run_stages_benchmark.sh` | research command | `scripts/research/benchmarks/` | Keep thin; extract reusable benchmark/resource logic to owning packages. |
| `market_data/commands/discover.py` | compatibility wrapper | `market_data CLI + stable console command` | Move implementation to market_data; delete wrapper in Phase 11 after callers migrate. |
| `market_data/commands/prepare.py` | compatibility wrapper | `market_data CLI + stable console command` | Move implementation to market_data; delete wrapper in Phase 11 after callers migrate. |
| `scripts/docker/parallel-run.sh` | obsolete after gate migration | `tools/check + Docker Compose using stable commands` | Replace duplicate Docker orchestration; retain Docker only as an execution environment. |
| `scripts/docker/pipeline.sh` | obsolete after gate migration | `tools/check + Docker Compose using stable commands` | Replace duplicate Docker orchestration; retain Docker only as an execution environment. |
| `scripts/research/thesis/evaluation/defense_figures.py` | research command | `scripts/research/thesis/evaluation/` | Keep reproducible analysis thin; consume evaluation artifacts/APIs. |
| `scripts/research/thesis/evaluation/deflated_sharpe_family.py` | research command | `scripts/research/thesis/evaluation/` | Keep reproducible analysis thin; consume evaluation artifacts/APIs. |
| `scripts/research/thesis/evaluation/probe_seed50_opens.py` | research command | `scripts/research/thesis/evaluation/` | Keep reproducible analysis thin; consume evaluation artifacts/APIs. |
| `scripts/research/thesis/evaluation/report_ablations.py` | research command | `scripts/research/thesis/evaluation/` | Keep reproducible analysis thin; consume evaluation artifacts/APIs. |
| `scripts/research/thesis/evaluation/report_g3.py` | research command | `scripts/research/thesis/evaluation/` | Keep reproducible analysis thin; consume evaluation artifacts/APIs. |
| `scripts/research/thesis/evaluation/run_baseline_eval.py` | compatibility wrapper | `evaluation CLI/public API` | Move reusable evaluation semantics to evaluation; retire deep script entry point. |
| `scripts/research/thesis/evaluation/score_rule_baselines.py` | compatibility wrapper | `evaluation CLI/public API` | Move reusable evaluation semantics to evaluation; retire deep script entry point. |
| `scripts/research/thesis/evaluation/test_oos.py` | compatibility wrapper | `evaluation CLI/public API` | Move reusable evaluation semantics to evaluation; retire deep script entry point. |
| `experiments/variants/runner.py` | compatibility wrapper | `experiments CLI/public API` | Extract controller/state/orchestration into experiments; wrapper disappears after stable CLI adoption. |
| `scripts/research/campaigns/analyze_campaign_runtime_scaling.py` | research command | `scripts/research/campaigns/` | Keep audit/report command thin; move reusable campaign/artifact logic to packages. |
| `scripts/research/campaigns/audit_campaign_checkpoint_loads.py` | research command | `scripts/research/campaigns/` | Keep audit/report command thin; move reusable campaign/artifact logic to packages. |
| `scripts/research/campaigns/audit_campaign_source_fidelity.py` | research command | `scripts/research/campaigns/` | Keep audit/report command thin; move reusable campaign/artifact logic to packages. |
| `scripts/research/campaigns/audit_real_market_bundle_parity.py` | research command | `scripts/research/campaigns/` | Keep audit/report command thin; move reusable campaign/artifact logic to packages. |
| `scripts/research/campaigns/audit_real_market_feature_prefix_causality.py` | research command | `scripts/research/campaigns/` | Keep audit/report command thin; move reusable campaign/artifact logic to packages. |
| `scripts/research/campaigns/audit_variant_spaces.py` | research command | `scripts/research/campaigns/` | Keep audit/report command thin; move reusable campaign/artifact logic to packages. |
| `experiments/campaigns/checkpoints.py` | compatibility wrapper | `experiments.provenance/public API` | Move lineage semantics into experiments; retain only command adapter during migration. |
| `experiments/campaigns/execution.py` | compatibility wrapper | `experiments CLI/public API` | Extract controller/state/orchestration into experiments; wrapper disappears after stable CLI adoption. |
| `scripts/maintenance/migrations/migrate_preflight_campaign.py` | maintenance command | `scripts/maintenance/migrations/` | Use package APIs and artifact transactions; keep only migration/repair orchestration. |
| `scripts/maintenance/data-and-cache/rebuild_named_prebuilt_feature_caches.py` | maintenance command | `scripts/maintenance/data-and-cache/` | Use package APIs and artifact transactions; keep only migration/repair orchestration. |
| `scripts/maintenance/data-and-cache/rebuild_prebuilt_baseline_cache.py` | maintenance command | `scripts/maintenance/data-and-cache/` | Use package APIs and artifact transactions; keep only migration/repair orchestration. |
| `scripts/maintenance/data-and-cache/rebuild_full_period_data_cache.py` | maintenance command | `scripts/maintenance/data-and-cache/` | Use market-data/features/artifact APIs; preserve immutable cache identity and provenance. |
| `config/campaigns/resource_profile_a0_variant_list.txt` | config asset | `config/experiments or config/campaigns` | Move declarative list out of scripts; update every launcher/config reference. |
| `scripts/compute/slurm/campaigns/run_18variant_matrix.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_18variant_matrix_gh200_tmux.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_18variant_matrix_tmux.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_2variant_node.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_cpumask_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_cpumask_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_finalization_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_full_split_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_full_split_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_migration_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_migration_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_two_node_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_two_node_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_vector_resume_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_18way_vector_resume_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_3way_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_3way_prebuilt_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_3way_prebuilt_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_3way_shared_bundle_n64_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_3way_shared_bundle_n64_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_3way_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_campaign_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_replacement_campaign_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_replacement_campaign_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_rq1_pair_allocation.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_rq1_pair_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/campaigns/run_rr_two_gpu_tmux.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `experiments/campaigns/reporting.py` | research command | `scripts/research/campaigns/` | Keep audit/report command thin; move reusable campaign/artifact logic to packages. |
| `experiments/campaigns/orchestration.py` | compatibility wrapper | `experiments CLI/public API` | Extract controller/state/orchestration into experiments; wrapper disappears after stable CLI adoption. |
| `scripts/compute/slurm/campaigns/validate_cpu_mask_campaign_gpu.sh` | platform/Slurm wrapper | `scripts/compute/slurm/campaigns/` | Keep shell wrapper thin; move campaign/resource decisions to experiments/compute. |
| `scripts/compute/slurm/verify_worker_cpu_affinity.py` | platform/Slurm wrapper | `scripts/compute/slurm/` | Move reusable resource inspection to compute; wrapper invokes it. |
| `scripts/research/audits/audit_candidate_holdout_archive_exposure.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_candidate_holdout_repository_exposure.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_missing_m1_tick_reconstruction.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_replacement_fold_fit.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_replacement_m1_quality.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_replacement_partition_integrity.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_replacement_schedule_shape.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_replacement_tick_quality.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_rq1_candidate_config.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_seed50_screen_report_recoverability.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_tick_cache_timeline.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_tick_execution_coverage.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/audit_tick_tie_policy.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/fingerprint_tick_decision_schedule.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/maintenance/data-and-cache/rebuild_full_period_data_cache.py` | maintenance command | `scripts/maintenance/data-and-cache/` | Use market_data/features/artifacts APIs; keep only explicit maintenance orchestration. |
| `scripts/research/audits/render_idea12_synthetic_figures.py` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/run_18_feature_cache_mapping_gate.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/maintenance/data-and-cache/run_18_feature_cache_rebuild.sh` | maintenance command | `scripts/maintenance/data-and-cache/` | Use market_data/features/artifacts APIs; keep only explicit maintenance orchestration. |
| `scripts/research/audits/run_candidate_holdout_archive_audit_gate.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/maintenance/data-and-cache/run_full_period_data_and_feature_cache_rebuild.sh` | maintenance command | `scripts/maintenance/data-and-cache/` | Use market_data/features/artifacts APIs; keep only explicit maintenance orchestration. |
| `scripts/research/audits/run_observation_schema_gate.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/run_p1_schedule_freeze_gate.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/run_p1_tick_fold_validation.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/run_p1_tick_quality_validation.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/run_p3_checkpoint_recovery_smoke.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/run_rq1_candidate_preflight_gate.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `scripts/research/audits/run_tick_tie_policy_gate.sh` | research command | `scripts/research/audits/` | Preserve reproducible research gate; extract reusable validation primitives to owners. |
| `tools/setup-oasis-agents` | engineering wrapper | `tools/bootstrap or tools/orchestra helper` | Move developer setup out of research scripts; delete old path after docs/callers migrate. |
| `tools/setup-orchestra-view` | engineering wrapper | `tools/bootstrap or tools/orchestra helper` | Move developer setup out of research scripts; delete old path after docs/callers migrate. |
| `tools/setup-venv` | engineering wrapper | `tools/bootstrap or tools/orchestra helper` | Move developer setup out of research scripts; delete old path after docs/callers migrate. |
| `tools/setup-venv-mi210` | engineering wrapper | `tools/bootstrap or tools/orchestra helper` | Move developer setup out of research scripts; delete old path after docs/callers migrate. |
| `tools/setup-venv-x86` | engineering wrapper | `tools/bootstrap or tools/orchestra helper` | Move developer setup out of research scripts; delete old path after docs/callers migrate. |
| `scripts/strategy_checks/__init__.py` | obsolete | `none` | Scripts are not a reusable package; remove after imports are eliminated. |
| `scripts/research/strategy_checks/edge_harness.py` | research command | `scripts/research/strategy_checks/` | Keep experiment-specific analysis thin; move shared feature/evaluation logic to owners. |
| `scripts/research/strategy_checks/ema_tp_replay.py` | research command | `scripts/research/strategy_checks/` | Keep experiment-specific analysis thin; move shared feature/evaluation logic to owners. |
| `scripts/research/strategy_checks/entry_edge_check.py` | research command | `scripts/research/strategy_checks/` | Keep experiment-specific analysis thin; move shared feature/evaluation logic to owners. |
| `scripts/research/strategy_checks/eval_seed_variance.py` | research command | `scripts/research/strategy_checks/` | Keep experiment-specific analysis thin; move shared feature/evaluation logic to owners. |
| `scripts/research/strategy_checks/ifvg_chain_check.py` | research command | `scripts/research/strategy_checks/` | Keep experiment-specific analysis thin; move shared feature/evaluation logic to owners. |
| `scripts/research/strategy_checks/plot_ifvg_chain_trades.py` | research command | `scripts/research/strategy_checks/` | Keep experiment-specific analysis thin; move shared feature/evaluation logic to owners. |
| `scripts/research/strategy_checks/pre_ny_bias_check.py` | research command | `scripts/research/strategy_checks/` | Keep experiment-specific analysis thin; move shared feature/evaluation logic to owners. |
| `scripts/research/strategy_checks/sweep_displacement_check.py` | research command | `scripts/research/strategy_checks/` | Keep experiment-specific analysis thin; move shared feature/evaluation logic to owners. |
| `scripts/tests/run_full_pytest_gpu.sh` | obsolete after gate migration | `tools/check gpu\|triton\|full` | Replace test-launch shell variants with named canonical gates. |
| `scripts/tests/run_full_tests.sh` | obsolete after gate migration | `tools/check gpu\|triton\|full` | Replace test-launch shell variants with named canonical gates. |
| `scripts/tests/run_full_tests_tmux.sh` | obsolete after gate migration | `tools/check gpu\|triton\|full` | Replace test-launch shell variants with named canonical gates. |
| `scripts/tests/run_sac_test_cores.sh` | obsolete after gate migration | `tools/check gpu\|triton\|full` | Replace test-launch shell variants with named canonical gates. |
| `scripts/research/thesis/evidence/analyze_paired_seed_results.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/thesis/evidence/audit_checkpoint_frame_bundle_parity.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/thesis/evidence/build_campaign_evidence_index.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/thesis/evidence/build_campaign_trial_ledger.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/thesis/evidence/build_seed50_screen_finding_cards.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/thesis/evidence/compare_deterministic_reanalysis.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/thesis/evidence/reconcile_campaign_attempts.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/thesis/evidence/verify_numerical_report.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/thesis/evidence/verify_result_manifest.py` | research command | `scripts/research/thesis/evidence/` | Keep evidence assembly/verification reproducible; use canonical artifacts/evaluation APIs. |
| `scripts/research/benchmarks/compare_encoders.py` | research command | `scripts/research/benchmarks/` | Use training/evaluation APIs; preserve comparison protocol, not duplicate setup logic. |
| `training/runner/lstm_baseline.py` | compatibility wrapper | `training CLI/public API` | Move reusable training implementation to training; retire script entry point. |
| `scripts/research/frozen-launches/train_one_variant.sh` | research/platform wrapper | `scripts/research/frozen-launches or scripts/compute/slurm/` | Move semantics to config + training CLI; retain only if needed to replay a named experiment. |
| `scripts/train/train_rl.py` | temporary compatibility wrapper | `training CLI/public API` | Already deprecated; preserve only until all saved/current launch commands migrate, then delete. |
| `scripts/research/frozen-launches/train_rr_idea1.sh` | research/platform wrapper | `scripts/research/frozen-launches or scripts/compute/slurm/` | Move semantics to config + training CLI; retain only if needed to replay a named experiment. |
| `scripts/research/frozen-launches/train_rr_idea2.sh` | research/platform wrapper | `scripts/research/frozen-launches or scripts/compute/slurm/` | Move semantics to config + training CLI; retain only if needed to replay a named experiment. |
| `training/runner/vae.py` | compatibility wrapper | `training CLI/public API` | Move reusable training implementation to training; retire script entry point. |
| `docs/operations/triton/README.md` | documentation | `docs/operations/triton/` | Move operational documentation to docs; keep command references current. |
| `scripts/compute/slurm/activate_venv.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `scripts/compute/slurm/attach_or_alloc.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `scripts/compute/slurm/campaign_allocation_wrapper.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `scripts/compute/slurm/campaign_git_identity.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `scripts/compute/slurm/campaign_recovery_supervisor.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `scripts/compute/slurm/campaign_srun_gh200_guard.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `scripts/compute/slurm/worker_bin/srun` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `compute/slurm/worker_dispatch.py` | compatibility wrapper | `compute CLI/public API` | Extract reusable Slurm/resource/recovery behavior to compute; keep thin adapter temporarily. |
| `compute/slurm/capacity.py` | compatibility wrapper | `compute CLI/public API` | Extract reusable Slurm/resource/recovery behavior to compute; keep thin adapter temporarily. |
| `scripts/compute/slurm/find_gpu_session.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `market_data/commands/prebuild_campaign_cache.py` | compatibility wrapper | `market_data + experiments/compute API` | Move cache semantics to market_data/artifacts; orchestration to experiments/compute. |
| `compute/slurm/resume.py` | compatibility wrapper | `compute CLI/public API` | Extract reusable Slurm/resource/recovery behavior to compute; keep thin adapter temporarily. |
| `scripts/research/frozen-launches/reexport_oct3_runs.sh` | frozen research wrapper | `scripts/research/frozen-launches/` | Retain only for provenance/replay if evidence still depends on it. |
| `experiments/campaigns/status.py` | compatibility wrapper | `experiments reporting CLI` | Move status model/reporting to experiments; retain login-node-safe wrapper if useful. |
| `compute/slurm/continuation.py` | compatibility wrapper | `compute CLI/public API` | Extract reusable Slurm/resource/recovery behavior to compute; keep thin adapter temporarily. |
| `scripts/compute/slurm/run_current_tree_gpu_ci.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `compute/slurm/resources.py` | compatibility wrapper | `compute CLI/public API` | Extract reusable Slurm/resource/recovery behavior to compute; keep thin adapter temporarily. |
| `scripts/compute/slurm/status.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `scripts/compute/slurm/two_trains_one_h200.sh` | platform/Slurm wrapper | `scripts/compute/slurm/` | Keep allocation/shell mechanics thin; invoke compute/experiments APIs. |
| `scripts/utility/_fix_device_indent.py` | obsolete one-off patch | `none` | Delete after verifying the intended source change is already represented in versioned code. |
| `scripts/utility/_patch_max_n_envs.py` | obsolete one-off patch | `none` | Delete after verifying the intended source change is already represented in versioned code. |
| `scripts/research/audits/leak_detectors.py` | research command | `scripts/research/audits/` | Extract reusable leakage checks to market_data/features/tests as appropriate. |
| `scripts/research/thesis/evaluation/rescore_sl_fill.py` | frozen research command | `scripts/research/thesis/evaluation/` | Preserve only as explicit historical rescore protocol using evaluation APIs. |
| `scripts/research/strategy_checks/verify_strategy_features.py` | research command | `scripts/research/strategy_checks/` | Use owning package APIs; remove generic utility category. |
| `config/campaigns/variant_list.txt` | config asset | `config/experiments or config/campaigns` | Move declarative list out of scripts; update every launcher/config reference. |
| `config/campaigns/variant_list_3_strategy_20261014.txt` | config asset | `config/experiments or config/campaigns` | Move declarative list out of scripts; update every launcher/config reference. |
| `config/campaigns/variant_list_rq1_entry_fsm.txt` | config asset | `config/experiments or config/campaigns` | Move declarative list out of scripts; update every launcher/config reference. |

<!-- SCRIPT_LEDGER_END -->

## Tests architecture

Tests mirror package ownership:

```text
tests/
├── architecture/
│   ├── test_public_imports.py
│   ├── test_dependency_rules.py
│   ├── test_no_scripts_imports.py
│   └── test_legacy_imports.py
├── market_data/
├── features/
├── trading/
├── simulation/
├── models/
├── training/
├── evaluation/
├── experiments/
├── live/
├── integrations/
├── compute/
└── orchestra/
```

## Architecture tests

Before large migrations, add lightweight repository architecture checks.

### Public import tests

Verify the stable imports shown above.

### Dependency-rule tests

Use an AST-based repository check initially to detect forbidden imports. Examples include features importing training, trading importing evaluation, production code importing scripts, and models importing training orchestration.

### Root-ownership tests

Add checks for forbidden production writes into `baseline/`, scripts containing reusable package imports in the wrong direction, and package code importing command wrappers.

### Configuration contract tests

Verify unknown-key rejection, deterministic merges, canonical hashes, historical snapshot loading, and schema migration behavior.

### Artifact lifecycle tests

Verify atomic completion, rejection of incomplete artifacts, manifest/schema validation, provenance parent links, immutability/overwrite policy, and baseline promotion gates.

### Cache identity tests

Mutating any semantic cache input must change the cache identity. Non-semantic runtime changes must not invalidate caches unnecessarily.

### Repository hygiene tests

Verify tracked generated files have declared producers where required, ordinary tests do not write to repository artifact roots, and ignore rules do not hide source packages or tracked examples.

### Shared-helper ownership tests

Add focused checks for high-value primitives that must have one implementation. At minimum, prohibit new ad-hoc file-hashing and atomic-artifact-writing helpers outside their canonical owners once migration is complete. Use allowlists during transition and reduce them monotonically.

Do not enforce a blanket "no duplicate function names" rule; interface methods and conventional entry points legitimately share names.

### Equivalence tests for consolidated helpers

Every consolidation that can affect research behavior needs before/after parity tests at the owning boundary. For hashes/serialization this should normally be exact. For numerical functions use explicitly documented tolerances. For stateful functions compare resulting state/events/artifacts, not only return values.

### Semantic naming tests

Add lightweight architecture checks for clearly non-semantic numbered placeholders in newly migrated code, such as `function1`, `part2`, `setting3`, or `step_4`. Maintain a narrow allowlist for legitimate domain numbering such as take-profit levels, timeframes, seeds, schema versions, and explicitly documented migration phases.

Do not enforce naming mechanically beyond obvious placeholder patterns. Code review remains responsible for judging whether a name communicates the underlying domain concept.

### Vocabulary consistency checks

Where names are part of machine-readable contracts, add schema/tests that protect canonical terminology across configuration, runtime state, metrics, manifests, and reports. Historical field aliases may remain only through explicit compatibility adapters.

### Execution-surface drift tests

Verify that stable wrappers resolve to existing package APIs, Docker/console entry points refer to supported commands, and production packages never import scripts, tools, or CI wrappers.

### CI-definition consistency tests

Validate the architectural pieces most likely to drift: canonical named gates exist, workflow references resolve, required tools do not silently skip, and every first-class package root is included by lint/type/build configuration. Do not unit-test workflow YAML line by line.

### Legacy-import budget

Track remaining imports from the legacy monolithic package mechanically.

The budget must decrease phase by phase and reach zero before that package is removed.

The architecture plan intentionally does not encode its package name.

## Packaging and tooling changes

Repository-root packages require project-tooling changes.

### Build configuration

The wheel/build configuration must include every new top-level package, including `configuration` and `artifacts`, rather than a single monolithic application package plus Orchestra.

### Ruff

Update source roots, first-party package declarations, and import sorting expectations.

### Mypy

Update checked package roots so strict typing covers every top-level package. Avoid namespace-package ambiguity; every importable code package has `__init__.py`.

### Pytest

Keep `tests/` as the root and mirror the package structure.

### Console entry points

Move frequent supported commands to project console entry points backed by package-owned CLI functions. Research, maintenance, and Slurm wrappers remain under scripts when they are not stable user-facing commands.

Console commands, scripts, Docker Compose, Slurm wrappers, and CI must converge on the same package APIs.

### Quality-gate entry points

Provide one canonical local gate surface, preferably tools/check, with named fast/unit/integration/full and platform-specific gate selections. GitHub workflows and thin CI adapters invoke this surface instead of duplicating raw lint/type/test command definitions.

### Documentation

Update `README.md`, `docs/architecture.md`, `.agents/architecture.md`, domain maps, and running-command documentation.

## Migration phases

### Phase 0 — Freeze behavior and define contracts

- Record current CI baseline.
- Record current script/CI command surfaces, test markers, Docker/Slurm entry points, and duplicated or platform-specific gates.
- Freeze the current 15-job `ci.yml` graph and 152-file script ledger as migration inputs; changes to these inventories during the refactor must be intentional and documented.
- Add architecture import/dependency tests.
- Add characterization tests for environment, backtest, features, training setup, evaluation, and campaign behavior.
- Record checkpoint/action/observation/feature schemas.
- Make no production behavior changes.

### Phase 1 — Create root package and lifecycle foundations

Introduce the first needed package shells and `__init__.py` APIs. Establish `configuration` and `artifacts` first because later packages depend on consistent config/provenance/output mechanics.

Then create the initial domain surfaces for `trading`, `simulation`, `market_data`, and `features`. Define the root ownership/retention rules for `config/`, `data/`, `cache/`, `outputs/`, `logs/`, and `baseline/` before migrating writers.

Do not create empty target directories just to match the diagram.

### Phase 2 — Extract pure trading contracts

Move account, policy, risk, orders, costs, and deterministic execution primitives into `trading`.

Acceptance: legacy and repeated-payout behavior unchanged; risk boundary tests pass; no duplicate domain formulas.

### Phase 3 — Split simulation runtime

Move environment mechanics and backtest orchestration into `simulation`.

Acceptance: environment golden traces, backtest parity, action/observation schemas, tick execution, multi-TP, EOD, guardrails, and payout behavior match.

### Phase 4 — Split market-data and feature pipelines

Move data ingestion/transforms/splits to `market_data` and feature stages to `features`.

At the same time, migrate dataset manifests and cache identities so data/cache layout is produced through package APIs rather than filename conventions.

Acceptance: feature values/order, causal-prefix tests, split identities, and cache hashes match for unchanged semantics.

### Phase 5 — Split models and training

Move model factories/encoders/policies to `models` and training composition to `training`.

Move run-directory creation, resolved-config snapshots, checkpoint manifests, diagnostics, and training logs behind the artifact layout/provenance APIs.

Acceptance: existing commands work through wrappers, seeds match, checkpoints load, manifests match.

### Phase 6 — Consolidate evaluation

Move rollout, metrics, reanalysis, reporting, and plotting into `evaluation`.

Consolidate evaluation artifact schemas/writers so reanalysis, reports, plots, and numerical evidence share one manifest lineage.

Acceptance: fixed checkpoint reanalysis remains numerically unchanged and only one canonical reusable evaluation package remains.

### Phase 7 — Extract experiments and campaigns

Move variant/campaign/provenance logic into `experiments` and scheduler/resource mechanics into `compute`.

Migrate campaign directory creation, worker logs, checkpoint maps, reports, completion markers, and resume lineage into the shared artifact lifecycle.

Acceptance: campaign dry runs, checkpoint discovery/resume, worker commands, and provenance match.

### Phase 8 — Separate live runtime and integrations

Move broker-neutral logic to `live` and MT5-specific code to `integrations.mt5`.

Acceptance: paper mode, live risk parity, and integration tests pass.

### Phase 9 — Refactor Orchestra and engineering execution surfaces

Split Orchestra phases/execution/recovery/reporting behind its package API.

Establish the canonical quality-gate surface, migrate GitHub workflows to named gates, normalize test markers, eliminate duplicate CI command definitions, and migrate Docker/Slurm/script entry points to stable package APIs.

Acceptance: normal `ci.yml` matches the 9-validation-job + `ci-success` target graph; local and hosted CPU gates share semantics; `ci-scripts/` contains only proven platform glue or is deleted; the 152-file current script ledger has no unresolved entry; stable operations use console/package APIs; Tier D Triton/GPU validation is documented and runnable; no required gate silently skips; stale script, Docker, Slurm, and deep-module command paths are gone.

### Phase 10 — Normalize repository support architecture

After domain writers have moved, normalize `config/`, `data/`, `cache/`, `outputs/`, `logs/`, `baseline/`, `scripts/`, `tests/`, `docs/`, `tools/`, and `ci-scripts/` around the ownership rules above.

This phase is not a cosmetic directory shuffle. Migrate generators/consumers first, add compatibility path resolution where needed, verify historical run reanalysis, then retire obsolete layouts.

Acceptance: no production caller depends on an undocumented generated path; every retained artifact class has an owner/schema/provenance rule; scripts and CI are thin; test writes are isolated; baseline promotion is explicit.

### Phase 11 — Remove compatibility layout

Only after repository-wide import search, public-import tests, full CI, smoke training, checkpoint reanalysis, and campaign dry-run all succeed.

Then remove compatibility forwarding modules and update build/lint/type-check configuration and architecture docs.

## Trading-policy and repeated-payout invariants

Architecture refactoring must preserve:

- selectable legacy policy;
- repeated-payout policy;
- $100,000 operating balance;
- $110,000 realized-balance payout target;
- $90,000 static account floor;
- fixed-session $5,000 daily-loss constraint;
- disabled account-level trailing drawdown in repeated-payout mode;
- 0-0.90% continuous agent risk;
- zero risk as a valid no-trade action;
- requested versus approved/executed risk diagnostics;
- execution-reserve-aware sizing;
- flat realized payout trigger;
- cumulative payout accounting;
- chronological continuation across evaluation cycles;
- single-cycle training behavior.

Equations and state transitions belong in `trading`; environment sequencing belongs in `simulation`.

## Cross-cutting research invariants

No architecture phase may weaken:

- causal features;
- train-only normalization;
- locked OOS separation;
- purged walk-forward behavior;
- session eligibility semantics;
- cache identity/versioning;
- train/evaluation/backtest execution parity;
- checkpoint reproducibility;
- deterministic provenance;
- experiment seed identity.

## Size and complexity guidance

Architecture is package-first, but size remains a warning signal.

- aim for cohesive modules usually below roughly 800 lines;
- investigate functions above roughly 150 lines;
- orchestration functions above roughly 250 lines should usually be decomposed;
- avoid classes above roughly 1,000 lines unless intentionally declarative;
- prefer 3-10 cohesive modules in a subpackage over one monolith;
- avoid one-function micro-modules unless they are genuine boundary abstractions;
- keep `__init__.py` small.

These are review heuristics, not automatic rewrite rules.

## Commit strategy

Implement the refactor in the ordered phases above, but land the completed architecture refactor as **one final commit** after all required validation passes.

During implementation, use the phases as working checkpoints for reasoning, testing, and rollback discipline. Do not create a chain of architecture commits solely because the plan has multiple phases.

Before the final commit:

1. complete every planned ownership move;
2. remove temporary compatibility shims that are no longer required;
3. run the architecture import/dependency checks;
4. run scoped parity tests for every moved subsystem;
5. run the full repository CI gate;
6. run smoke training and fixed-checkpoint reanalysis where required by the migration phases;
7. inspect the final diff for semantic drift and unrelated dirty-tree changes;
8. stage only the architecture-refactor files and intended compatibility updates.

Use one detailed commit message that describes the repository-wide package architecture, major ownership moves, compatibility guarantees, and validation performed.

## Agent execution guidance

For each phase, an implementation agent should:

1. read this plan;
2. inspect the owning package's `__init__.py`;
3. inspect only the source methods and callers involved in the move;
4. inventory every config/data/cache/output/log/baseline path read or written by those callers;
5. identify the producer, consumer, identity inputs, schema, completion rule, and retention class for generated artifacts;
6. inventory duplicated helpers/functions in the affected ownership boundary and decide whether each stays local, moves to the owning package, or becomes a shared primitive;
7. write equivalence/characterization tests before replacing duplicated behavior;
8. perform a semantic naming audit for moved modules, functions, state fields, config keys, metrics, artifacts, tests, and docs; replace ordinal/placeholder names with domain names where compatibility allows;
9. verify that extracted functions represent meaningful responsibilities rather than numbered fragments;
10. audit every execution surface that invokes the moved responsibility: console commands, Python scripts, shell/Slurm wrappers, tools, Docker, GitHub workflows, and saved campaign commands;
11. migrate those callers to one stable package API or gate command and remove duplicated orchestration logic;
12. add missing characterization tests first;
13. create the target package/subpackage with `__init__.py`;
14. move one coherent responsibility;
15. expose the intended public API;
16. add a forwarding compatibility import if required;
17. migrate direct callers;
18. run scoped parity tests;
19. run architecture import/dependency tests;
20. inspect the diff for semantic drift;
21. run the repository commit gate;
22. update architecture/domain documentation.

The agent should not redesign the entire repository during every phase.

## Definition of done

The repository-wide architecture refactor is complete when:

- repository-root packages express the domain architecture directly;
- repository-root asset/generated directories have explicit ownership, lifecycle, and retention rules;
- shared configuration resolution has one implementation and historical resolved configs remain interpretable;
- generated artifacts have typed identities, manifests, provenance links, completion semantics, and regeneration paths;
- datasets and partitions are identified by manifests rather than inferred only from filenames;
- caches are disposable, validated, identity-keyed, and never the sole source of research evidence;
- run/campaign/evaluation outputs are immutable artifact trees with canonical manifests;
- run-scoped logs are colocated with their owning artifacts and scientific state is not log-only;
- baselines can only be created through an explicit validated promotion workflow;
- scripts, tools, and CI wrappers contain no duplicate domain algorithms;
- stable console commands, scripts, Docker, Slurm, tools, and CI converge on the same package APIs;
- one canonical validation-gate vocabulary is shared by local development and hosted CI;
- normal `ci.yml` contains only the target semantic jobs (`dependency-lock`, `quality`, `architecture`, `contracts`, `unit`, `build-smoke`, `integration`, `parity`, `training-smoke`) plus the `ci-success` aggregator, unless a later reviewed architecture decision explicitly changes the graph;
- every current script has a resolved migration-ledger disposition and no generic script category survives without a defined steady-state role;
- `ci-scripts/` is removed if no irreducible platform-only glue remains;
- hosted CI avoids redundant full-suite execution unless each run validates a distinct environment contract;
- test markers form a scheduling contract for unit, integration, slow, GPU, Triton, and research validation;
- CPU CI uses isolated data/artifact roots and cannot consume developer caches or outputs accidentally;
- Triton/GPU validation is explicit for resource-sensitive training and campaign behavior;
- Docker/release automation exists only for supported reproducibility/runtime needs and has no stale command paths;
- high-value shared primitives have one semantic owner and one canonical implementation;
- no catch-all utility package becomes a dependency dumping ground;
- every duplicate-function replacement with research impact has explicit result-equivalence coverage;
- public names express domain intent and avoid ordinal/placeholder naming except where the number is part of the domain contract;
- extracted functions represent meaningful responsibilities rather than `part1`/`part2` decomposition;
- canonical vocabulary is consistent across code, config, state, metrics, artifacts, tests, and documentation;
- ordinary tests never mutate repository data/cache/output/log/baseline roots;
- generated documentation/evidence records its generator and source artifacts;
- every importable package/subpackage has an intentional `__init__.py`;
- public imports are obvious and protected by tests;
- the trading environment is a thin simulation facade rather than a mechanics container;
- trading equations and state transitions have one owning package;
- training coordinates services instead of owning upstream domain behavior;
- feature construction is a named causal pipeline;
- evaluation has one canonical package;
- campaign logic is reusable package code rather than script internals;
- live logic is broker-neutral and external systems are isolated under integrations;
- scripts are command wrappers rather than libraries;
- build, lint, type-check, and test configuration recognize the root packages;
- architecture docs and the actual directory tree agree;
- legacy compatibility imports are zero;
- deterministic parity, checkpoints, smoke training, and OOS reanalysis remain valid;
- a developer can identify where to import and where to modify a concern by reading the repository tree and package `__init__.py` files.
