# Agent Implementation Plan — Code Graph Trust & Configuration Dependency Hardening

## Objective

Improve the reliability of `code-review-graph` for impact analysis in `Aalto_MS_Thesis`, especially for:

1. Internal Python call resolution.
2. YAML/TOML configuration dependencies.
3. Training-critical execution paths.
4. Repeatable graph quality measurement.

The graph already covers tracked Python and shell files well. The goal is therefore **relationship quality**, not source-file discovery.

Treat this as **P1 engineering infrastructure**. Do not allow graph work to block training fixes involving data leakage, reward correctness, environment/action semantics, train/eval parity, or configuration correctness.

---

## Repository Context

Remote repository:

```text
code.triton.aalto.fi
/scratch/work/nguyenl37/Aalto_MS_Thesis
```

At planning time, the inspected HEAD was:

```text
45a62d0
```

The implementation agent must record its own exact `HEAD` and dirty-tree state at the start.

Existing graph infrastructure:

```text
.code-review-graph/graph.db
tools/crgraph
.agents/rules/code-review-graph.md
```

`tools/crgraph` already exposes commands including:

```text
build
update
postprocess
status
eval
detect-changes
query
impact
search
flows
flow
architecture
communities
dead-code
refactor
```

Important configuration sources include:

```text
pyproject.toml
config/base/default.yaml
config/experiments.yaml
config/quality_experiments.yaml
config/1_baseline.yaml
config/2_po3_ifvg.yaml
config/3_distribution.yaml
config/features/*.yaml
config/campaigns/*.yaml
```

Configuration is consumed through mechanisms including:

```text
OmegaConf.load(...)
OmegaConf.merge(...)
load_config(...)
yaml.safe_load(...)
tomllib
```

---

## Phase 0 — Establish a Reproducible Baseline

Record before changing anything:

```bash
git rev-parse HEAD
git status --short
./tools/crgraph status
```

Run the graph's built-in evaluation/status capabilities if available:

```bash
./tools/crgraph eval
./tools/crgraph architecture
./tools/crgraph flows
```

Capture indexed files by language, nodes, edges by relation type, `CALLS` edge count, exactly resolved `CALLS` edges, `TESTED_BY` edge count, unresolved targets, graph database size, and graph build/update duration if readily available.

Store:

```text
outputs/code_graph_quality/baseline.json
outputs/code_graph_quality/baseline.md
```

Include:

```text
git_commit
dirty_tree
timestamp
graph_tool_version
```

### Acceptance

A future agent must be able to compare graph quality before/after changes using the same procedure.

---

## Phase 1 — Measure the Correct `CALLS` Metric

Do not optimize the existing raw `exact CALLS / all CALLS` ratio directly.

Classify each `CALLS` edge into:

```text
internal_exact
internal_ambiguous
internal_unresolved
external_library
python_builtin
dynamic_or_indeterminate
```

Where possible record:

```text
source_file
source_symbol
call_expression
candidate_target
resolution_reason
```

Create a repository-side utility if needed:

```text
scripts/dev/code_graph/audit_calls.py
```

Inspect the graph DB schema before assuming table or column names.

Primary metric:

```text
internal_resolution_rate =
internal_exact /
(internal_exact + internal_ambiguous + internal_unresolved)
```

Also report:

```text
internal_unresolved_rate
internal_ambiguous_rate
external_call_count
dynamic_call_count
```

### Acceptance

The report must answer:

> Of calls that appear to target this repository, how many can the graph resolve reliably?

Do not use the raw `8,810 / 47,216` ratio as the principal graph-quality metric.

---

## Phase 2 — Diagnose Internal Resolution Failures

Sample unresolved and ambiguous internal calls and classify root causes.

Inspect at least relative imports, import aliases, re-exports, instance/class/static methods, inheritance, attribute calls, package `__init__.py` exports, decorated functions, dynamic strategy/component selection, and factory functions.

Prioritize failures affecting:

```text
src/
scripts/train*/
training entrypoints
TradingEnv
feature pipeline
reward pipeline
execution/action processing
evaluation
campaign controller
```

Produce:

```text
outputs/code_graph_quality/unresolved_internal_calls.json
```

Fix resolution where relationships can be determined statically with high confidence. Mark genuinely dynamic dispatch as `dynamic_or_indeterminate`.

---

## Phase 3 — Add Configuration Dependency Analysis

This is the most important missing relationship class.

Detect configuration consumers through at least:

```python
OmegaConf.load(...)
OmegaConf.merge(...)
load_config(...)
yaml.safe_load(...)
tomllib.load(...)
tomllib.loads(...)
Path(...).read_text(...)
```

Represent or export relations equivalent to:

```text
CONFIG_FILE -> LOAD_SITE
LOAD_SITE -> PYTHON_SYMBOL
```

Examples:

```text
config/base/default.yaml
    -> load_config
    -> training/environment construction

config/2_po3_ifvg.yaml
    -> OmegaConf.load
    -> PO3/IFVG experiment configuration

config/features/full_po3_mtf.yaml
    -> feature configuration
    -> build feature pipeline

pyproject.toml
    -> architecture/test/tool configuration
```

If the upstream graph cannot safely store custom YAML/TOML nodes, implement a companion dependency index.

Suggested paths:

```text
scripts/dev/code_graph/audit_config_dependencies.py
outputs/code_graph_quality/config_dependencies.json
```

---

## Phase 4 — Add Key-Level Config Tracing for Critical Settings

Prioritize:

```text
env.*
reward*
features.*
strategy.*
algo.*
risk*
guardrail*
action*
tp*
sl*
campaign overrides
train/test split settings
seed
timesteps
n_envs
```

Target representation:

```text
config/base/default.yaml:env.reward_mode
    -> config merge/resolution
    -> reward selection
    -> TradingEnv

config/features/full_po3_mtf.yaml:<feature-key>
    -> feature resolver
    -> build_features

campaign YAML:<override>
    -> campaign parser
    -> resolved experiment configuration
```

Do not attempt universal arbitrary YAML data-flow analysis. Implement keys that affect training semantics and thesis experiments.

### Acceptance

For a critical config key, an agent should be able to determine where it is defined, overridden, loaded, and which Python path consumes it.

---

## Phase 5 — Verify Thesis-Critical Execution Flows

### Flow A — Training

```text
campaign YAML
→ campaign launcher/controller
→ training entrypoint
→ config loading/merging
→ feature construction
→ environment construction
→ model/PPO construction
→ training loop
→ checkpoint/artifact logging
```

### Flow B — Environment step

```text
policy action
→ action decoding
→ direction/intensity/risk
→ SL/TP selection
→ multi-TP handling
→ position execution
→ TradingEnv.step
→ reward
→ equity/drawdown guardrails
→ episode termination/logging
```

### Flow C — Thesis feature variants

Verify baseline, PO3/IFVG, distribution, full MTF feature configuration, and ablation variants.

### Flow D — Evaluation

```text
checkpoint
→ config restoration
→ evaluation environment
→ deterministic inference
→ execution
→ metrics
→ thesis evidence/output
```

For every flow, compare graph trace against source trace and record discrepancies.

---

## Phase 6 — Add Automated Graph Quality Checks

Create:

```text
scripts/dev/code_graph/check_graph_quality.py
```

Validate invariants such as:

```text
all tracked .py files indexed
all tracked .sh files indexed
critical config files discovered
no unexpected increase in internal_unresolved rate
critical flows remain discoverable
graph database can be queried successfully
```

Avoid fixed node-count assertions. Prefer invariant and threshold checks such as:

```text
internal_resolution_rate >= baseline - permitted_regression
```

Fail explicitly if a critical file disappears from indexing.

---

## Phase 7 — Integrate With Existing `tools/crgraph`

Possible commands:

```bash
./tools/crgraph quality
./tools/crgraph quality --json
./tools/crgraph verify-critical
```

If wrapper modification would make upstream upgrades harder, add `tools/crgraph-quality` instead.

Do not fork or vendor the graph package unless a concrete upstream limitation requires it.

---

## Phase 8 — Documentation

Update:

```text
.agents/rules/code-review-graph.md
```

Add:

> Use the graph for navigation, blast-radius discovery, and candidate impact analysis. Verify important execution paths and configuration dependencies against source and tests.

Document:

```text
internal_exact
internal_ambiguous
internal_unresolved
external_library
dynamic_or_indeterminate
```

Also document:

```text
TESTED_BY resolution != test coverage
```

A resolved `TESTED_BY` relation only proves that the recorded edge points to an indexed symbol.

---

## Tests

Add focused tests for repository-owned classification and dependency extraction:

```text
direct internal function call
aliased import
relative import
instance method
external library call
builtin call
ambiguous method call
dynamic dispatch
literal OmegaConf.load path
Path-based YAML load
yaml.safe_load
TOML read
config overlay/merge
missing config path
duplicate config consumers
```

Do not test upstream `code-review-graph` behavior itself unless the repository wraps or transforms that behavior.

---

## Verification Commands

Run appropriate project checks plus:

```bash
./tools/crgraph build
./tools/crgraph status
./tools/crgraph eval
```

Then run the new quality tooling and compare baseline/final reports.

Inspect:

```bash
git diff --stat
git diff
git status --short
```

Do not modify or discard unrelated pre-existing working-tree changes.

---

## Definition of Done

- Python and shell source coverage remains complete.
- `CALLS` edges are classified meaningfully.
- A repository-internal call resolution rate is available.
- Internal unresolved calls have categorized causes.
- Critical YAML/TOML files have discoverable dependency relationships.
- Important configuration keys can be traced into their principal Python consumers.
- Training, environment/action/reward, feature variant, and evaluation flows are verified against source.
- Dynamic relationships are explicitly marked rather than guessed.
- Automated graph-quality checks exist.
- Baseline and final graph-quality reports are generated.
- Existing `tools/crgraph` workflow remains usable.
- Relevant tests pass.
- Existing unrelated working-tree changes remain intact.

---

## Agent Execution Priority

```text
P0  Baseline and inspect graph schema/tool capabilities
P1  CALLS classification
P2  Diagnose internal unresolved calls
P3  Configuration file dependencies
P4  Critical config-key tracing
P5  Critical execution-flow verification
P6  Automated regression checks
P7  Wrapper integration + documentation
P8  Final rebuild, tests, before/after report
```

Do not spend time maximizing total graph-edge count.

Optimize for **correct impact analysis of repository-owned code and experiment configuration**.

The final implementation report should state:

```text
before metrics
after metrics
remaining unresolved categories
critical paths verified
configuration coverage
tests run
known limitations
```
