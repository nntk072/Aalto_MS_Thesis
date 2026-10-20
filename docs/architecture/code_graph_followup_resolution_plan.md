# Agent Implementation Plan — Code Graph Follow-up Resolution & Strict Config Traceability

## Objective

Complete the remaining code-graph hardening work after the first graph-quality implementation.

This follow-up has three concrete goals:

1. Resolve indirect configuration dependencies for critical experiment configs, especially:
   - config/quality_experiments.yaml
   - config/1_baseline.yaml
   - config/3_distribution.yaml
2. Reduce ambiguous repository-internal CALLS classifications, prioritizing training, environment, feature, execution, reward, campaign, and evaluation paths.
3. Make verify-critical fail when a critical config that is expected to be traceable has no validated consumer path.

This work improves trust in graph-assisted impact analysis. It is engineering infrastructure and must not block thesis training or experiment execution unless the new checks reveal a real configuration or execution-path defect.

---

## Repository Context

Remote repository:

~~~text
code.triton.aalto.fi
/scratch/work/nguyenl37/Aalto_MS_Thesis
~~~

Planning-time HEAD:

~~~text
ce71438
~~~

The implementation agent must record its own full HEAD and dirty-tree state before changing anything.

Existing graph hardening components:

~~~text
.code-review-graph/graph.db
scripts/dev/code_graph/graph_quality.py
tests/test_code_graph_quality.py
tools/crgraph
.agents/rules/code-review-graph.md
docs/architecture/code_graph_trust_hardening_agent_plan.md
outputs/code_graph_quality/
~~~

Current measured state from the first hardening pass:

~~~text
tracked Python files:       738
indexed Python files:       738
tracked shell files:         56
indexed shell files:         56

total CALLS:              47,236
internal candidates:      13,465
internal exact:            8,814
internal ambiguous:        3,352
internal unresolved:       1,299
internal resolution rate:  65.46%
internal ambiguous rate:   24.89%
internal unresolved rate:   9.65%

tracked config files:          48
static config consumers:       40
dynamic config loads:         108
~~~

The three known critical config gaps are:

~~~text
config/quality_experiments.yaml
config/1_baseline.yaml
config/3_distribution.yaml
~~~

They are tracked, but the current companion dependency index does not yet establish their expected consumer chain with sufficient confidence.

---

## Success Criteria

The follow-up is complete when:

- All critical configuration files have an explicit traceability policy.
- Critical configs marked as required-to-resolve have at least one validated consumer path.
- Indirect registry/variant/campaign config selection is represented in the companion dependency index.
- Ambiguous CALLS are reduced using evidence-backed static resolution.
- Ambiguous calls on thesis-critical paths are either resolved or explicitly classified with a defensible reason.
- verify-critical fails when a required critical config loses its expected trace.
- The graph rebuild still indexes all tracked Python and shell files.
- Existing graph-quality tests pass and new targeted tests cover the added logic.
- No unrelated working-tree changes are modified or discarded.
- The final report quantifies improvement and lists remaining ambiguity classes.

---

## Non-Goals

Do not turn this into a general Python type-inference engine.

Do not attempt universal arbitrary YAML/TOML data-flow analysis.

Do not mutate the upstream .code-review-graph/graph.db schema unless a concrete upstream limitation makes that unavoidable.

Do not create guessed CALLS edges for runtime-dispatched behavior.

Do not optimize total node/edge count.

Do not spend effort resolving low-value third-party or builtin calls.

The target is reliable repository-owned impact analysis for thesis-critical paths.

---

# Phase 0 — Capture a Fresh Baseline

Before editing:

~~~bash
git rev-parse HEAD
git status --short
./tools/crgraph status --json
./tools/crgraph quality
./tools/crgraph verify-critical --baseline outputs/code_graph_quality/baseline.json
~~~

Copy or regenerate a follow-up baseline:

~~~text
outputs/code_graph_quality/followup_baseline.json
outputs/code_graph_quality/followup_baseline.md
~~~

The baseline must record:

~~~text
git_commit
dirty_tree
graph_tool_version
nodes
edges
files
CALLS classification counts
internal_resolution_rate
internal_ambiguous_rate
internal_unresolved_rate
config file count
static consumer count
dynamic load count
critical config traceability state
~~~

Do not overwrite evidence needed to compare the first hardening pass against this follow-up.

---

# Phase 1 — Model Indirect Configuration Selection

## Problem

The current config scanner primarily resolves direct literal/path-based loads such as OmegaConf.load with literal or Path-composed config paths.

Some important configs are selected indirectly through mappings, registries, experiment definitions, CLI choices, campaign specs, or variant resolvers.

Those relationships need to appear in the companion dependency index.

## Required relationship model

Extend the configuration dependency model to support at least these relation types:

~~~text
DIRECT_LOAD
REGISTRY_ENTRY
VARIANT_SELECTION
CAMPAIGN_REFERENCE
BASE_CONFIG
OVERLAY_CONFIG
QUALITY_CONFIG
CLI_SELECTION
DYNAMIC_LOAD
~~~

Each relationship record should include:

~~~text
config_path
relation_type
source_file
source_symbol
line
selector_or_key
consumer_or_next_hop
confidence
resolution_reason
~~~

Suggested confidence vocabulary:

~~~text
exact
high
medium
dynamic
~~~

Do not label an indirect relationship exact unless the selector-to-path mapping is statically deterministic.

---

# Phase 2 — Trace the Three Known Config Gaps

## A. config/1_baseline.yaml

Find every location that:

- directly references the filename,
- maps a strategy/variant name to it,
- uses a baseline identifier that deterministically selects it,
- merges it with config/base/default.yaml,
- restores it from saved run metadata,
- launches training/evaluation from it.

Expected trace shape:

~~~text
config/1_baseline.yaml
→ registry / variant selector
→ merged training config
→ feature/environment/model construction
~~~

Acceptance:

- At least one evidence-backed consumer chain exists.
- The chain reaches a training or evaluation configuration resolver.
- If the file is intentionally obsolete or compatibility-only, record that state explicitly rather than inventing an active consumer.

## B. config/3_distribution.yaml

Trace:

~~~text
config/3_distribution.yaml
→ distribution strategy/variant registry
→ config merge
→ feature/strategy construction
→ training/evaluation consumer
~~~

Confirm whether it is selected through:

- experiment registry,
- strategy mapping,
- campaign override,
- variant resolver,
- audit/evaluation tooling.

Acceptance:

- The companion index identifies the selector that makes config/3_distribution.yaml reachable.
- The final trace must distinguish production/training consumers from research-only scripts.

## C. config/quality_experiments.yaml

Trace:

~~~text
config/quality_experiments.yaml
→ quality experiment loader/registry
→ experiment selection
→ validation/training/evaluation path
~~~

Acceptance:

- The report identifies whether this file is active runtime configuration, quality-gate metadata, research-only configuration, or legacy data.
- If active, at least one validated consumer path is required.
- If inactive, mark it tracked_but_inactive and define the evidence for that conclusion.

---

# Phase 3 — Add Registry and Mapping Extraction

Extend the AST analysis to recognize statically resolvable config registries.

Minimum supported patterns:

~~~python
CONFIGS = {
    "baseline": "config/1_baseline.yaml",
    "distribution": "config/3_distribution.yaml",
}
~~~

~~~python
STRAT_YAML = {
    "po3_ifvg": "config/2_po3_ifvg.yaml",
    "distribution": "config/3_distribution.yaml",
}
~~~

Also support:

- Path("config") / mapping[key]
- registry[name]
- mapping.get(name)
- aliases of module-level registries
- nested literal dictionaries
- module-level constants where practical

For every extracted mapping, preserve:

~~~text
mapping_name
selector_key
resolved_path
definition_file
definition_line
read_sites
confidence
~~~

Do not execute repository code to infer registries. Use AST/static evidence only.

---

# Phase 4 — Add Config Reachability States

Every critical config should end in exactly one high-level state:

~~~text
directly_traceable
indirectly_traceable
dynamic_but_expected
tracked_but_inactive
unresolved
~~~

Definitions:

- directly_traceable: direct static load from repository code.
- indirectly_traceable: selected through a static registry/mapping/campaign/variant chain.
- dynamic_but_expected: consumer exists but runtime input prevents exact static path resolution.
- tracked_but_inactive: evidence shows the config is retained but not part of active execution.
- unresolved: no defensible consumer/reachability explanation.

Only unresolved should automatically fail strict critical-config validation unless policy explicitly permits another state.

Add this state to:

~~~text
outputs/code_graph_quality/config_dependencies.json
outputs/code_graph_quality/final.json
outputs/code_graph_quality/final.md
~~~

---

# Phase 5 — Make Critical Config Policy Explicit

Replace the current tuple-only semantics with explicit policy metadata.

Suggested policy structure:

~~~text
pyproject.toml
  required: directly_traceable or indirectly_traceable

config/base/default.yaml
  required: directly_traceable or indirectly_traceable

config/experiments.yaml
  required: directly_traceable or indirectly_traceable

config/quality_experiments.yaml
  allowed: directly_traceable, indirectly_traceable, dynamic_but_expected,
           or tracked_but_inactive, depending on source evidence

config/1_baseline.yaml
  required: directly_traceable or indirectly_traceable

config/2_po3_ifvg.yaml
  required: directly_traceable or indirectly_traceable

config/3_distribution.yaml
  required: directly_traceable or indirectly_traceable

config/features/full_po3_mtf.yaml
  required: directly_traceable or indirectly_traceable
~~~

The exact states must be decided from repository evidence during implementation.

Do not weaken a policy merely to make the check pass.

---

# Phase 6 — Strengthen verify-critical

## Current weakness

The present verification succeeds when critical configs are tracked even if some lack a resolved static consumer.

## Required behavior

verify-critical should fail when:

- a required critical config is missing,
- a required critical config is untracked,
- its reachability state is not allowed by policy,
- a required registry mapping disappears,
- a critical config chain points only to missing files,
- the consumer chain terminates before reaching an expected resolver/consumer class.

Add failure details such as:

~~~text
config
expected_states
actual_state
evidence
missing_hop
source_file
line
~~~

The CLI must produce an actionable message and exit non-zero.

Example:

~~~text
FAIL config/3_distribution.yaml
state=unresolved
expected={directly_traceable, indirectly_traceable}
no registry/load chain reaches training/configuration/variants.py
~~~

---

# Phase 7 — Decompose the 3,352 Ambiguous Internal Calls

Before trying to resolve them, measure the ambiguity classes.

Produce:

~~~text
outputs/code_graph_quality/ambiguous_call_breakdown.json
outputs/code_graph_quality/ambiguous_call_breakdown.md
~~~

Group ambiguous calls by likely root cause:

~~~text
instance_method
class_method
static_method
inherited_method
reexport
import_alias
factory_return
protocol_or_interface
same_short_name_across_modules
receiver_type_unknown
dynamic_dispatch
other
~~~

For each category report:

~~~text
count
percent_of_ambiguous
top_target_names
top_source_files
top_candidate_modules
critical_path_count
sample_edges
~~~

Also split by repository area:

~~~text
training
simulation/environment
features
configuration
experiments/campaigns
evaluation
learning
live
other
~~~

The implementation priority is based on critical-path impact, not global count alone.

---

# Phase 8 — Resolve Same-Module and Import-Qualified Ambiguity

Implement the safest high-confidence rules first.

## Rule A — Same module preference

When:

- an unresolved call target name has multiple graph candidates,
- exactly one candidate lives in the caller's source module,
- source syntax is compatible with that candidate,

resolve/classify it as evidence-backed exact.

Do not apply when receiver syntax clearly points elsewhere.

## Rule B — Explicit imported symbol

For from-import direct symbols and aliases, prefer the imported symbol when exactly one graph node matches the imported origin.

## Rule C — Module-qualified calls

For module aliases such as mod.thing(), resolve against the imported module plus symbol when exactly one candidate matches.

Acceptance:

- Every new exact classification includes resolution_reason.
- No rule converts multiple plausible targets into an exact target.

---

# Phase 9 — Resolve self/cls Instance and Class Method Calls

Use enclosing class context.

For self.method() and cls.method(), resolution order is:

1. method on the current class,
2. method on a statically known base class,
3. unique inherited candidate,
4. remain ambiguous.

Do not infer the runtime class of arbitrary object receivers.

Capture:

~~~text
receiver_form
enclosing_class
candidate_class
inheritance_path
confidence
~~~

This phase should reduce common method-name collisions such as load, save, step, run, build, resolve, validate, update, from_config, and to_dict without globally picking by short name.

---

# Phase 10 — Add Inheritance-Aware Resolution

Build a repository-owned class hierarchy from:

- AST class definitions,
- import aliases,
- graph INHERITS edges when reliable.

For unresolved self.method(), super().method(), and cls.method():

- resolve only when one valid implementation exists along the statically known hierarchy,
- remain ambiguous when multiple mixins/base classes expose competing implementations,
- record conflicts when graph and AST evidence disagree.

---

# Phase 11 — Resolve Re-exports

Inspect repository __init__.py files and simple import re-exports.

Support:

- from .module import Symbol
- explicit __all__ entries
- imported aliases exposed at package level

Represent a re-export map:

~~~text
public_name
public_module
origin_module
origin_symbol
source_file
line
~~~

Use this map to disambiguate calls imported from package APIs.

Do not infer wildcard re-exports unless static evidence is explicit.

---

# Phase 12 — Handle Factory-Selected Components Conservatively

Factories are important in this repo but can become unsafe if over-resolved.

Recognize statically deterministic factory maps and selector candidate sets.

The graph-quality layer may record:

~~~text
factory_candidates
selector
candidate_set
~~~

Do not classify a runtime-selected method as internal_exact unless the selector is statically fixed at the call site.

For candidate sets that are known but not singular, consider a more informative ambiguity subtype such as internal_ambiguous_known_set.

---

# Phase 13 — Critical-Path Ambiguity Budget

Create an explicit list of critical roots/areas.

At minimum:

~~~text
configuration/loader.py
training/configuration/
training/runner/
features/pipeline/
simulation/environment/
learning/rl/
experiments/campaigns/
evaluation/rollout/
evaluation/reanalysis/
~~~

For ambiguous calls originating from these paths:

- attempt all high-confidence resolvers,
- retain unresolved evidence,
- produce a dedicated count.

Add metrics:

~~~text
critical_internal_candidates
critical_internal_exact
critical_internal_ambiguous
critical_internal_unresolved
critical_internal_resolution_rate
~~~

This metric is more important than forcing global repository resolution toward 100%.

---

# Phase 14 — Regression Thresholds

Do not define an arbitrary target such as 95% global resolution.

Required hard gates:

~~~text
tracked Python coverage == 100%
tracked shell coverage == 100%
critical configs satisfy policy
critical symbols present
no unexplained loss of previously exact CALLS
critical-path internal resolution does not regress
~~~

Recommended improvement gates:

~~~text
global internal_ambiguous < follow-up baseline
critical-path internal_ambiguous < follow-up baseline
internal_resolution_rate >= follow-up baseline
~~~

If an ambiguity cannot be safely resolved, leaving it explicit is preferable to a false exact relationship.

---

# Phase 15 — Expand Tests

Extend tests/test_code_graph_quality.py.

Minimum new config-tracing tests:

~~~text
literal config registry
registry selected by subscript
registry selected by get()
aliased registry name
Path config plus registry key
campaign mapping to config
variant mapping to config
inactive critical config classification
unresolved critical config failure
strict verify-critical success
strict verify-critical failure
~~~

Minimum new CALLS-resolution tests:

~~~text
same-module unique candidate
same-module multiple candidates remains ambiguous
from-import direct symbol
from-import alias
module import alias
self.method current class
cls.method current class
inherited self.method
multiple inheritance conflict remains ambiguous
package re-export
explicit __all__ re-export
static factory with one candidate
dynamic factory with multiple candidates remains ambiguous
~~~

Add negative tests that prove the resolver refuses unsafe guesses.

---

# Phase 16 — Add Focused Fixture Modules

Avoid constructing every ambiguity test through raw SQLite rows alone.

Create lightweight fixtures if useful under:

~~~text
tests/fixtures/code_graph/
~~~

Suggested fixture modules:

~~~text
imports.py
methods.py
inheritance.py
reexports/
factories.py
config_registry.py
~~~

Fixtures should be small and deterministic.

Tests should validate repository-owned extraction/classification behavior, not upstream code-review-graph internals.

---

# Phase 17 — Improve Reports

The final Markdown report should include:

~~~text
1. source coverage
2. CALLS classification
3. critical-path CALLS classification
4. ambiguity breakdown
5. config reachability states
6. strict critical-config policy results
7. unresolved/dynamic config loads
8. before/after metrics
9. remaining limitations
~~~

Add a compact critical-config table with real evidence only.

Required columns:

~~~text
Config
State
Selector/Loader
Principal Consumer
Evidence
Result
~~~

---

# Phase 18 — CLI Integration

Keep the existing commands:

~~~bash
./tools/crgraph quality
./tools/crgraph verify-critical
~~~

Possible additions:

~~~bash
./tools/crgraph quality --ambiguity-breakdown
./tools/crgraph quality --critical-only
./tools/crgraph verify-critical --strict-configs
~~~

Prefer extending the existing Python utility instead of creating many single-purpose scripts.

Preserve compatibility with the system python3 available on Triton where practical because tools/crgraph currently invokes python3.

Also run tests under the project Python 3.12 environment.

---

# Phase 19 — Verify Thesis-Critical Flows Again

After changing resolution logic, re-verify source versus graph/companion traces for:

## Training

~~~text
campaign config
→ variant/config selection
→ merged config
→ feature build
→ environment creation
→ model construction
→ training
~~~

## Environment/action/reward

~~~text
policy action
→ action decoding
→ SL/TP/risk
→ execution
→ TradingEnv.step
→ reward
→ guardrails
~~~

## Feature variants

~~~text
baseline
PO3/IFVG
distribution
full MTF
ablation variants
~~~

## Evaluation

~~~text
checkpoint/config restore
→ evaluation env
→ inference
→ execution
→ metrics
→ thesis evidence
~~~

The purpose is to ensure new resolution rules improve navigation without introducing false paths.

---

# Phase 20 — Verification Commands

Run:

~~~bash
python3 -m py_compile scripts/dev/code_graph/graph_quality.py

.venv-x86/bin/python -m ruff check \
  scripts/dev/code_graph/graph_quality.py \
  tests/test_code_graph_quality.py

.venv-x86/bin/python -m pytest -q \
  tests/test_code_graph_quality.py

./tools/crgraph build
./tools/crgraph status --json
./tools/crgraph quality
./tools/crgraph verify-critical \
  --baseline outputs/code_graph_quality/followup_baseline.json
~~~

Then generate the comparison report.

Inspect:

~~~bash
git diff --stat
git diff -- \
  scripts/dev/code_graph/graph_quality.py \
  tests/test_code_graph_quality.py \
  tools/crgraph \
  .agents/rules/code-review-graph.md

git status --short
~~~

Do not modify or discard unrelated working-tree changes.

---

# Required Before/After Metrics

The implementation report must contain:

~~~text
metric                              before        after
--------------------------------------------------------
tracked Python coverage
tracked shell coverage
total CALLS
internal candidates
internal exact
internal ambiguous
internal unresolved
internal resolution rate
critical-path internal exact
critical-path internal ambiguous
critical-path internal unresolved
critical-path resolution rate
static config consumers
indirect config relations
dynamic config loads
critical configs passing policy
~~~

Also report ambiguity reductions by root-cause category.

---

# Definition of Done

The follow-up is complete when all of these are true:

- config/1_baseline.yaml has a validated reachability state.
- config/3_distribution.yaml has a validated reachability state.
- config/quality_experiments.yaml has a validated reachability state.
- Every critical config has an explicit policy.
- verify-critical enforces that policy.
- Missing required config traceability causes non-zero exit.
- Registry/variant/campaign config selection is represented.
- Ambiguous CALLS have a root-cause breakdown.
- High-confidence same-module/import/receiver/inheritance/re-export rules are implemented.
- False exact resolution is avoided.
- Critical-path ambiguity is separately measured.
- Global internal resolution does not regress.
- Critical-path internal resolution does not regress.
- 100% tracked Python and shell coverage remains.
- Focused unit tests cover positive and negative cases.
- Ruff passes.
- The full graph rebuild succeeds.
- Final quality and comparison reports are generated.
- Known remaining dynamic relationships are documented.
- Unrelated working-tree changes remain untouched.

---

# Agent Execution Order

~~~text
P0   Fresh follow-up baseline
P1   Trace the three critical config gaps manually
P2   Add registry/mapping extraction
P3   Add config reachability states
P4   Add explicit critical-config policy
P5   Make verify-critical enforce config policy
P6   Produce ambiguity root-cause breakdown
P7   Resolve import/same-module ambiguity
P8   Resolve self/cls methods
P9   Add inheritance-aware resolution
P10  Add re-export resolution
P11  Add conservative factory candidate handling
P12  Add critical-path ambiguity metrics
P13  Expand tests and negative cases
P14  Rebuild graph and re-verify critical flows
P15  Generate before/after and remaining-limitations report
~~~

---

# Stop Conditions

Stop and preserve ambiguity rather than guessing when:

- receiver type cannot be established statically,
- multiple inheritance gives multiple plausible implementations,
- a factory selector is runtime-dependent,
- a config path is constructed from uncontrolled runtime input,
- a registry value cannot be statically resolved,
- package re-export provenance is ambiguous,
- source and graph evidence disagree.

Record such cases with reason and evidence.

The correct outcome for genuinely dynamic behavior is an explicit unresolved/dynamic classification, not a fabricated exact relationship.

---

# Final Deliverables

Expected implementation outputs:

~~~text
scripts/dev/code_graph/graph_quality.py
tests/test_code_graph_quality.py
tools/crgraph
.agents/rules/code-review-graph.md

outputs/code_graph_quality/followup_baseline.json
outputs/code_graph_quality/followup_baseline.md
outputs/code_graph_quality/ambiguous_call_breakdown.json
outputs/code_graph_quality/ambiguous_call_breakdown.md
outputs/code_graph_quality/config_dependencies.json
outputs/code_graph_quality/final.json
outputs/code_graph_quality/final.md
outputs/code_graph_quality/comparison.json
~~~

Plan document:

~~~text
docs/architecture/code_graph_followup_resolution_plan.md
~~~

The final implementation report must state:

~~~text
configs newly traced
config reachability states
critical policy results
ambiguity classes reduced
before/after CALLS metrics
critical-path metrics
tests run
graph rebuild result
remaining dynamic/unresolved classes
known limitations
~~~
