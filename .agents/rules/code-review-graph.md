# Code-review-graph

A structured review method: treat a change as a **graph** and walk it deliberately, reviewing by node priority — not by reading files top to bottom.

## When to use

- Reviewing a diff, PR, or local change touching >1 file or shared symbols.
- Assessing blast radius before/after an edit.
- Skip for single-file typo/trim edits.

## 1. Build the node set (the diff)

- `git diff --stat` (or `git log -1 --stat`) → list changed files/symbols as **nodes**.
- State a one-line goal: what is this change trying to do?

## 2. Discover edges (relationships)

Scoped searches only — no repo-wide dumps:

- **Callers (fan-in):** who calls this symbol? → blast radius.
- **Callees (fan-out):** what does the change call / import / include?
- **Types / contracts:** signatures, interfaces, base classes, public API touched?
- **Tests:** glob `**/test*` + search module or symbol near changed code.

## 3. Prioritize hot spots

- **High fan-in** or **shared / public API** → deep review: one targeted read per hot spot.
- **Leaf / internal nodes** → diff + summary only.
- Always deep: concurrency, error paths, resource lifecycle, security boundaries.

## 4. Walk by depth, not breadth

- **Depth-1 first:** direct callers / callees / tests of changed nodes.
- **Depth-2** only when a depth-1 node looks risky or a contract changed.
- Hard cap: ≤5 locate steps before opening files; slices only (~80–150 lines).

## 5. Report as a graph

- Short summary: what changed, blast radius, risk level.
- `path:line` citations for each finding.
- Flag: untested changed public symbols, broken contracts, missing error handling.
- Do not paste whole files; show small hunks only.

## Budget

- Stop when a subgraph is all leaves or low-risk.
- Spend tokens on hot spots; diff + summary for low-risk subgraphs.

## Relations

- Activates: reviewing a diff/PR/local change touching >1 file or shared symbols; skip for single-file typo edits.
- Uses [token-efficient-context-replies.md](token-efficient-context-replies.md) read discipline and [token-efficient-shell.md](token-efficient-shell.md) git commands; findings feed [git-commit-rules.md](git-commit-rules.md) pre-commit checks.

## Graph trust and configuration dependency checks

Use the persistent graph for navigation, blast-radius discovery, and candidate impact analysis. Verify important execution paths and configuration dependencies against source and tests before relying on a graph-only conclusion.

Repository-owned quality commands:

```bash
./tools/crgraph quality
./tools/crgraph verify-critical --baseline outputs/code_graph_quality/baseline.json
```

The quality audit classifies `CALLS` edges as `internal_exact`, `internal_ambiguous`, `internal_unresolved`, `external_library`, `python_builtin`, or `dynamic_or_indeterminate`. The principal quality metric is the resolution rate among repository-internal call candidates, rather than exact calls divided by all calls.

Configuration analysis is stored as a companion index under `outputs/code_graph_quality/config_dependencies.json`. This keeps YAML/TOML dependency evidence separate from the upstream graph database schema and records static load sites plus dynamic loads that still require source review.

A resolved `TESTED_BY` relationship means only that the recorded relation points to an indexed node. It does not establish line, branch, path, or behavioral test coverage.

### Strict config reachability and ambiguity policy

`verify-critical` is fail-closed for critical configuration reachability. Each critical config has an allowed reachability state: direct load, evidence-backed indirect registry/default-parameter selection, or an explicitly documented inactive state. A tracked file alone is no longer sufficient.

`config/1_baseline.yaml` is intentionally `tracked_but_inactive`: baseline training uses `config/base/default.yaml` and the baseline branch does not auto-merge the symmetry file. `config/3_distribution.yaml` and `config/quality_experiments.yaml` must remain traceable through their active strategy/quality registry paths.

CALLS quality reports now include both global and thesis-critical-path resolution. Evidence-backed same-module, explicit-import/re-export, `self`/`cls`, and unique direct-base resolutions may be promoted to `internal_exact`. Generic container/file methods with unknown receivers remain dynamic rather than being matched to unrelated repository methods with the same short name.

Use `outputs/code_graph_quality/ambiguous_call_breakdown.{json,md}` to inspect remaining ambiguity classes. Preserve ambiguity whenever static evidence does not identify one target.
