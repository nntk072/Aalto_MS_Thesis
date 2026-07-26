# Code-review-graph

A structured review method: treat a change as a **graph** and walk it deliberately, reviewing by node priority — not by reading files top to bottom.

## When to use

- Reviewing a diff, PR, or local change touching >1 file or shared symbols.
- Assessing blast radius before/after an edit.
- Skip for single-file typo/trim edits.

## 1. Build the node set (the diff)

- `git diff --stat` (or `git log -1 --stat`) → list changed files/symbols as **nodes**.
- State a one-line goal: what is this change trying to do?
- Nodes = changed functions / classes / symbols / files.

## 2. Discover edges (relationships)

For each changed node, find edges with **scoped** greps (path-limited, capped — no repo-wide dumps):

- **Callers (fan-in):** who calls this symbol? → blast radius.
- **Callees (fan-out):** what does the change now call / import / include?
- **Types / contracts:** signatures, interfaces, base classes, public API touched?
- **Tests:** `Glob` `**/test*` / `**/py_ut/**` + `Grep` module or symbol name near changed code.

## 3. Prioritize hot spots

- **High fan-in** (many callers) or **shared / public API** → deep review: one targeted `Read` slice per hot spot.
- **Leaf / internal nodes** → diff + summary only.
- Always deep: concurrency, error paths, resource lifecycle, security boundaries.
- Risk heuristic: many files, shared headers, missing tests → deeper walk.

## 4. Walk by depth, not breadth

- **Depth-1 first:** direct callers / callees / tests of changed nodes.
- Expand to **depth-2** only when a depth-1 node looks risky or a contract changed.
- Hard cap: ≤5 locate steps (stat/diff/grep/glob) before `Read`; then slices only (~80–150 lines).

## 5. Report as a graph, not a file dump

- Short summary: what changed, blast radius, risk level.
- `path:line` citations for each finding.
- Flag: untested changed public symbols, broken contracts, missing error handling, new fan-in on fragile code.
- Do not paste whole files; show small hunks only.

## Budget

- Graph expansion is bounded: stop when a subgraph is all leaves or low-risk.
- Spend tokens on hot spots; diff + summary for low-risk subgraphs.
- Re-walk only edges that changed; do not re-derive the whole graph each turn.