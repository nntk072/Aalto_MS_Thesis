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