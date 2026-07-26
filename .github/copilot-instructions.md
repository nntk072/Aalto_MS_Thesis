# Aalto_MS_Thesis project agent rules

## Token-efficient context and replies

Rules add to every request — keep behavior tight; do not compensate with long chat.

### Before reading files

- **Locate first**: use search with narrow patterns; then open only likely files.
- **Slice reads**: default ~80–150 lines around the target; widen only when needed.
- **Skip unless needed**: `node_modules`, `dist`, `build`, `.venv`, `*-lock.*`, `minified assets, large fixtures, full migration histories.
- **Prefer diffs**: `git diff -- <path>` over reading old + new full files.
- **No duplicate work**: do not re-read or re-run what is already in the thread.

### Scope

- Touch only files required for the task; no drive-by refactors or doc sweeps.
- For repo-wide questions, sample key paths — do not read every similar file.

### Assistant output

- Answer first; avoid restating the question and long preambles.
- Summarize tool results in prose; do not paste large logs or directory listings.
- Show code via line citations or small hunks — not whole files after an edit.
- Match depth to the ask; do not repeat explanations across turns.

## Token-efficient shell and tool output

Use compact commands so terminal output stays small.

### Prefer tools over shell

- **Search code**: use search with tight patterns — not `grep -r` / `rg` dumping whole repos.
- **Find files**: use file globs — not `find .` without depth limits.
- **Read files**: open with offset/limit — not `cat`, `head -n 5000`, or generated/vendor trees.

### Git (compact first)

1. `git status -sb` or `git status --porcelain=v1`
2. `git log --oneline -20` (increase only when debugging history)
3. `git diff --stat` or `git diff -U0 -- <paths>` before full hunks
4. `git show --stat <rev>` before `git show -p`
5. `git diff main...HEAD --stat` then path-limited diffs

### Search in terminal (only when tools are insufficient)

- Ripgrep: `rg -n --max-columns 120 -m 80 <pattern> <path>`
- Never run unbounded `rg` / `grep -r` on `.` or large dirs (`node_modules`, `dist`, `.git`).

### RTK (shell proxy)

Always prefix shell commands with `rtk` to minimize token consumption. Run `rtk gain` to see savings.

### Builds, tests, installs

- Run the narrowest command (single package, single test file, `-q` / quiet).
- On success: one-line confirmation; do not paste full logs.
- On failure: failing test names, error lines, and ~20 lines of context only.

### Docker / k8s / cloud CLIs

- `docker ps --format '...'`, `kubectl get pods -o wide` over wide defaults.
- Avoid `kubectl logs` without `--tail=50` (or smaller).

### JSON / logs

- `jq 'keys'` or targeted `jq '.field'` — not multi-MB pretty-printed JSON.
- Logs: filter (`rg -i 'error|warn|fail'`) and cap lines.

### General

- Do not read secrets (`.env`, credentials).
- Batch independent commands in parallel; avoid redundant runs.
- Explain conclusions in prose; do not quote large output.

## Code-review-graph

A structured review method: treat a change as a **graph** and walk it deliberately, reviewing by node priority — not by reading files top to bottom.

### When to use

- Reviewing a diff, PR, or local change touching >1 file or shared symbols.
- Assessing blast radius before/after an edit.
- Skip for single-file typo/trim edits.

### 1. Build the node set (the diff)

- `git diff --stat` (or `git log -1 --stat`) → list changed files/symbols as **nodes**.
- State a one-line goal: what is this change trying to do?

### 2. Discover edges (relationships)

Scoped searches only — no repo-wide dumps:

- **Callers (fan-in):** who calls this symbol? → blast radius.
- **Callees (fan-out):** what does the change call / import / include?
- **Types / contracts:** signatures, interfaces, base classes, public API touched?
- **Tests:** glob `**/test*` + search module or symbol near changed code.

### 3. Prioritize hot spots

- **High fan-in** or **shared / public API** → deep review: one targeted read per hot spot.
- **Leaf / internal nodes** → diff + summary only.
- Always deep: concurrency, error paths, resource lifecycle, security boundaries.

### 4. Walk by depth, not breadth

- **Depth-1 first:** direct callers / callees / tests of changed nodes.
- **Depth-2** only when a depth-1 node looks risky or a contract changed.
- Hard cap: ≤5 locate steps before opening files; slices only (~80–150 lines).

### 5. Report as a graph

- Short summary: what changed, blast radius, risk level.
- `path:line` citations for each finding.
- Flag: untested changed public symbols, broken contracts, missing error handling.
- Do not paste whole files; show small hunks only.

### Budget

- Stop when a subgraph is all leaves or low-risk.
- Spend tokens on hot spots; diff + summary for low-risk subgraphs.