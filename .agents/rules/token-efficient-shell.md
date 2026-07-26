# Token-efficient shell and tool output

Use compact commands so terminal output stays small.

## Prefer tools over shell

- **Search code**: use search with tight patterns — not `grep -r` / `rg` dumping whole repos.
- **Find files**: use file globs — not `find .` without depth limits.
- **Read files**: open with offset/limit — not `cat`, `head -n 5000`, or generated/vendor trees.

## Git (compact first)

1. `git status -sb` or `git status --porcelain=v1`
2. `git log --oneline -20` (increase only when debugging history)
3. `git diff --stat` or `git diff -U0 -- <paths>` before full hunks
4. `git show --stat <rev>` before `git show -p`
5. `git diff main...HEAD --stat` then path-limited diffs

## Search in terminal (only when tools are insufficient)

- Ripgrep: `rg -n --max-columns 120 -m 80 <pattern> <path>`
- Never run unbounded `rg` / `grep -r` on `.` or large dirs (`node_modules`, `dist`, `.git`).

## RTK

Always prefix shell commands with `rtk` as documented in `antigravity-rtk-rules.md`.

## Builds, tests, installs

- Run the narrowest command (single package, single test file, `-q` / quiet).
- On success: one-line confirmation; do not paste full logs.
- On failure: failing test names, error lines, and ~20 lines of context only.

## Docker / k8s / cloud CLIs

- `docker ps --format '...'`, `kubectl get pods -o wide` over wide defaults.
- Avoid `kubectl logs` without `--tail=50` (or smaller).

## JSON / logs

- `jq 'keys'` or targeted `jq '.field'` — not multi-MB pretty-printed JSON.
- Logs: filter (`rg -i 'error|warn|fail'`) and cap lines.

## General

- Do not read secrets (`.env`, credentials).
- Batch independent commands in parallel; avoid redundant runs.
- Explain conclusions in prose; do not quote large output.