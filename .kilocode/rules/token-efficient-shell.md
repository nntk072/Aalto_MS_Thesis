# Token-efficient shell and tool output

Use compact commands and built-in tools so shell output stays small.

## Prefer Kilo tools over shell

- **Search code**: `Grep` with tight include patterns — not `grep -r` / `rg` dumping whole repos.
- **Find files**: `Glob` — not `find .` without depth limits.
- **Read files**: `Read` with `offset` / `limit` — not `cat`, `head -n 5000`, or reading generated/vendor trees.

## Git (compact first, expand only if needed)

1. `git status -sb` or `git status --porcelain=v1`
2. `git log --oneline -20` (increase only when debugging history)
3. `git diff --stat` or `git diff -U0 -- <paths>` before full hunks
4. `git show --stat <rev>` before `git show -p`
5. For branch scope: `git diff main...HEAD --stat` then path-limited diffs

## Search in terminal (only when tools are insufficient)

- Ripgrep: `rg -n --max-columns 120 -m 80 <pattern> <path>`
- Never run unbounded `rg` / `grep -r` on `.` or large dirs (`node_modules`, `dist`, `.git`).

## RTK

Always prefix shell commands with `rtk` to minimize token consumption. See `rtk-rules.md`.

## Builds, tests, installs

- Run the narrowest command (single package, single test file, `-q` / quiet reporters).
- On success: one-line confirmation; do not paste full build/test logs.
- On failure: paste only failing test names, error lines, and ~20 lines of surrounding context — not entire CI output.
- Long runs: rely on exit code; summarize unless the user asked for full logs.

## Docker / k8s / cloud CLIs

- Prefer `docker ps --format '...'`, `kubectl get pods -o wide` over wide default tables.
- Avoid `kubectl logs` without `--tail=50` (or smaller).

## JSON / logs

- Structure: `jq 'keys'` or targeted `jq '.field'` — not pretty-printed multi-MB JSON.
- Logs: filter (`rg -i 'error|warn|fail'`) and cap lines before sharing.

## General

- Do not read secrets (`.env`, credentials); warn if asked to commit them.
- Batch independent read-only commands in parallel; avoid redundant status/diff runs.
- Explain conclusions in prose; do not quote large command output the user did not need.