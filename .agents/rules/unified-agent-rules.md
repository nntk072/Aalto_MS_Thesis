# Unified Agent Rules (Aalto_MS_Thesis)

Single source of truth for all coding agents in this repo (Cline, Windsurf, Kilo Code, Antigravity, Mistral Vibe). Mirrors `.clinerules`, `.windsurfrules`, `.kilocode/rules/`, and the workspace `AGENTS.md`.

---

## 1. RTK - Rust Token Killer

**Usage**: Token-optimized CLI proxy for shell commands. Always prefix shell commands with `rtk` to minimize token consumption.

```bash
rtk git status
rtk pytest tests/ -q
rtk ls src/
rtk grep "pattern" src/
rtk find "*.py" .
rtk docker ps
```

Meta commands:

```bash
rtk gain              # Show token savings
rtk gain --history    # Command history with savings
rtk discover          # Find missed RTK opportunities
rtk proxy <cmd>       # Run raw (no filtering, for debugging)
```


---

## 2. Token-efficient context and replies

Keep behavior tight; do not compensate with long chat.

### Before reading files
- **Locate first**: use search/grep with narrow patterns; then open only likely files.
- **Slice reads**: default ~80-150 lines around the target; widen only when needed.
- **Skip unless needed**: `node_modules`, `dist`, `build`, `.venv`, `*-lock.*`, `uv.lock`, minified assets, large fixtures, full migration histories.
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

---

## 3. Token-efficient shell and tool output

### Prefer tools over shell
- **Search code**: tight-pattern search — not `grep -r` / `rg` dumping whole repos.
- **Find files**: file globs — not `find .` without depth limits.
- **Read files**: offset/limit — not `cat`, `head -n 5000`, or generated/vendor trees.

### Git (compact first)
1. `git status -sb` or `git status --porcelain=v1`
2. `git log --oneline -20` (increase only when debugging history)
3. `git diff --stat` or `git diff -U0 -- <paths>` before full hunks
4. `git show --stat <rev>` before `git show -p`
5. `git diff main...HEAD --stat` then path-limited diffs

### Search in terminal (only when tools are insufficient)
- Ripgrep: `rg -n --max-columns 120 -m 80 <pattern> <path>`
- Never run unbounded `rg` / `grep -r` on `.` or large dirs (`node_modules`, `dist`, `.git`).

### Builds, tests, installs
- Run the narrowest command (single package, single test file, `-q` / quiet).
- On success: one-line confirmation; do not paste full logs.

---

## 4. Git commit rules

**Commit message format:**

```
<subject>

[body]

[footer]
```

#### Subject line (REQUIRED)
- Imperative mood, **50 characters max**, no trailing period.
- Start with an action verb (Add, Fix, Refactor, Update, Remove, Implement).

#### Body (OPTIONAL but RECOMMENDED)
- One paragraph is usually enough; every line **100 characters max**.
- Describe **what this commit delivers and why**.

#### Footer (OPTIONAL for agent-assisted commits)
- `Ref:` on its own line.
- Trailers: `%AI=MISTRAL_VIBE %AIRATIO=<0-100>`.

Never add secrets or `Co-authored-by` (auto-added).

#### Before committing

```bash
git status -sb
git diff --stat
ruff check <changed_paths>
ruff format --diff <changed_paths>
mypy <changed_paths>
pytest tests/ -v --tb=short
```

Never force-push to main/master without explicit approval.

---

## 5. Python coding standards

- PEP 8 + Google-style docstrings + PEP 484 type hints.
- Max line length: 120 characters.
- Tools: Ruff (format/lint), mypy (strict), pytest.


---

## 6. Code-review-graph

A structured review method: treat a change as a **graph** and walk it deliberately, reviewing by node priority — not by reading files top to bottom.

### When to use
- Reviewing a diff, PR, or local change touching >1 file or shared symbols.
- Assessing blast radius before/after an edit.
- Skip for single-file typo/trim edits.

### Steps
1. **Build the node set**: `git diff --stat` → changed files/symbols as nodes; state a one-line goal.
2. **Discover edges** (scoped searches only): callers (fan-in), callees (fan-out), types/contracts touched, tests (`**/test*` near changed code).
3. **Prioritize hot spots**: high fan-in / public API → deep review (one targeted read per hot spot); leaves → diff + summary. Always deep: concurrency, error paths, resource lifecycle, security boundaries.
4. **Walk by depth**: depth-1 first; depth-2 only if a depth-1 node looks risky or a contract changed. Hard cap ≤5 locate steps before opening files; slices ~80-150 lines.
5. **Report as a graph**: summary of changes, blast radius, risk level; `path:line` citations per finding; flag untested changed public symbols, broken contracts, missing error handling; small hunks only.

### Budget
Stop when a subgraph is all leaves or low-risk; spend tokens on hot spots.

Code quality:
- No mutable default args (`def f(x=None): x = x or []`).
- No bare `except:` — catch specific exceptions.
- Use `pathlib.Path` over `os.path`.
- Public API re-exported via `__init__.py`.

Module structure:
- Module: 400 lines max; Function: 60 lines max; Commit: 500 changed lines max.
- Split by concern, each with its own tests.

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

RTK filters and compresses command output before it reaches the LLM context, saving 60-90% tokens on common operations.
