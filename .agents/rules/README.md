# Agent rules index & activation map (Aalto_MS_Thesis)

Single source of truth for all coding agents in this repo (Cursor, Cline, Windsurf, Kilo Code, Antigravity, Mistral Vibe). Rules live here — not in `.cursor/` or `.kilocode/` (local IDE folders; gitignored). Entry point for Cursor: root `AGENTS.md` and `.cursorrules`.

## Rule files

| File | Topic | Always on? |
|---|---|---|
| [antigravity-rtk-rules.md](antigravity-rtk-rules.md) | RTK CLI proxy (`rtk` prefix) | Yes |
| [token-efficient-context-replies.md](token-efficient-context-replies.md) | Read/context/reply discipline | Yes |
| [token-efficient-shell.md](token-efficient-shell.md) | Compact terminal commands | On shell use |
| [git-commit-rules.md](git-commit-rules.md) | Commit format + pre-commit checks | On commit/PR |
| [python-coding-standards.md](python-coding-standards.md) | Python style & structure | On `.py` edits |
| [code-review-graph.md](code-review-graph.md) | Graph-based change review | On multi-file review |
| [testing-rules.md](testing-rules.md) | TDD, AAA, coverage ≥80% | On feature/bugfix work |
| [security-rules.md](security-rules.md) | Secret & input safety checklist | Always at commit; deep on sensitive code |
| [development-workflow.md](development-workflow.md) | Research-first → plan → TDD → review → commit | On non-trivial features |
| [ci-verification.md](ci-verification.md) | **Blocking** full-tree `ruff` + `mypy .` (not path-scoped); scoped pytest before commit/push | On commit/push |

## Activation flow

```
request
  ├─ ALWAYS: token-efficient-context-replies + RTK prefix (antigravity-rtk-rules)
  ├─ running commands?        → token-efficient-shell (+ RTK)
  ├─ non-trivial feature?     → development-workflow
  │                             ├─ research/reuse first
  │                             ├─ plan (blast radius via code-review-graph)
  │                             ├─ TDD → testing-rules
  │                             └─ self-review via code-review-graph
  ├─ editing .py files?       → python-coding-standards (via token-efficient reads)
  ├─ reviewing a diff >1 file → code-review-graph
  ├─ editing code?            → scoped ruff + pytest on touched paths only
  └─ committing / push / PR   → git-commit-rules + ci-verification + security-rules
                                  └─ BLOCKING: ruff format --check . → ruff check .
                                     → mypy . (FULL TREE — never path-scoped mypy)
                                     → pytest <relevant paths> -q
                                     (NOT full tests/ -v; that is CI/Orchestra);
                                     fix format drift with `ruff format .` first;
                                     scoped ruff/mypy while editing ≠ commit gate
```

## Priority

Layered principle: **specific overrides general**.

1. **RTK** wraps every command — never run raw commands.
2. **Token-efficiency** (context + shell) governs all I/O volume.
3. **Domain rules** (Python standards, testing, security, commit rules, review graph, workflow) apply by activity, layered on top of 1–2.
4. When two rules conflict, the more specific one for the current activity wins; when still ambiguous, ask before proceeding.

