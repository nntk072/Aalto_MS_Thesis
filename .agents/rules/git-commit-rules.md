# Git commit rules

## Commit message format

```
<subject>

[body]

[footer]
```

### Subject line (REQUIRED)
- Imperative mood, **50 characters max**, no trailing period.
- Start with an action verb (Add, Fix, Refactor, Update, Remove, Implement).

### Body (OPTIONAL but RECOMMENDED)
- One paragraph is usually enough; every line **100 characters max**.
- Describe **what this commit delivers and why**.

### Footer (OPTIONAL for agent-assisted commits)
- `Ref:` on its own line.
- Trailers: `%AI=MISTRAL_VIBE %AIRATIO=<0-100>`.

Never add secrets or `Co-authored-by` (auto-added).

## Before committing

```bash
git status -sb
git diff --stat
ruff check <changed_paths>
ruff format --diff <changed_paths>
mypy <changed_paths>
pytest tests/ -v --tb=short
```

Never force-push to main/master without explicit approval.

## Relations

- Activates: whenever creating or amending any commit; before opening a PR.
- Pairs with [code-review-graph.md](code-review-graph.md) (review before commit) and [python-coding-standards.md](python-coding-standards.md) (pre-commit checks enforce it).
