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

**Blocking.** Do not commit or push until these exit 0 (see
[ci-verification.md](ci-verification.md)):

```bash
git status -sb
git diff --stat
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest <relevant test paths> -q
```

If `ruff format --check .` fails: run `uv run ruff format .`, stage the
reformats, then re-run the gate. Do not push with format drift.

Do **not** run `pytest tests/ -v` before every commit — only the tests that cover
this change. Full-suite pytest is GitHub CI / Orchestra phase 9.

Never force-push to main/master without explicit approval.

## Relations

- Activates: whenever creating or amending any commit; before opening a PR.
- Pairs with [code-review-graph.md](code-review-graph.md) (review before commit) and [python-coding-standards.md](python-coding-standards.md) (pre-commit checks enforce it).
