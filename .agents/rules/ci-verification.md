# CI verification gate

Applies to **all agents** (Cursor, Orchestra, Cline, Windsurf, Kilo, Vibe, etc.).

Two layers:

| Layer | Who | Commands |
|-------|-----|----------|
| **Agent commit/push gate** | Any coding agent before `git commit` / `git push` | format + lint + mypy (full tree) + **scoped** pytest |
| **Merge CI bar** | GitHub Actions + Orchestra phase 9 (`orchestra/ci_gate.py`) | same + `pytest tests/ -v` |

Agents must **not** run `pytest tests/ -v` by default — it is too heavy. Prefer the
smallest relevant test set for the change.

## Hard rule — agent commit / push

**Do not** `git commit` or `git push` until this gate passes. If any step fails,
fix and re-run; do not skip with `--no-verify` unless the user explicitly requests it.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest <relevant test paths> -q
```

**Full tree is mandatory for format / lint / types.** At commit/push you MUST run
exactly `ruff format --check .`, `ruff check .`, and `mypy .` (the trailing `.`
means the whole project). These are **not** optional and **must not** be replaced by
path-scoped commands.

| Forbidden at commit/push | Required instead |
|--------------------------|------------------|
| `uv run mypy quant_rl/...` or `mypy <touched files>` | `uv run mypy .` |
| `uv run ruff check <touched paths>` only | `uv run ruff check .` |
| `uv run ruff format --check <touched paths>` only | `uv run ruff format --check .` |

Scoped ruff/mypy is allowed **only while editing** (see Local work below). A scoped
pass earlier in the chat does **not** satisfy the commit gate — re-run the full-tree
commands on the final tree, including new/edited `tests/` files (CI typechecks them).

Only **pytest** stays scoped at commit time.

### Choosing relevant tests

- Prefer tests that cover the modules you touched (same package / mirror path under
  `tests/`, plus any smoke tests you added).
- Use graph/`tests_for` when available; otherwise grep or path convention
  (`quant_rl/foo.py` → `tests/**/test_*foo*`).
- Cap to a small set (typically 1–few files). Do **not** expand to the whole tree
  “to be safe.”
- Run `pytest tests/ -v` only when the user explicitly asks, or when you are the
  Orchestra verification phase / reproducing a CI failure.

### Format failures (common CI red)

If `ruff format --check .` reports files that would be reformatted:

1. Run `uv run ruff format .` (or only the listed paths).
2. Re-run `uv run ruff format --check .` until it exits 0.
3. Include the formatting diff in the commit (do not push unformatted code).

Never treat “lint already passed earlier in the chat” as a substitute — re-run the
gate on the **final** tree you are about to commit.

### Lint/types only

When the user asks only for format/lint/types before commit:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
```

## Local work (implementation)

While editing, prefer scoped checks only — do **not** run full-tree ruff/mypy after
every edit:

```bash
uv run ruff format --check <touched paths>
uv run ruff check <touched paths>
uv run pytest <relevant test files> -q
```

When you are about to commit or push, stop using this section and switch to the
**Hard rule** full-tree gate (`mypy .`, not path-scoped mypy).

## Merge CI bar (not the agent default)

GitHub CI and Orchestra phase 9 still run the full suite:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest tests/ -v
```

## Out of scope unless explicitly requested

- Nightly workflow subsets (`pytest -m slow`, integration-only paths)
- `pytest -m "not slow"` as a substitute for the merge-bar suite
- Agents substituting scoped pytest for GitHub’s full suite when debugging CI

## Test discipline

- Mock methods with `monkeypatch.setattr`, not `instance.method = ...` (mypy
  `method-assign`).
- Orchestra router/health tests on CI: use `stub_clis` fixture
  (`tests/test_orchestra/conftest.py`).

## Relations

- Pre-commit checklist: [git-commit-rules.md](git-commit-rules.md)
- Feature workflow: [development-workflow.md](development-workflow.md)
- Agent index: [README.md](README.md)
- Full suite: `.github/workflows/ci.yml`, `orchestra/ci_gate.py`
