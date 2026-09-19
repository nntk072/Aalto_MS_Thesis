# CI verification gate

Applies to **all agents** (Cursor, Orchestra, Cline, Windsurf, Kilo, Vibe, etc.).
Canonical commands match `.github/workflows/ci.yml` and `orchestra/ci_gate.py`
(`CI_CHECKS`).

## Hard rule — commit / push / open PR

**Do not** `git commit`, `git push`, or open a PR until this gate passes locally.
If any step fails, fix and re-run; do not skip with `--no-verify` unless the user
explicitly requests it.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest tests/ -v
```

### Format failures (common CI red)

If `ruff format --check .` reports files that would be reformatted:

1. Run `uv run ruff format .` (or only the listed paths).
2. Re-run `uv run ruff format --check .` until it exits 0.
3. Include the formatting diff in the commit (do not push unformatted code).

Never treat “lint already passed earlier in the chat” as a substitute — re-run the
gate on the **final** tree you are about to commit.

### Minimal gate when the user asks only for lint/types

When the user asks to verify before commit/push but not to run the full suite:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
```

Still run full `pytest tests/ -v` when opening a PR or when `ci-verification` /
Orchestra phase 9 applies.

## Local work (implementation)

While editing, prefer scoped checks only — do **not** run the full gate after every
edit:

```bash
uv run ruff format --check <touched paths>
uv run ruff check <touched paths>
uv run pytest <relevant test files> -q
```

## Out of scope unless explicitly requested

- Nightly workflow subsets (`pytest -m slow`, integration-only paths)
- `pytest -m "not slow"` as a substitute for the merge-bar suite
- Treating scoped `mypy`/`pytest` as the GitHub CI bar

## Test discipline

- Mock methods with `monkeypatch.setattr`, not `instance.method = ...` (mypy
  `method-assign`).
- Orchestra router/health tests on CI: use `stub_clis` fixture
  (`tests/test_orchestra/conftest.py`).

## Relations

- Pre-commit checklist: [git-commit-rules.md](git-commit-rules.md)
- Feature workflow: [development-workflow.md](development-workflow.md)
- Agent index: [README.md](README.md)
- Enforced in GitHub CI and Orchestra phase 9 (`orchestra/ci_gate.py`)
