# CI verification gate

Applies to **all agents** (Cursor, Orchestra pipeline, Cline, Vibe, etc.).

## Local work (implementation)

Run only checks that cover the changed behavior. Do not run the full suite,
`mypy .`, or `ruff` on `.` after every edit.

```bash
uv run ruff format --check <touched paths>
uv run ruff check <touched paths>
uv run pytest <relevant test files> -q
```

## Standard gate (commit / push)

Same as `.github/workflows/ci.yml` jobs `code-formatting` and `ut-venv`.
Single source of truth in code: `orchestra/ci_gate.py` (`CI_CHECKS`).

Run this **on commit/push**. Orchestra phase 9 (verification) runs the same gate
after implementation; implementer/fixer agents must not duplicate it.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest tests/ -v
```

## Out of scope unless explicitly requested

- Nightly workflow subsets (`pytest -m slow`, integration-only paths)
- `pytest -m "not slow"` as a substitute for the merge-bar suite
- Treating scoped `mypy`/`pytest` as the GitHub CI bar

## Test discipline

- Mock methods with `monkeypatch.setattr`, not `instance.method = ...` (mypy `method-assign`).
- Orchestra router/health tests on CI: use `stub_clis` fixture (`tests/test_orchestra/conftest.py`).

## Relations

- Enforced in GitHub CI, on commit/push, and Orchestra phase 9; agent index: `.agents/rules/README.md`
- Pre-commit checklist: [git-commit-rules.md](git-commit-rules.md)
- Feature workflow: [development-workflow.md](development-workflow.md)
