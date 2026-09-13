## Local verification (implementation)

Run from `{{workspace}}`. Cover **only** what you changed:

- `uv run ruff format --check <touched paths>`
- `uv run ruff check <touched paths>`
- `uv run pytest <relevant test files> -q`

Do not run the full CI gate during implementation. Do not use `pytest -m "not slow"`
as a substitute when a targeted test file already exists.

## Full CI gate (commit / push and Orchestra phase 9)

Orchestra verification and git commit/push run this list (`orchestra/ci_gate.py`):

{{ci_commands}}

Implementer and fixer agents must not duplicate that suite. If a CI check already
failed, re-run only the failing command.
