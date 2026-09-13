"""CI verification gate — mirrors .github/workflows/ci.yml (not nightly)."""

from __future__ import annotations

from dataclasses import dataclass

# Nightly runs integration subsets and ``-m slow`` training tests; orchestra uses
# the standard CI jobs ``code-formatting`` + ``ut-venv`` only.
CI_PYTEST_TIMEOUT_S = 3600.0


@dataclass(frozen=True)
class CiCheck:
    name: str
    argv: list[str]
    timeout: float | None = None


CI_CHECKS: tuple[CiCheck, ...] = (
    CiCheck("format", ["uv", "run", "ruff", "format", "--check", "."]),
    CiCheck("lint", ["uv", "run", "ruff", "check", "."]),
    CiCheck("typecheck", ["uv", "run", "mypy", "."]),
    CiCheck("tests", ["uv", "run", "pytest", "tests/", "-v"], timeout=CI_PYTEST_TIMEOUT_S),
)


def ci_command_lines() -> str:
    """Markdown bullet list of CI commands for agent prompts."""
    return "\n".join(f"- `{' '.join(check.argv)}`" for check in CI_CHECKS)


def gate_passed(results: dict[str, dict[str, object]]) -> bool:
    """True when every CI check entry exists and succeeded."""
    for check in CI_CHECKS:
        entry = results.get(check.name)
        if not entry or not entry.get("success"):
            return False
    return True


def first_failure(results: dict[str, dict[str, object]]) -> str | None:
    for check in CI_CHECKS:
        entry = results.get(check.name, {})
        if not entry.get("success"):
            return check.name
    return None
