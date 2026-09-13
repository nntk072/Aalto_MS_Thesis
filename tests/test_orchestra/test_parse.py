"""Tests for orchestra output parsers."""

from __future__ import annotations

from pathlib import Path

from orchestra.parse import parse_complexity, parse_verdict
from orchestra.prompts import load_prompt


def test_parse_complexity_ignores_prompt_template() -> None:
    template = load_prompt("triage")
    assert "trivial" in template.lower()
    assert parse_complexity(template) == "medium"


def test_parse_complexity_ignores_example_json_in_template() -> None:
    text = load_prompt("triage") + "\nYOLO mode is enabled.\nQuota exceeded\n"
    assert parse_complexity(text) == "medium"


def test_parse_complexity_reads_model_json() -> None:
    template = load_prompt("triage")
    output = (
        template
        + """
```json
{
  "complexity": "complex",
  "domain": "bugfix",
  "scope": "single-file",
  "risk": "low",
  "summary": "real model output",
  "files_likely_affected": ["quant_rl/eval/plots.py"],
  "needs_planning": true,
  "needs_review": true,
  "special_considerations": "none"
}
```
"""
    )
    assert parse_complexity(output) == "complex"


def test_parse_complexity_last_valid_wins() -> None:
    text = '{"complexity": "trivial"} extra {"complexity": "medium"}'
    assert parse_complexity(text) == "medium"


def test_parse_verdict_from_heading() -> None:
    text = "## Summary\nok\n\n## Verdict\nfail\n\n## Fix Priority\n1. x\n"
    assert parse_verdict(text) == "fail"


def test_parse_verdict_conditional() -> None:
    text = "## Verdict\nconditional_pass\n"
    assert parse_verdict(text) == "conditional_pass"


def test_parse_tier_from_json() -> None:
    text = '{"tier": "T3", "complexity": "complex"}'
    from orchestra.parse import parse_tier

    assert parse_tier(text) == "T3"


def test_classify_failure_rate_limit() -> None:
    from orchestra.failures import FailureKind, classify_failure

    result = classify_failure("HTTP 429 rate limit exceeded", cli="gemini")
    assert result.kind == FailureKind.TRANSIENT_RATE_LIMIT


def test_static_tier_fast_path() -> None:
    from orchestra.parse import static_tier_fast_path

    assert static_tier_fast_path("hello world") == "T0"
    assert static_tier_fast_path("refactor the entire architecture") is None


def test_triage_prompt_file_exists() -> None:
    path = Path(__file__).resolve().parents[2] / "orchestra" / "prompts" / "triage.md"
    assert path.is_file()
