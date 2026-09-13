"""Tests for orchestra output parsers."""

from __future__ import annotations

import time
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


def test_classify_codex_usage_limit_and_websocket_307() -> None:
    from orchestra.failures import FailureKind, classify_failure

    text = (
        "failed to connect to websocket: HTTP error: 307 Temporary Redirect, "
        "url: wss://chatgpt.com/backend-api/codex/responses\n"
        "ERROR: You've hit your usage limit. To continue using Codex ... "
        "try again at Oct 13th, 2026 3:06 PM.\n"
    )
    result = classify_failure(text, cli="codex")
    assert result.kind == FailureKind.DAILY_EXHAUSTED
    assert result.exhausted_until is not None
    assert result.exhausted_until > time.time() + 86400


def test_classify_codex_websocket_307_without_quota_is_transport() -> None:
    from orchestra.failures import FailureKind, classify_failure

    text = (
        "failed to connect to websocket: HTTP error: 307 Temporary Redirect, "
        "url: wss://chatgpt.com/backend-api/codex/responses\n"
    )
    result = classify_failure(text, cli="codex")
    assert result.kind == FailureKind.PROVIDER_ERROR
    assert result.retry_after is not None


def test_abort_dead_session_stops_on_websocket_307() -> None:
    from orchestra.failures import abort_dead_session

    text = (
        "failed to connect to websocket: HTTP error: 307 Temporary Redirect, "
        "url: wss://chatgpt.com/backend-api/codex/responses\n"
    )
    assert abort_dead_session(text.lower(), cli="codex") is True
    assert abort_dead_session("gateway.zscaler.net/_sm_ctn", cli="codex") is True


def test_classify_timeout_with_websocket_307_is_transport() -> None:
    from orchestra.failures import FailureKind, classify_failure

    text = (
        "Session x timed out after 300s (exit_code=missing)\n"
        "failed to connect to websocket: HTTP error: 307 Temporary Redirect\n"
    )
    result = classify_failure(text, exit_code=124, cli="codex")
    assert result.kind == FailureKind.PROVIDER_ERROR
    assert result.retry_after is not None


def test_classify_codex_websocket_403_as_quota_exhausted() -> None:
    from orchestra.failures import FailureKind, classify_failure

    text = (
        "Session orchestra-review_synthesizer-codex exited 1\n"
        "failed to connect to websocket: HTTP error: 403 Forbidden, "
        "url: wss://chatgpt.com/backend-api/codex/responses\n"
    )
    result = classify_failure(text, cli="codex")
    assert result.kind == FailureKind.DAILY_EXHAUSTED
    assert result.exhausted_until is not None


def test_classify_cline_econnreset_as_provider_error() -> None:
    from orchestra.failures import FailureKind, classify_failure

    text = (
        '{"type":"error","message":"Cannot connect to API: The socket connection '
        'was closed unexpectedly. ... fetch() (ECONNRESET)","model":'
        '{"id":"z-ai/glm-5.3-flash","provider":"cline"}}'
    )
    result = classify_failure(text, cli="cline")
    assert result.kind == FailureKind.PROVIDER_ERROR
    assert result.retry_after is not None


def test_classify_codex_proxy_connect_failure_as_provider_error() -> None:
    from orchestra.failures import FailureKind, classify_failure

    text = (
        "failed to connect to websocket: URL error: Proxy connection failed: "
        "HTTP CONNECT response missing status line, "
        "url: wss://chatgpt.com/backend-api/codex/responses\n"
        "Falling back from WebSockets to HTTPS transport.\n"
    )
    result = classify_failure(text, cli="codex")
    assert result.kind == FailureKind.PROVIDER_ERROR
    assert result.retry_after is not None
    assert result.exhausted_until is None


def test_static_tier_fast_path() -> None:
    from orchestra.parse import static_tier_fast_path

    assert static_tier_fast_path("hello world") == "T0"
    assert static_tier_fast_path("refactor the entire architecture") is None


def test_triage_prompt_file_exists() -> None:
    path = Path(__file__).resolve().parents[2] / "orchestra" / "prompts" / "triage.md"
    assert path.is_file()
