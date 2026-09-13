"""Tests for agent log sanitization."""

from __future__ import annotations

from orchestra.agent_output import extract_from_jsonl, extract_usable_output


def test_extract_from_jsonl_drops_tool_file_dumps() -> None:
    raw = """
[orchestra] exec cline -p <prompt> --json
{"type":"agent_event","event":{"type":"content_end","contentType":"tool","toolName":"read_files","output":[{"result":"line1\\nline2"}]}}
{"type":"agent_event","event":{"type":"done","reason":"completed","text":"Fixed mypy in test_routing.py"}}
"""
    text = extract_usable_output(raw, cli="cline")
    assert "line1" not in text
    assert "read_files" not in text
    assert "Fixed mypy in test_routing.py" in text


def test_extract_from_jsonl_keeps_assistant_text_chunks() -> None:
    raw = """
{"type":"agent_event","event":{"type":"content_end","contentType":"text","text":"## Summary\\nAll good."}}
{"type":"agent_event","event":{"type":"done","reason":"completed","text":"Done."}}
"""
    assert "All good." in extract_from_jsonl(raw)
    assert "Done." in extract_from_jsonl(raw)


def test_extract_usable_output_truncates_huge_plain_text() -> None:
    raw = "x" * 60_000
    out = extract_usable_output(raw, max_chars=1000)
    assert len(out) < 1100
    assert "[output truncated for pipeline]" in out
