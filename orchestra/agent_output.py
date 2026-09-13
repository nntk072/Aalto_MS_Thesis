"""Extract concise agent responses from noisy CLI / PTY logs."""

from __future__ import annotations

import json
import re
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
_DONE_RE = re.compile(
    r'"type"\s*:\s*"done"[^}]*"reason"\s*:\s*"completed"',
    re.IGNORECASE,
)
_JSONL_CLIS = frozenset({"cline", "opencode", "kilo"})
_DEFAULT_MAX_CHARS = 48_000


def _parse_json_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _event_text(event: dict[str, Any]) -> str | None:
    for key in ("text", "content", "message"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_from_jsonl(raw: str) -> str:
    """Keep assistant text from JSONL agent events; drop tool/file dumps."""
    parts: list[str] = []
    for line in raw.splitlines():
        obj = _parse_json_line(line)
        if obj is None:
            continue
        if obj.get("type") == "error":
            message = obj.get("message")
            if isinstance(message, str) and message.strip():
                parts.append(message.strip())
            continue
        event = obj.get("event")
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type") or "")
        content_type = str(event.get("contentType") or event.get("content_type") or "")
        if etype == "done":
            text = _event_text(event)
            if text:
                parts.append(text)
            continue
        if content_type == "tool":
            continue
        if etype in ("content_end", "content", "message", "text"):
            text = _event_text(event)
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def strip_orchestra_header(raw: str) -> str:
    lines = [
        ln
        for ln in _ANSI_RE.sub("", raw).splitlines()
        if not ln.startswith("[orchestra] exec") and not ln.startswith("hook:")
    ]
    return "\n".join(lines).strip()


def looks_like_jsonl_agent_log(text: str) -> bool:
    if '"type":"agent_event"' in text or '"type": "agent_event"' in text:
        return True
    return any(line.strip().startswith('{"ts":') for line in text.splitlines()[:20])


def extract_usable_output(
    raw: str,
    cli: str = "",
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Return assistant-facing text for routing, parsing, and state — not raw tool I/O."""
    body = strip_orchestra_header(raw)
    if not body:
        return ""

    use_jsonl = cli in _JSONL_CLIS or looks_like_jsonl_agent_log(body)
    if use_jsonl:
        extracted = extract_from_jsonl(body)
        if extracted:
            body = extracted

    if len(body) > max_chars:
        body = body[:max_chars] + "\n... [output truncated for pipeline]"
    return body
