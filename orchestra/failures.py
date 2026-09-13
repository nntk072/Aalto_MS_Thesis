"""Failure taxonomy for orchestra agent runs."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

_PACIFIC = ZoneInfo("America/Los_Angeles")
_TRANSPORT_BLOCK_S = 3600.0
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


class FailureKind(StrEnum):
    TRANSIENT_RATE_LIMIT = "transient_rate_limit"
    DAILY_EXHAUSTED = "daily_exhausted"
    AUTH_INVALID = "auth_invalid"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    QUALITY_FAILURE = "quality_failure"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedFailure:
    kind: FailureKind
    message: str
    retry_after: float | None = None
    exhausted_until: float | None = None


def next_midnight_pacific() -> float:
    """Unix timestamp for next midnight America/Los_Angeles."""
    from datetime import datetime, timedelta

    now = datetime.now(_PACIFIC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.timestamp()


def classify_failure(
    text: str,
    exit_code: int | None = None,
    cli: str = "",
    signatures: dict[str, Any] | None = None,
) -> ClassifiedFailure:
    """Classify agent failure from log text and exit code."""
    combined = _ANSI_RE.sub("", text).lower()
    sigs = signatures or {}
    cli_sigs = sigs.get(cli, {}) if cli else {}
    default_rate = ["429", "rate limit", "quota exceeded", "resource_exhausted"]
    default_auth = ["401", "403", "permission_denied", "unauthorized", "invalid api key"]

    if _quota_exhausted(combined, cli):
        return ClassifiedFailure(
            FailureKind.DAILY_EXHAUSTED,
            text[:500],
            exhausted_until=_quota_resume_at(combined),
        )

    if transport_blocked(combined):
        return ClassifiedFailure(
            FailureKind.PROVIDER_ERROR,
            text[:500],
            retry_after=time.time() + _TRANSPORT_BLOCK_S,
        )

    auth_patterns = cli_sigs.get("auth", []) or default_auth
    for pattern in auth_patterns:
        if pattern.lower() in combined or (str(exit_code) == pattern):
            return ClassifiedFailure(FailureKind.AUTH_INVALID, text[:500])

    if "perday" in combined or "generate_content_free_tier_requests" in combined:
        return ClassifiedFailure(
            FailureKind.DAILY_EXHAUSTED,
            text[:500],
            exhausted_until=next_midnight_pacific(),
        )

    rate_patterns = cli_sigs.get("rate_limit", []) or default_rate
    for pattern in rate_patterns:
        if pattern.lower() in combined or (str(exit_code) == pattern):
            retry = 60.0
            match = re.search(r"retry[- ]?after[:\s]+(\d+)", combined, re.I)
            if match:
                retry = float(match.group(1))
            if "perminute" in combined or "rpm" in combined:
                kind = FailureKind.TRANSIENT_RATE_LIMIT
            elif "perday" in combined or "rpd" in combined:
                return ClassifiedFailure(
                    FailureKind.DAILY_EXHAUSTED,
                    text[:500],
                    exhausted_until=next_midnight_pacific(),
                )
            else:
                kind = FailureKind.TRANSIENT_RATE_LIMIT
            return ClassifiedFailure(
                kind,
                text[:500],
                retry_after=time.time() + retry,
            )

    if exit_code == 124 or "timeout" in combined or "timed out" in combined:
        return ClassifiedFailure(FailureKind.TIMEOUT, text[:500])

    if exit_code and exit_code >= 500:
        return ClassifiedFailure(FailureKind.PROVIDER_ERROR, text[:500])

    return ClassifiedFailure(FailureKind.UNKNOWN, text[:500])


def transport_blocked(combined: str) -> bool:
    """True when the CLI cannot reach the provider (proxy/WSS/CONNECT)."""
    return _hard_transport(combined) or _soft_transport(combined)


def abort_dead_session(combined: str, cli: str = "") -> bool:
    """True when the live log shows the agent cannot recover."""
    return _quota_exhausted(combined, cli) or _hard_transport(combined)


def inspect_live_log(snippet: str) -> str | None:
    """Classify only top-level CLI lines, not errors quoted in prompts/tools.

    Returns ``complete``, ``abort``, or None.
    """
    for raw in snippet.splitlines():
        line = _ANSI_RE.sub("", raw).strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(obj.get("type") or "")
            if kind == "error":
                return "abort"
            event = obj.get("event")
            if kind == "agent_event" and isinstance(event, dict):
                if event.get("type") == "done" and str(event.get("reason") or "") == "completed":
                    return "complete"
            continue
        if abort_dead_session(line.lower()):
            return "abort"
    return None


def agent_completed(combined: str) -> bool:
    """True when a Cline JSONL done event is present (tests / excerpts)."""
    return inspect_live_log(combined) == "complete"


def failure_excerpt(output: str, n: int = 8000) -> str:
    """Tail plus transport/quota lines so classify_failure sees them."""
    markers = (
        "usage limit",
        "307 temporary redirect",
        "failed to connect to websocket",
        "gateway.zscaler",
        "falling back from websockets",
        "econnreset",
        "proxy connection failed",
        "cannot connect to api",
    )
    hits = [ln for ln in output.splitlines() if any(m in ln.lower() for m in markers)]
    head = "\n".join(hits[-50:])
    tail = output[-n:] if output else ""
    if head and head not in tail:
        return f"{head}\n{tail}"
    return tail


def _hard_transport(combined: str) -> bool:
    markers = (
        "proxy connection failed",
        "connect response missing status line",
        "waiting for network",
        "http connect response missing",
        "econnreset",
        "socket connection was closed unexpectedly",
        "cannot connect to api",
        "307 temporary redirect",
        "gateway.zscaler",
        "failed to connect to websocket",
    )
    return any(m in combined for m in markers)


def _soft_transport(combined: str) -> bool:
    markers = ("falling back from websockets",)
    return any(m in combined for m in markers)


def _quota_exhausted(combined: str, cli: str) -> bool:
    """True when the provider rejected the call because usage is spent."""
    if "usage limit" in combined or "tokens exhausted" in combined:
        return True
    return cli == "codex" and "403" in combined and "websocket" in combined


def _quota_resume_at(combined: str) -> float:
    """When Codex says try again on a calendar date, honor it; else midnight PT."""
    match = re.search(
        r"try again at ([a-z]{3})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
        combined,
    )
    if match:
        try:
            resume = datetime.strptime(
                f"{match.group(1)} {match.group(2)} {match.group(3)}",
                "%b %d %Y",
            )
            return resume.replace(tzinfo=_PACIFIC).timestamp()
        except ValueError:
            pass
    return next_midnight_pacific()
