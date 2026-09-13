"""Failure taxonomy for orchestra agent runs."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

_PACIFIC = ZoneInfo("America/Los_Angeles")


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
    if exit_code == 124:
        return ClassifiedFailure(FailureKind.TIMEOUT, text[:500])
    combined = text.lower()
    sigs = signatures or {}
    cli_sigs = sigs.get(cli, {}) if cli else {}
    default_rate = ["429", "rate limit", "quota exceeded", "resource_exhausted"]
    default_auth = ["401", "403", "permission_denied", "unauthorized", "invalid api key"]

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

    if exit_code and exit_code >= 500:
        return ClassifiedFailure(FailureKind.PROVIDER_ERROR, text[:500])

    if "timeout" in combined or "timed out" in combined:
        return ClassifiedFailure(FailureKind.TIMEOUT, text[:500])

    return ClassifiedFailure(FailureKind.UNKNOWN, text[:500])
