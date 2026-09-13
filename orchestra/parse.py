"""Parse orchestra agent outputs (complexity, verdict, JSON blobs)."""

from __future__ import annotations

import json
import re
from typing import Any

VALID_COMPLEXITY = frozenset({"trivial", "medium", "complex"})
VALID_TIERS = frozenset({"T0", "T1", "T2", "T3", "T4"})
VALID_VERDICTS = frozenset({"pass", "conditional_pass", "fail"})
_T0_PATTERN = re.compile(
    r"hello\s*world|rename|typo|comment\s+only|docstring",
    re.IGNORECASE,
)


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Return dicts decoded from JSON objects in ``text``.

    Uses ``JSONDecoder.raw_decode`` so a greedy ``{...}`` regex cannot swallow
    the whole pane (prompt template + model output).
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        i = end
    return objects


def parse_complexity(text: str) -> str:
    """Return triage complexity from model JSON, ignoring prompt templates.

    Prompt templates contain the word ``trivial`` and a JSON *example* whose
    complexity value is ``trivial|medium|complex``. Those must not win.
    The last valid complexity in the text is used (model output after any echo).
    """
    found: str | None = None
    for obj in extract_json_objects(text):
        value = obj.get("complexity")
        if isinstance(value, str) and value in VALID_COMPLEXITY:
            found = value
    return found or "medium"


def static_tier_fast_path(task: str) -> str | None:
    """Return T0 for trivial tasks without a model call."""
    text = task.strip()
    if len(text) >= 80:
        return None
    if _T0_PATTERN.search(text):
        return "T0"
    return None


def parse_tier(text: str) -> str:
    """Return triage tier from model JSON (T0..T4)."""
    from .models import complexity_to_tier

    found: str | None = None
    for obj in extract_json_objects(text):
        value = obj.get("tier")
        if isinstance(value, str):
            key = value.upper()
            if key in VALID_TIERS:
                found = key
    if found:
        return found
    return complexity_to_tier(parse_complexity(text))


def parse_verdict(text: str) -> str:
    """Extract pass/conditional_pass/fail from a review-synthesis report."""
    match = re.search(
        r"##\s*Verdict\s*\n\s*(pass|conditional_pass|fail)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()
    lower = text.lower()
    if "verdict" not in lower:
        return "pass"
    after = lower.split("verdict", 1)[1][:80]
    for token in ("conditional_pass", "fail", "pass"):
        if token in after:
            return token
    return "pass"
