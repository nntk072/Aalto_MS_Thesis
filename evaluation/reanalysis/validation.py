"""Input, schema, and provenance checks for deterministic reanalysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.reporting.run_manifest import file_sha256


def resolve_execution_seed(
    requested_seed: int | None, source_manifest: dict[str, Any]
) -> int | None:
    """Use an explicit replay seed or the recorded training seed for fill jitter."""
    if requested_seed is not None:
        if requested_seed < 0:
            raise ValueError("evaluation seed must be non-negative")
        return requested_seed
    training_protocol = source_manifest.get("training_protocol")
    metrics_protocol = source_manifest.get("metrics_protocol")
    candidates = (
        source_manifest.get("seed"),
        training_protocol.get("seed") if isinstance(training_protocol, dict) else None,
        metrics_protocol.get("execution_seed") if isinstance(metrics_protocol, dict) else None,
    )
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return candidate
    return None


def validate_sequence_input_width(model: Any, expected_width: int) -> int:
    """Require the checkpoint's sequence tensor width to match its feature schema."""
    observation_spaces = getattr(getattr(model, "observation_space", None), "spaces", None)
    get_space = getattr(observation_spaces, "get", None)
    sequence_space = get_space("seq") if callable(get_space) else None
    shape = getattr(sequence_space, "shape", None)
    if not shape or int(shape[-1]) != int(expected_width):
        raise ValueError(
            "checkpoint sequence input width disagrees with evaluated feature coordinates: "
            f"checkpoint={shape}, expected={expected_width}"
        )
    return int(shape[-1])


def validate_stop_candidate_schema(
    *,
    checkpoint_schema: dict[str, Any] | None,
    manifest_schema: dict[str, Any] | None,
    expected_schema: dict[str, Any],
    allow_legacy: bool,
) -> str:
    """Reject checkpoint/config drift and label explicit latest-swing legacy use."""
    if checkpoint_schema is None or manifest_schema is None:
        if not allow_legacy:
            raise ValueError(
                "checkpoint or run manifest has no stop-candidate schema; pass "
                "--allow-legacy-stop-schema only when intentionally reanalysing a legacy run"
            )
        available_schema = checkpoint_schema if checkpoint_schema is not None else manifest_schema
        if available_schema is None:
            if expected_schema.get("menu") != "legacy_last_accepted_swing_v1":
                raise ValueError(
                    "legacy checkpoint has no stop schema and the run config requests the "
                    "expanded multi-timeframe menu; legacy opt-in cannot assert that menu "
                    "was trained"
                )
        elif available_schema != expected_schema:
            raise ValueError(
                "available stop-candidate schema differs from the run config; legacy opt-in "
                "does not authorize evaluating under a changed stop menu"
            )
        return "legacy_opt_in"
    if checkpoint_schema != manifest_schema:
        raise ValueError("checkpoint and run manifest stop-candidate schemas disagree")
    if checkpoint_schema != expected_schema:
        raise ValueError(
            "checkpoint stop-candidate schema differs from the run config; refusing to evaluate "
            "under a changed stop menu"
        )
    return "versioned"


def validate_checkpoint_manifest_hash(path: Path, manifest: dict[str, Any]) -> str:
    """Verify the checkpoint is the exact artifact recorded by its run manifest."""
    checkpoint_record = manifest.get("checkpoint")
    expected_hash = checkpoint_record.get("sha256") if isinstance(checkpoint_record, dict) else None
    if expected_hash is None:
        return "unavailable_legacy_manifest"
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            "checkpoint SHA-256 does not match the source run manifest; refusing reanalysis"
        )
    return "verified"


__all__ = [
    "resolve_execution_seed",
    "validate_checkpoint_manifest_hash",
    "validate_sequence_input_width",
    "validate_stop_candidate_schema",
]
