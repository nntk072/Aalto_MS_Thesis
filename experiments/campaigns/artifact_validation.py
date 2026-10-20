"""Focused campaign execution artifact validation services."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import zipfile
from typing import Any

from experiments.campaigns.source_snapshot import (
    action_space_matches_width,
    cached_sha256_file,
    sha256_file,
)


def validate_campaign_evaluation(value: object) -> dict[str, Any]:
    """Validate report-only use or explicit selection on exposed historical data."""
    if not isinstance(value, dict) or value.get("deterministic") is not True:
        raise ValueError("campaign evaluation must be deterministic")
    allowed = {"locked_oos_role", "deterministic", "locked_oos_exposure", "selection_allowed"}
    if set(value) - allowed:
        raise ValueError("campaign evaluation contains unknown fields")
    exposure = value.get("locked_oos_exposure")
    if exposure is not None and exposure != "previously_inspected_holdout":
        raise ValueError("unrecognized locked-OOS exposure status")
    selection_allowed = value.get("selection_allowed", False)
    if not isinstance(selection_allowed, bool):
        raise ValueError("selection_allowed must be a boolean")
    if selection_allowed and exposure != "previously_inspected_holdout":
        raise ValueError("selection is allowed only on explicitly exposed historical data")
    expected_role = "selection_acknowledged" if selection_allowed else "final_report_only"
    if value.get("locked_oos_role") != expected_role:
        raise ValueError(f"evaluation role must be {expected_role!r} for this selection setting")
    return dict(value)


def verify_run_artifacts(
    run_out: pathlib.Path,
    key: str,
    expected_steps: int,
    *,
    expected_fingerprint: str,
    expected_attempt: int,
) -> tuple[bool, str]:
    """Require the requested budget, a readable checkpoint, and a valid training record."""
    provenance_path = run_out / "attempt_provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "missing or invalid attempt_provenance.json"
    variant, _, seed_text = key.rpartition("__seed")
    if (
        not isinstance(provenance, dict)
        or not seed_text.isdigit()
        or provenance.get("run_key") != key
        or provenance.get("variant") != variant
        or provenance.get("seed") != int(seed_text)
        or provenance.get("requested_timesteps") != expected_steps
        or provenance.get("attempt") != expected_attempt
        or provenance.get("campaign_fingerprint") != expected_fingerprint
    ):
        return False, "attempt provenance does not match the expected campaign run"
    identity = provenance.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("steps") != expected_steps
        or not isinstance(identity.get("jobs"), list)
        or key not in identity["jobs"]
    ):
        return False, "attempt provenance identity does not include the requested run"
    identity_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if identity_fingerprint != expected_fingerprint:
        return False, "attempt provenance identity does not match the campaign fingerprint"
    for path in run_out.glob("**/training_log.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        if (
            not isinstance(payload.get("env_flags"), dict)
            or not isinstance(payload.get("observation_space_shapes"), dict)
            or not isinstance(payload.get("action_space_width"), int)
        ):
            continue
        if not seed_text.isdigit() or payload.get("seed") != int(seed_text):
            continue
        try:
            config_manifest = json.loads(
                (path.parent.parent / "resolved_config.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        if not isinstance(config_manifest, dict):
            continue
        if (
            config_manifest.get("variant") != variant
            or config_manifest.get("seed") != int(seed_text)
            or config_manifest.get("steps") != expected_steps
        ):
            continue
        required_metrics = (
            "test_sharpe",
            "test_max_dd",
            "test_trades",
            "test_return",
            "test_survived_full_year",
            "train_equity_ok",
            "feature_cache_hash",
            "feature_columns",
            "feature_schema_sha256",
            "effective_config_sha256",
            "action_space_width",
            "observation_space_shapes",
            "env_flags",
        )
        if any(key not in payload for key in required_metrics):
            continue
        numeric_metrics = ("test_sharpe", "test_max_dd", "test_return")
        if any(
            not isinstance(payload[key], (int, float)) or not math.isfinite(payload[key])
            for key in numeric_metrics
        ):
            continue
        if (
            not isinstance(payload["test_trades"], int)
            or not isinstance(payload["test_survived_full_year"], bool)
            or not isinstance(payload["train_equity_ok"], bool)
        ):
            continue
        if not payload["train_equity_ok"]:
            reason = "training risk-survival gate failed"
            gate_path = path.parent / "train_equity_gate.json"
            try:
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                gate = None
            if isinstance(gate, dict) and isinstance(gate.get("reason"), str):
                reason += f": {gate['reason']}"
            return False, reason
        # The cache sidecar is intentionally a 16-hex content key produced by
        # feature_cache_content_hash(); the full parquet SHA-256 is recorded
        # separately in run_manifest.json. Requiring 64 characters here made
        # every otherwise-complete run fail artifact admission.
        cache_hash = payload.get("feature_cache_hash")
        cache_name = payload.get("feature_cache_name")
        manifest_names = payload.get("run_manifests")
        if (
            not isinstance(cache_hash, str)
            or not re.fullmatch(r"[0-9a-f]{16}", cache_hash)
            or not isinstance(cache_name, str)
            or not cache_name
            or not isinstance(manifest_names, list)
            or not manifest_names
            or any(not isinstance(name, str) or not name for name in manifest_names)
        ):
            continue
        cache_manifest_checkpoint_hash: str | None = None
        cache_manifest_valid = True
        for name in manifest_names:
            manifest_path = (path.parent / name).resolve()
            try:
                manifest_path.relative_to(path.parent.resolve())
                run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache_manifest_valid = False
                break
            if not isinstance(run_manifest, dict):
                cache_manifest_valid = False
                break
            feature_cache = run_manifest.get("feature_cache")
            checkpoint_manifest = run_manifest.get("checkpoint")
            config_manifest = run_manifest.get("config")
            if not isinstance(feature_cache, dict):
                cache_manifest_valid = False
                break
            if not isinstance(checkpoint_manifest, dict):
                cache_manifest_valid = False
                break
            if not isinstance(config_manifest, dict):
                cache_manifest_valid = False
                break
            cache_path_text = feature_cache.get("path")
            cache_file_hash = feature_cache.get("file_sha256")
            if (
                feature_cache.get("identity") != cache_name
                or feature_cache.get("sha256") != cache_hash
                or not isinstance(cache_path_text, str)
                or not isinstance(cache_file_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", cache_file_hash)
                or config_manifest.get("sha256") != payload.get("effective_config_sha256")
                or not isinstance(checkpoint_manifest.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_manifest["sha256"])
            ):
                cache_manifest_valid = False
                break
            cache_path = pathlib.Path(cache_path_text)
            sidecar_path = pathlib.Path(str(cache_path) + ".hash")
            try:
                if (
                    not cache_path.is_file()
                    or sidecar_path.read_text(encoding="utf-8").strip() != cache_hash
                    or cached_sha256_file(cache_path) != cache_file_hash
                ):
                    cache_manifest_valid = False
                    break
            except OSError:
                cache_manifest_valid = False
                break
            if cache_manifest_checkpoint_hash not in (None, checkpoint_manifest["sha256"]):
                cache_manifest_valid = False
                break
            cache_manifest_checkpoint_hash = checkpoint_manifest["sha256"]
        if not cache_manifest_valid:
            continue
        feature_columns = payload.get("feature_columns")
        schema_hash = payload.get("feature_schema_sha256")
        if not isinstance(feature_columns, list) or any(
            not isinstance(column, str) for column in feature_columns
        ):
            continue
        actual_schema_hash = hashlib.sha256(
            json.dumps(feature_columns, separators=(",", ":")).encode()
        ).hexdigest()
        if schema_hash != actual_schema_hash:
            continue
        import yaml

        try:
            effective_config = yaml.safe_load(
                (path.parent / "config.yaml").read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(effective_config, dict) or not isinstance(
            effective_config.get("env"), dict
        ):
            continue
        config_hash = payload.get("effective_config_sha256")
        if (
            not isinstance(config_hash, str)
            or sha256_file(path.parent / "config.yaml") != config_hash
        ):
            continue
        if any(
            effective_config["env"].get(name) != value
            for name, value in payload["env_flags"].items()
        ):
            continue
        completed = payload.get("timesteps_completed")
        if not isinstance(completed, int) or completed < expected_steps:
            continue
        if payload.get("timesteps") != expected_steps:
            continue
        model_file = path.parent / "model" / "ppo_final.zip"
        if not model_file.is_file() or model_file.stat().st_size == 0:
            continue
        if cache_manifest_checkpoint_hash != sha256_file(model_file):
            continue
        try:
            with zipfile.ZipFile(model_file) as archive:
                if archive.testzip() is not None:
                    continue
            algo = str(payload.get("algo", "")).lower()
            model_class: Any
            if algo == "ppo":
                from stable_baselines3 import PPO

                model_class = PPO
            elif algo == "sac":
                from stable_baselines3 import SAC

                model_class = SAC
            else:
                continue
            model = model_class.load(model_file, device="cpu")
            if int(model.num_timesteps) < expected_steps:
                continue
            if not action_space_matches_width(model.action_space, payload["action_space_width"]):
                continue
            model_obs = getattr(model.observation_space, "spaces", {})
            model_obs_shapes = {
                str(name): [int(size) for size in space.shape]
                for name, space in model_obs.items()
                if hasattr(space, "shape")
            }
            if model_obs_shapes != payload["observation_space_shapes"]:
                continue
        except Exception:
            continue
        return True, ""
    return False, f"missing valid training record/checkpoint at requested {expected_steps} steps"


def hash_run_artifacts(run_out: pathlib.Path) -> dict[str, str]:
    """Record checksums for completed logs, configs, and model checkpoints."""
    suffixes = {
        "training_log.json",
        "ppo_final.zip",
        "config.yaml",
        "config.json",
        "attempt_provenance.json",
    }
    return {
        str(path.relative_to(run_out)): sha256_file(path)
        for path in run_out.rglob("*")
        if path.is_file() and (path.name in suffixes or path.name == "resolved_config.json")
    }
