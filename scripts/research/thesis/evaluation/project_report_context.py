"""Deterministic project and run context for human-readable thesis reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

PIPELINE_STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Configuration",
        "Defines data, date splits, features, trading rules, and learner settings.",
        ("config/base/default.yaml", "configuration/loader.py"),
    ),
    (
        "Market data",
        "Loads bars and tick quotes; prepares timeframes and saved data caches.",
        ("market_data/loading/pipeline.py", "market_data/loading/ticks.py"),
    ),
    (
        "Features and split",
        "Builds model inputs and assigns observations to configured date periods.",
        ("features/pipeline/builder.py", "market_data/splits/definitions.py"),
    ),
    (
        "Trading simulation",
        "Applies policy actions, simulated orders, costs, fills, and account updates.",
        ("simulation/environment/env.py", "trading/execution/costs.py"),
    ),
    (
        "Model and learner",
        "Encodes feature history and fits a PPO or SAC policy.",
        ("models/encoders/", "learning/rl/agent.py"),
    ),
    (
        "Training workflow",
        "Prepares runs, learns, saves checkpoints, and records run artifacts.",
        ("training/runner/training.py", "training/runner/seed_runtime.py"),
    ),
    (
        "Evaluation and plots",
        "Replays policies, calculates metrics, creates diagnostic plots and reports.",
        (
            "evaluation/rollout/runner.py",
            "evaluation/metrics/performance.py",
            "evaluation/plotting/",
            "scripts/research/thesis/evaluation/report_variant_details.py",
        ),
    ),
    (
        "Campaign identity",
        "Records variant state, source/data identity, checkpoints, and output paths.",
        ("experiments/campaigns/", "evaluation/reporting/run_manifest.py"),
    ),
)

PIPELINE_DIAGRAM = """flowchart LR
    C[Configuration] --> D[Market bars and tick quotes]
    D --> K[Versioned data and feature caches]
    K --> F[Features and date split]
    F --> E[Trading simulation and execution costs]
    E --> M[Model encoder and RL policy]
    M --> T[Training and checkpoint]
    T --> V[Evaluation on the recorded split]
    V --> R[Metrics, trade records, and plots]
    X[Campaign and run manifests] -. record identity .-> D
    X -. record identity .-> T
    X -. record identity .-> V
"""

GLOSSARY = {
    "Bar": "A time-bucketed price record, such as one M1 candle.",
    "Tick quote": "A time-stamped bid/ask quote used as a detailed input and for fill pricing.",
    "Feature": "A number calculated from market history and supplied to the policy.",
    "Cache": "A saved intermediate dataset or feature matrix used to avoid rebuilding it.",
    "Checkpoint": "The saved neural-network policy that can be loaded for later evaluation.",
    "Net PnL": "Profit or loss after the costs represented in the saved simulation outputs.",
    "Maximum drawdown": "The largest measured fall from an earlier account-equity peak.",
    "Seed": "The recorded random seed for one run; one seed does not measure across-seed uncertainty.",
}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _inline(value: Any) -> str:
    return f"{chr(96)}{value}{chr(96)}"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain") or ""
    rows = status.splitlines()
    return {
        "path": str(root),
        "head": git("rev-parse", "HEAD"),
        "tracked_changes": sum(not row.startswith("??") for row in rows),
        "untracked_changes": sum(row.startswith("??") for row in rows),
    }


def _raw_inputs(campaign: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    result = []
    for identity, digest in sorted(_dict(campaign.get("data_inputs")).items()):
        role, separator, raw_path = str(identity).partition(":")
        path = Path(raw_path if separator else role)
        match = re.search(r"_(M\d+|H\d+|D1)_", path.name)
        result.append(
            {
                "role": role if separator else "unlabeled",
                "path": path.as_posix(),
                "timeframe_in_filename": match.group(1) if match else "unsuffixed",
                "sha256": digest if isinstance(digest, str) else None,
                "present_at_report_time": (root / path).is_file(),
            }
        )
    return result


def _datasets(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = []
    for name, item in sorted(_dict(manifest.get("datasets")).items()):
        value = _dict(item)
        columns = value.get("columns")
        datasets.append(
            {
                "name": name,
                "start": value.get("start"),
                "end": value.get("end"),
                "rows": value.get("rows"),
                "column_count": len(columns) if isinstance(columns, list) else None,
                "sha256": value.get("sha256"),
            }
        )
    return datasets


def _variant_evidence(
    campaign_dir: Path, variant: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    run_dir = campaign_dir / str(variant.get("run_artifact_dir") or "")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        return {
            "variant": variant.get("variant"),
            "seed": variant.get("seed"),
            "manifest_available": False,
        }

    repository = _dict(manifest.get("repository"))
    source = _dict(manifest.get("source"))
    source_head = source.get("head") or repository.get("head")
    dirty_files = source.get("dirty_files")
    architecture = _dict(manifest.get("training_architecture"))
    protocol = _dict(manifest.get("training_protocol"))
    execution = _dict(manifest.get("execution_protocol"))
    metrics_protocol = _dict(manifest.get("metrics_protocol"))
    learning = _dict(manifest.get("learning_result"))
    config_identity = _dict(manifest.get("config"))
    checkpoint = _dict(manifest.get("checkpoint"))
    ppo = _dict(protocol.get("ppo"))
    config = _read_yaml(run_dir / "config.yaml")
    schedule = _dict(execution.get("policy_market_context"))
    action_mode = next(
        (
            metrics_protocol[key]
            for key in ("action_mode", "policy_action_mode", "inference_action_mode")
            if key in metrics_protocol
        ),
        None,
    )
    required = ("split", "datasets", "feature_cache", "training_architecture")
    return {
        "variant": variant.get("variant"),
        "seed": manifest.get("seed", variant.get("seed")),
        "manifest_available": True,
        "manifest_path": manifest_path.relative_to(campaign_dir).as_posix(),
        "missing_evidence": [key for key in required if manifest.get(key) is None],
        "repository_at_run": {
            "path": repository.get("path"),
            "head": source_head,
            "dirty_files": dirty_files if isinstance(dirty_files, list) else [],
            "matches_report_checkout_head": source_head == current.get("head")
            if source_head and current.get("head")
            else None,
        },
        "split": _dict(manifest.get("split")),
        "datasets": _datasets(manifest),
        "feature_fit_scope": manifest.get("feature_fit_scope"),
        "feature_cache": _dict(manifest.get("feature_cache")),
        "config_identity": config_identity,
        "checkpoint": checkpoint,
        "model": {
            "algorithm": "PPO" if ppo else config.get("algo", "Not recorded"),
            "backend": learning.get("backend"),
            "encoder": architecture.get("encoder_class")
            or _dict(config.get("encoder")).get("arch"),
            "channels": architecture.get("encoder_channels"),
            "dilations": architecture.get("encoder_dilations"),
            "kernel_size": architecture.get("encoder_kernel_size"),
            "latent_dimension": architecture.get("encoder_latent_dim"),
            "policy_head": architecture.get("policy_head_architecture"),
            "parameter_count": architecture.get("policy_parameter_count"),
            "action_schema": _dict(manifest.get("action_schema")),
            "observation_schema": _dict(manifest.get("observation_schema")),
        },
        "training": {
            "requested_steps": protocol.get("requested_steps"),
            "completed_steps": manifest.get("completed_steps"),
            "action_distribution": protocol.get("action_distribution"),
            "parallel_environments": architecture.get("logical_n_envs"),
            "rollout_steps_per_environment": architecture.get("sb3_n_steps_per_env"),
            "batch_size": ppo.get("batch_size"),
            "epochs": ppo.get("epochs"),
            "learning_rate": ppo.get("learning_rate"),
            "reward": _dict(protocol.get("reward")),
            "risk": _dict(protocol.get("risk")),
        },
        "evaluation": {
            "test_role": metrics_protocol.get("historical_test_role"),
            "action_mode": action_mode,
            "action_seed": metrics_protocol.get("execution_seed"),
            "tick_execution": execution.get("tick_data_configured"),
            "random_fill_latency_ms": _dict(execution.get("execution")).get(
                "randomized_fill_latency_ms"
            ),
            "policy_context": schedule,
            "costs": _dict(execution.get("costs")),
        },
    }


def build_project_context(
    campaign_dir: Path,
    campaign: dict[str, Any],
    variants: list[dict[str, Any]],
    report_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Collect source map, input/cache identity, and actual per-run configuration."""
    source_map = []
    for name, purpose, paths in PIPELINE_STAGES:
        sources = []
        for source in paths:
            path = repo_root / source
            sources.append(
                {
                    "path": source,
                    "exists": path.exists(),
                    "report_link": os.path.relpath(path, report_dir).replace(os.sep, "/"),
                }
            )
        source_map.append({"name": name, "purpose": purpose, "sources": sources})

    current = _git_identity(repo_root)
    evidence = [_variant_evidence(campaign_dir, item, current) for item in variants]
    core_names = {"train_bars", "train_features", "test_bars", "test_features", "primary_ticks"}
    comparable: list[dict[str, Any]] = []
    complete_comparison = bool(evidence) and all(run.get("manifest_available") for run in evidence)
    for run in evidence:
        hashes = {
            row["name"]: row.get("sha256")
            for row in run.get("datasets", [])
            if row["name"] in core_names
        }
        if set(hashes) != core_names or any(not hashes.get(name) for name in core_names):
            complete_comparison = False
        comparable.append(hashes)
    aligned = (
        all(row == comparable[0] for row in comparable[1:])
        if complete_comparison and len(comparable) > 1
        else None
    )
    generator_paths = (
        "scripts/research/thesis/evaluation/report_variant_details.py",
        "scripts/research/thesis/evaluation/project_report_context.py",
        "scripts/research/thesis/evaluation/project_report_markdown.py",
    )
    snapshot = _dict(campaign.get("source_snapshot"))
    snapshot_path = campaign_dir / str(snapshot.get("path") or "")
    snapshot = {
        **snapshot,
        "present_at_report_time": bool(snapshot.get("path") and snapshot_path.is_file()),
        "source_file_hash_count": len(_dict(campaign.get("source_hashes"))),
    }
    return {
        "schema_version": 1,
        "current_repository": current,
        "report_generator_sources": [
            {"path": path, "sha256": _sha256(repo_root / path)} for path in generator_paths
        ],
        "campaign_source_revision": campaign.get("git_sha"),
        "campaign_source_snapshot": snapshot,
        "pipeline_diagram": PIPELINE_DIAGRAM,
        "source_map": source_map,
        "raw_inputs": _raw_inputs(campaign, repo_root),
        "cache_summary": _read_json(campaign_dir / "reports" / "split_cache_ready.json"),
        "run_evidence": evidence,
        "cross_variant_data_hashes_align": aligned,
        "detail_report_is_artifact_based": True,
    }
