from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.research.thesis.evaluation.project_report_context import (
    PIPELINE_DIAGRAM,
    build_project_context,
)
from scripts.research.thesis.evaluation.project_report_markdown import render_project_context


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pipeline_diagram_covers_the_saved_results_path() -> None:
    assert "Market bars and tick quotes" in PIPELINE_DIAGRAM
    assert "Versioned data and feature caches" in PIPELINE_DIAGRAM
    assert "Model encoder and RL policy" in PIPELINE_DIAGRAM
    assert "Evaluation on the recorded split" in PIPELINE_DIAGRAM
    assert "Metrics, trade records, and plots" in PIPELINE_DIAGRAM


def test_report_context_explains_data_model_source_and_evaluation_action_modes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    campaign_dir = tmp_path / "campaign"
    report_dir = campaign_dir / "reports"
    (campaign_dir / "source_snapshot.zip").parent.mkdir(parents=True)
    (campaign_dir / "source_snapshot.zip").write_text("snapshot", encoding="utf-8")
    run_dir = campaign_dir / "runs" / "a0"
    raw_file = repo / "data" / "training" / "US100.cash_M1.csv"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("fixture", encoding="utf-8")
    run_dir.mkdir(parents=True)
    evaluation_dir = run_dir / "evaluations"
    evaluation_dir.mkdir()
    (run_dir / "config.yaml").write_text("encoder:\n  arch: tcn\n", encoding="utf-8")
    _write_json(
        evaluation_dir / "train_in_sample.json",
        {"policy_action_mode": "stochastic", "action_rng_seed": 7},
    )
    _write_json(
        evaluation_dir / "test_report_only.json",
        {"policy_action_mode": "stochastic", "action_rng_seed": 7},
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "manifest_version": 1,
            "seed": 7,
            "repository": {"path": "/former/checkout"},
            "source": {
                "head": "old-revision",
                "dirty_files": [{"path": "run.py", "sha256": "dirty-hash"}],
            },
            "config": {"sha256": "config-hash"},
            "checkpoint": {"sha256": "checkpoint-hash"},
            "split": {
                "train_start": "2024-12-30",
                "train_end": "2025-12-31",
                "test_start": "2026-01-01",
            },
            "datasets": {
                "train_bars": {
                    "start": "2024-12-30",
                    "end": "2025-12-31",
                    "rows": 10,
                    "columns": ["open", "close"],
                    "sha256": "train-hash",
                },
                "train_features": {"sha256": "feature-hash"},
                "test_bars": {"sha256": "test-hash"},
                "test_features": {"sha256": "test-feature-hash"},
                "primary_ticks": {"sha256": "tick-hash"},
            },
            "feature_fit_scope": "train_mask_only",
            "feature_cache": {"identity": "features.parquet", "sha256": "cache-key"},
            "training_architecture": {
                "encoder_class": "TCNEncoder",
                "encoder_channels": [32, 64],
                "encoder_dilations": [1, 2],
                "encoder_kernel_size": 3,
                "encoder_latent_dim": 16,
                "policy_head_architecture": [64, 32],
                "policy_parameter_count": 12345,
                "logical_n_envs": 4,
                "sb3_n_steps_per_env": 128,
            },
            "training_protocol": {
                "requested_steps": 1000,
                "action_distribution": "clamped_diagonal_gaussian",
                "ppo": {"batch_size": 64, "epochs": 2, "learning_rate": 0.0003},
            },
            "execution_protocol": {
                "tick_data_configured": True,
                "execution": {"randomized_fill_latency_ms": 30},
                "policy_market_context": {
                    "features": ["ny_scheduled_short"],
                    "schedule_sha256": "calendar-hash",
                },
            },
            "metrics_protocol": {"historical_test_role": "previously_inspected_holdout"},
            "learning_result": {"backend": "stable_baselines3"},
            "action_schema": {"version": "overlay-action-v2", "width": 12},
            "observation_schema": {"version": "account-state-v1", "observation_width": 6},
            "completed_steps": 1024,
        },
    )
    _write_json(
        campaign_dir / "reports" / "split_cache_ready.json",
        {
            "cache_dir": "cache/train_test",
            "cache_files": 2,
            "tick_cache_files": 1,
            "bars": {"US100.cash": {"M1": 10}},
            "ticks": {"US100.cash": 20},
        },
    )
    campaign = {
        "git_sha": "campaign-revision",
        "data_inputs": {"training:data/training/US100.cash_M1.csv": "raw-hash"},
        "source_hashes": {"source.py": "source-hash"},
        "source_snapshot": {
            "path": "source_snapshot.zip",
            "sha256": "archive-hash",
            "manifest_sha256": "snapshot-manifest-hash",
            "file_count": 1,
        },
    }
    variants = [{"variant": "ladder_a0", "seed": 7, "run_artifact_dir": "runs/a0"}]

    context = build_project_context(campaign_dir, campaign, variants, report_dir, repo)
    markdown = render_project_context(context)

    run = context["run_evidence"][0]
    assert run["manifest_available"] is True
    assert run["model"]["algorithm"] == "PPO"
    assert run["model"]["encoder"] == "TCNEncoder"
    assert run["training"]["completed_steps"] == 1024
    assert run["config_identity"]["sha256"] == "config-hash"
    assert run["checkpoint"]["sha256"] == "checkpoint-hash"
    assert run["evaluation"]["training_action_mode"] == "stochastic"
    assert run["evaluation"]["action_mode"] == "stochastic"
    assert run["evaluation"]["action_mode_matches_training"] is True
    assert run["evaluation"]["action_seed"] == 7
    assert context["raw_inputs"][0]["present_at_report_time"] is True
    assert context["cache_summary"]["cache_files"] == 2
    assert context["campaign_source_snapshot"]["present_at_report_time"] is True
    assert "old-revision" in markdown
    assert "run.py" in markdown
    assert "config-hash" in markdown
    assert "checkpoint-hash" in markdown
    assert "snapshot-manifest-hash" in markdown
    assert "not recorded" in markdown.lower()
    assert "previously_inspected_holdout" in markdown
    assert "train_mask_only" in markdown
    assert "raw-hash" in markdown
    assert "Training dates" in markdown
    assert "| Modes match |" in markdown
    assert "stochastic" in markdown

    repeated_variants = [variants[0], {**variants[0], "variant": "ladder_a1"}]
    repeated_context = build_project_context(
        campaign_dir, campaign, repeated_variants, report_dir, repo
    )
    assert repeated_context["cross_variant_data_hashes_align"] is True

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["datasets"]["test_features"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    incomplete_context = build_project_context(
        campaign_dir, campaign, repeated_variants, report_dir, repo
    )
    assert incomplete_context["cross_variant_data_hashes_align"] is None
