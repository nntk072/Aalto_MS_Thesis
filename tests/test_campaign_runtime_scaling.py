"""Tests for bounded campaign timing extrapolations and their evidence source."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest
from scripts.research.campaigns.analyze_campaign_runtime_scaling import (
    build_runtime_screen,
    render_markdown,
)


def _write_worker(campaign_dir: Path, key: str, *, ppo: float, test: float, train: float) -> None:
    run_dir = campaign_dir / "runs" / "attempts" / key / "attempt1" / key / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "training_log.json").write_text(
        json.dumps(
            {
                "timesteps_completed": 8192,
                "phase_timings_seconds": {
                    "42": {
                        "ppo_learn": ppo,
                        "test_evaluation": test,
                        "train_evaluation": train,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_runtime_screen_scales_each_worker_and_retains_sample_mismatch(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "probe"
    (campaign_dir / "runs").mkdir(parents=True)
    (campaign_dir / "logs").mkdir()
    manifest = {
        "campaign_id": "probe",
        "timesteps": 8192,
        "data_source": "bundle",
        "n_envs": 64,
        "worker_cpus": 6,
        "worker_gpu_type": "gh200",
        "worker_gpus_per_node": 2,
        "worker_nodes": ["gpuarm2"],
        "workers": 2,
        "runtime": {"python": "3.12"},
        "jobs": [
            {"key": "variant_a__seed42", "seed": 42, "attempts": 1, "status": "failed"},
            {"key": "variant_b__seed42", "seed": 42, "attempts": 1, "status": "failed"},
        ],
    }
    (campaign_dir / "campaign.json").write_text(json.dumps(manifest), encoding="utf-8")
    resource_log = campaign_dir / "logs" / "resources_gpuarm2.csv"
    resource_log.write_text("timestamp,pss\n2026-10-14,1\n", encoding="utf-8")
    _write_worker(campaign_dir, "variant_a__seed42", ppo=409.6, test=10, train=20)
    _write_worker(campaign_dir, "variant_b__seed42", ppo=512, test=40, train=20)

    screen = build_runtime_screen(
        campaign_dir,
        target_steps=16_384,
        deadline_hours=1,
        target_data_source="frames",
        target_n_envs=4,
        target_worker_cpus=3,
    )

    derived = screen["derived_screen"]
    assert derived["worker_count"] == 2
    assert derived["effective_steps_per_second"] == pytest.approx(
        {"min": 16.0, "median": 18.0, "max": 20.0}
    )
    assert derived["linear_training_hours_at_sample_rate"]["median"] == pytest.approx(
        statistics.median([16_384 / 20 / 3600, 16_384 / 16 / 3600])
    )
    assert derived["one_final_train_test_evaluation_pair_hours"]["median"] == pytest.approx(
        45 / 3600
    )
    assert derived["sample_target_mismatch"] == {
        "data_source_differs": True,
        "n_envs_differs": True,
        "worker_cpus_differs": True,
    }
    assert screen["source_campaign"]["resource_samples"][0]["sha256"]
    assert "not a direct runtime forecast" in render_markdown(screen)


def test_runtime_screen_rejects_missing_training_phase_timings(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "incomplete"
    run_dir = campaign_dir / "runs" / "attempts" / "variant__seed42" / "attempt1" / "run"
    run_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text(
        json.dumps(
            {"jobs": [{"key": "variant__seed42", "seed": 42, "attempts": 1, "status": "failed"}]}
        ),
        encoding="utf-8",
    )
    (run_dir / "training_log.json").write_text(
        json.dumps({"timesteps_completed": 8192, "phase_timings_seconds": {"42": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete or invalid training/evaluation timings"):
        build_runtime_screen(campaign_dir)
