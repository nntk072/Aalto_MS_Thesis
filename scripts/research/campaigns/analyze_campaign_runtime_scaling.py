#!/usr/bin/env python3
"""Derive a labeled PPO runtime screen from completed campaign artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_worker_timings(campaign_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load PPO and final-evaluation timings for every campaign worker."""
    campaign_path = campaign_dir / "campaign.json"
    campaign = _read_json(campaign_path)
    jobs = campaign.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("campaign manifest has no jobs")

    results: list[dict[str, Any]] = []
    for job in sorted(jobs, key=lambda item: str(item.get("key", ""))):
        if not isinstance(job, dict) or not isinstance(job.get("key"), str):
            raise ValueError("campaign manifest contains an invalid job record")
        key = job["key"]
        attempt = int(job.get("attempts", 0))
        attempt_dir = campaign_dir / "runs" / "attempts" / key / f"attempt{attempt}"
        candidates = sorted(attempt_dir.rglob("training_log.json"))
        if len(candidates) != 1:
            raise ValueError(f"expected one training_log.json for {key}, found {len(candidates)}")
        log_path = candidates[0]
        training = _read_json(log_path)
        completed_steps = training.get("timesteps_completed")
        if not isinstance(completed_steps, int) or completed_steps <= 0:
            raise ValueError(f"invalid completed timestep count for {key}")
        seed = str(job.get("seed"))
        phase_groups = training.get("phase_timings_seconds")
        phases = phase_groups.get(seed) if isinstance(phase_groups, dict) else None
        if not isinstance(phases, dict):
            raise ValueError(f"missing per-seed phase timings for {key}, seed {seed}")
        ppo_seconds = phases.get("ppo_learn")
        test_seconds = phases.get("test_evaluation")
        train_seconds = phases.get("train_evaluation")
        if not isinstance(ppo_seconds, (float, int)) or ppo_seconds <= 0:
            raise ValueError(f"incomplete or invalid training/evaluation timings for {key}")
        if not isinstance(test_seconds, (float, int)) or test_seconds <= 0:
            raise ValueError(f"incomplete or invalid training/evaluation timings for {key}")
        if not isinstance(train_seconds, (float, int)) or train_seconds <= 0:
            raise ValueError(f"incomplete or invalid training/evaluation timings for {key}")
        results.append(
            {
                "run_key": key,
                "seed": int(job["seed"]),
                "attempt": attempt,
                "status": str(job.get("status", "unknown")),
                "completed_steps": completed_steps,
                "ppo_learn_seconds": float(ppo_seconds),
                "effective_steps_per_second": completed_steps / float(ppo_seconds),
                "test_evaluation_seconds": float(test_seconds),
                "train_evaluation_seconds": float(train_seconds),
                "final_evaluation_seconds": float(test_seconds) + float(train_seconds),
                "training_log": str(log_path.relative_to(campaign_dir)),
                "training_log_sha256": file_sha256(log_path),
            }
        )
    return campaign, results


def build_runtime_screen(
    campaign_dir: Path,
    *,
    target_steps: int = 20_000_000,
    deadline_hours: float = 5.75,
    target_data_source: str = "frames",
    target_n_envs: int = 4,
    target_worker_cpus: int = 3,
) -> dict[str, Any]:
    """Build a linear runtime screen while retaining the sample/target mismatch."""
    if target_steps <= 0 or deadline_hours <= 0 or target_n_envs <= 0 or target_worker_cpus <= 0:
        raise ValueError("target steps, deadline, env count, and worker CPUs must be positive")
    campaign, workers = load_worker_timings(campaign_dir)
    required_fps = target_steps / (deadline_hours * 3600)
    for worker in workers:
        sample_steps = worker["completed_steps"]
        sample_ppo = worker["ppo_learn_seconds"]
        projected_train = sample_ppo * target_steps / sample_steps
        projected_total = projected_train + worker["final_evaluation_seconds"]
        worker["projected_target_train_seconds_at_sample_rate"] = projected_train
        worker["projected_target_total_seconds_at_sample_rate"] = projected_total

    throughputs = [item["effective_steps_per_second"] for item in workers]
    training_hours = [
        item["projected_target_train_seconds_at_sample_rate"] / 3600 for item in workers
    ]
    total_hours = [item["projected_target_total_seconds_at_sample_rate"] / 3600 for item in workers]
    eval_seconds = [item["final_evaluation_seconds"] for item in workers]
    campaign_path = campaign_dir / "campaign.json"
    resource_files = sorted((campaign_dir / "logs").glob("resources_*.csv"))
    return {
        "schema_version": 1,
        "interpretation": "linear screening calculation; not a direct runtime forecast",
        "source_campaign": {
            "campaign_id": campaign.get("campaign_id", campaign_dir.name),
            "manifest_path": str(campaign_path),
            "manifest_sha256": file_sha256(campaign_path),
            "requested_steps": campaign.get("timesteps"),
            "data_source": campaign.get("data_source"),
            "n_envs": campaign.get("n_envs"),
            "worker_count": campaign.get("workers"),
            "worker_cpus": campaign.get("worker_cpus"),
            "worker_gpu_type": campaign.get("worker_gpu_type"),
            "worker_gpus_per_node": campaign.get("worker_gpus_per_node"),
            "worker_nodes": campaign.get("worker_nodes"),
            "runtime": campaign.get("runtime", {}),
            "resource_samples": [
                {"path": str(path), "sha256": file_sha256(path)} for path in resource_files
            ],
            "worker_results": workers,
        },
        "target_campaign": {
            "requested_steps_per_worker": int(target_steps),
            "deadline_hours": float(deadline_hours),
            "data_source": str(target_data_source),
            "n_envs": int(target_n_envs),
            "worker_cpus": int(target_worker_cpus),
            "required_training_steps_per_second_per_worker": required_fps,
        },
        "derived_screen": {
            "worker_count": len(workers),
            "effective_steps_per_second": {
                "min": min(throughputs),
                "median": statistics.median(throughputs),
                "max": max(throughputs),
            },
            "linear_training_hours_at_sample_rate": {
                "min": min(training_hours),
                "median": statistics.median(training_hours),
                "max": max(training_hours),
            },
            "one_final_train_test_evaluation_pair_hours": {
                "min": min(eval_seconds) / 3600,
                "median": statistics.median(eval_seconds) / 3600,
                "max": max(eval_seconds) / 3600,
            },
            "linear_training_plus_final_evaluation_hours_at_sample_rate": {
                "min": min(total_hours),
                "median": statistics.median(total_hours),
                "max": max(total_hours),
            },
            "required_throughput_divided_by_sample_median": (
                required_fps / statistics.median(throughputs)
            ),
            "required_throughput_divided_by_sample_max": required_fps / max(throughputs),
            "sample_target_mismatch": {
                "data_source_differs": campaign.get("data_source") != target_data_source,
                "n_envs_differs": campaign.get("n_envs") != target_n_envs,
                "worker_cpus_differs": campaign.get("worker_cpus") != target_worker_cpus,
            },
        },
    }


def render_markdown(screen: dict[str, Any]) -> str:
    sample = screen["source_campaign"]
    target = screen["target_campaign"]
    derived = screen["derived_screen"]
    fps = derived["effective_steps_per_second"]
    train = derived["linear_training_hours_at_sample_rate"]
    evaluation = derived["one_final_train_test_evaluation_pair_hours"]
    total = derived["linear_training_plus_final_evaluation_hours_at_sample_rate"]
    required = target["required_training_steps_per_second_per_worker"]
    return "\n".join(
        [
            "# Campaign Runtime Scaling Screen",
            "",
            "**Interpretation:** linear screening calculation, not a direct runtime forecast. The sample and target use different worker settings; do not use these numbers as a completion-time promise.",
            "",
            f"- Sample campaign: `{sample['campaign_id']}`; {sample['worker_count']} workers, {sample['requested_steps']} requested steps, `{sample['data_source']}`, `n_envs={sample['n_envs']}`, `{sample['worker_cpus']}` CPUs per worker.",
            f"- Target: {target['requested_steps_per_worker']} steps per worker, `{target['data_source']}`, `n_envs={target['n_envs']}`, `{target['worker_cpus']}` CPUs per worker, {target['deadline_hours']:.1f} hours.",
            f"- Sample per-worker throughput, min/median/max: {fps['min']:.1f} / {fps['median']:.1f} / {fps['max']:.1f} steps/s.",
            f"- Linear target PPO training time, min/median/max: {train['min']:.2f} / {train['median']:.2f} / {train['max']:.2f} hours.",
            f"- One final train/test evaluation pair, min/median/max: {evaluation['min'] * 60:.1f} / {evaluation['median'] * 60:.1f} / {evaluation['max'] * 60:.1f} minutes.",
            f"- Training plus that evaluation pair at the sample rate, min/median/max: {total['min']:.2f} / {total['median']:.2f} / {total['max']:.2f} hours.",
            f"- Single-allocation training-window threshold: {required:.1f} steps/s per worker for {target['deadline_hours']:.2f} hours; {derived['required_throughput_divided_by_sample_median']:.1f}× the sample median and {derived['required_throughput_divided_by_sample_max']:.1f}× the sample maximum.",
            f"- Supervisor allocation windows suggested by the same-rate median/maximum total-time screen: {total['median'] / target['deadline_hours']:.2f} / {total['max'] / target['deadline_hours']:.2f}; queue delays and the n_env/backend mismatch can increase these counts.",
            "",
            "These projections omit startup/feature loading, checkpoint export, queue delay, and retries. The sample's final evaluation time is added once rather than scaled with training steps.",
            "",
            "See `18way_runtime_scaling_screen_2026-10-14.md` for the verified interpretation, resource observations, and launch-readiness decision.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=20_000_000)
    parser.add_argument("--deadline-hours", type=float, default=5.75)
    parser.add_argument("--target-data-source", default="frames")
    parser.add_argument("--target-n-envs", type=int, default=4)
    parser.add_argument("--target-worker-cpus", type=int, default=3)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    screen = build_runtime_screen(
        args.campaign_dir,
        target_steps=args.target_steps,
        deadline_hours=args.deadline_hours,
        target_data_source=args.target_data_source,
        target_n_envs=args.target_n_envs,
        target_worker_cpus=args.target_worker_cpus,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(screen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(screen), encoding="utf-8")
    print(
        f"workers={screen['derived_screen']['worker_count']} json={args.out_json} md={args.out_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
