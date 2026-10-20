#!/usr/bin/env python3
"""Summarize provenance-matched seed pairs without treating bars as seeds.

The percentile interval resamples paired seeds. It is separate from the
moving-block within-path interval in ``evaluation.metrics.bootstrap``.
The exact sign-flip test tests a zero-centered paired mean under sign symmetry;
it is not a substitute for the thesis validity, selection, or holdout gates.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluation.metrics.matched_comparison import INTERVENTION_FIELDS, MATCH_FIELDS

BOOTSTRAP_DRAWS = 10_000
EXACT_SIGN_FLIP_MAX_PAIRS = 15
MONTE_CARLO_SIGN_FLIP_DRAWS = 100_000
MIN_PAIRED_SEEDS = 5


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _validate_pair_input(
    payload: Any,
) -> tuple[
    str,
    str,
    list[int],
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.float64]],
]:
    if not isinstance(payload, dict) or payload.get("schema") != "aalto.paired_seed_metrics.v1":
        raise ValueError("input schema must be aalto.paired_seed_metrics.v1")
    comparison_id = payload.get("comparison_id")
    if not isinstance(comparison_id, str) or not comparison_id.strip():
        raise ValueError("comparison_id must be a non-empty string")
    intervention = payload.get("intervention_field")
    if intervention not in INTERVENTION_FIELDS:
        raise ValueError(f"unsupported intervention_field: {intervention!r}")
    metrics = payload.get("metrics")
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(name, str) or not name.strip() for name in metrics)
        or len(set(metrics)) != len(metrics)
    ):
        raise ValueError("metrics must be a non-empty list of unique names")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < MIN_PAIRED_SEEDS:
        raise ValueError(f"at least {MIN_PAIRED_SEEDS} completed matched seed pairs are required")

    seen_seeds: set[int] = set()
    seen_run_ids: set[str] = set()
    treatment_values: dict[str, list[float]] = {name: [] for name in metrics}
    control_values: dict[str, list[float]] = {name: [] for name in metrics}
    seeds: list[int] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"pairs[{index}] must be an object")
        seed = pair.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in seen_seeds:
            raise ValueError(f"pairs[{index}].seed must be a unique integer")
        seen_seeds.add(seed)
        seeds.append(seed)
        arms: dict[str, dict[str, Any]] = {}
        for arm_name in ("control", "treatment"):
            arm = pair.get(arm_name)
            if not isinstance(arm, dict):
                raise ValueError(f"pairs[{index}].{arm_name} must be an object")
            run_id = arm.get("run_id")
            if not isinstance(run_id, str) or not run_id.strip() or run_id in seen_run_ids:
                raise ValueError(f"pairs[{index}].{arm_name}.run_id must be unique and non-empty")
            seen_run_ids.add(run_id)
            if arm.get("status") != "succeeded":
                raise ValueError(f"run {run_id} is not succeeded")
            if arm.get("evidence_role") not in {"matched_confirmation", "robustness"}:
                raise ValueError(f"run {run_id} is not confirmatory-pair or robustness evidence")
            provenance = arm.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"run {run_id} is missing provenance")
            missing = [field for field in MATCH_FIELDS if field not in provenance]
            if missing:
                raise ValueError(f"run {run_id} is missing match provenance: {missing}")
            if provenance["seed"] != seed:
                raise ValueError(f"run {run_id} seed does not match pair seed")
            if "run_id" in provenance and provenance["run_id"] != run_id:
                raise ValueError(f"run {run_id} provenance identity mismatch")
            if (
                "evidence_role" in provenance
                and provenance["evidence_role"] != arm["evidence_role"]
            ):
                raise ValueError(f"run {run_id} evidence role mismatch")
            arms[arm_name] = {
                "run_id": run_id,
                "provenance": provenance,
                "metrics": arm.get("metrics"),
            }

        left = arms["control"]["provenance"]
        right = arms["treatment"]["provenance"]
        for field in MATCH_FIELDS:
            left_value, right_value = left[field], right[field]
            if field == intervention:
                if (
                    left_value == right_value
                    or left_value in (None, "")
                    or right_value in (None, "")
                ):
                    raise ValueError(f"declared intervention must differ: {field}")
            elif left_value != right_value or left_value in (None, ""):
                raise ValueError(f"paired runs have missing or mismatched provenance: {field}")

        for arm_name, destination in (("control", control_values), ("treatment", treatment_values)):
            arm_metrics = arms[arm_name]["metrics"]
            if not isinstance(arm_metrics, dict):
                raise ValueError(f"run {arms[arm_name]['run_id']} is missing metrics")
            if set(arm_metrics) != set(metrics):
                raise ValueError(f"run {arms[arm_name]['run_id']} metric names do not match input")
            for metric in metrics:
                destination[metric].append(
                    _finite_number(
                        arm_metrics[metric], field=f"{arms[arm_name]['run_id']}.{metric}"
                    )
                )

    controls = {metric: np.asarray(control_values[metric], dtype=np.float64) for metric in metrics}
    treatments = {
        metric: np.asarray(treatment_values[metric], dtype=np.float64) for metric in metrics
    }
    return comparison_id, intervention, seeds, controls, treatments


def _sign_flip_p_value(differences: NDArray[np.float64], *, rng_seed: int) -> tuple[float, str]:
    observed = abs(float(differences.mean()))
    n = len(differences)
    if n <= EXACT_SIGN_FLIP_MAX_PAIRS:
        extreme = sum(
            abs(float(np.mean(differences * signs))) >= observed - 1e-12
            for signs in itertools.product((-1.0, 1.0), repeat=n)
        )
        return extreme / (2**n), "exact two-sided paired sign-flip"
    rng = np.random.default_rng(rng_seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(MONTE_CARLO_SIGN_FLIP_DRAWS, n))
    means = np.abs((signs * differences).mean(axis=1))
    extreme = int(np.count_nonzero(means >= observed - 1e-12))
    return (extreme + 1) / (
        MONTE_CARLO_SIGN_FLIP_DRAWS + 1
    ), "Monte Carlo two-sided paired sign-flip"


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def analyze_paired_seed_metrics(
    payload: Any, *, rng_seed: int = 42, bootstrap_draws: int = BOOTSTRAP_DRAWS
) -> dict[str, Any]:
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    comparison_id, intervention, seeds, controls, treatments = _validate_pair_input(payload)
    rng = np.random.default_rng(rng_seed)
    raw_p: dict[str, float] = {}
    rows: dict[str, dict[str, Any]] = {}
    for metric, treatment in treatments.items():
        control = controls[metric]
        values = treatment - control
        sample_indices = rng.integers(0, len(values), size=(bootstrap_draws, len(values)))
        bootstrap_means = values[sample_indices].mean(axis=1)
        p_value, p_method = _sign_flip_p_value(values, rng_seed=rng_seed)
        raw_p[metric] = p_value
        rows[metric] = {
            "paired_seed_count": len(values),
            "seeds": seeds,
            "seed_level_values": [
                {
                    "seed": seed,
                    "control": float(control_value),
                    "treatment": float(treatment_value),
                    "difference": float(difference),
                }
                for seed, control_value, treatment_value, difference in zip(
                    seeds, control, treatment, values, strict=True
                )
            ],
            "mean_treatment_minus_control": float(values.mean()),
            "median_treatment_minus_control": float(np.median(values)),
            "sd_of_paired_differences": float(values.std(ddof=1)),
            "min_paired_difference": float(values.min()),
            "max_paired_difference": float(values.max()),
            "paired_seed_bootstrap_ci_95pct": [
                float(x) for x in np.quantile(bootstrap_means, [0.025, 0.975])
            ],
            "sign_flip_p_value": p_value,
            "sign_flip_method": p_method,
        }
    adjusted_p = _holm_adjust(raw_p)
    for metric, row in rows.items():
        row["holm_adjusted_p_value"] = adjusted_p[metric]
    pair_lineage = [
        {
            "seed": pair["seed"],
            **{
                arm_name: {
                    "run_id": pair[arm_name]["run_id"],
                    "status": pair[arm_name]["status"],
                    "evidence_role": pair[arm_name]["evidence_role"],
                    "provenance": pair[arm_name]["provenance"],
                }
                for arm_name in ("control", "treatment")
            },
        }
        for pair in payload["pairs"]
    ]
    return {
        "schema": "aalto.paired_seed_statistics.v1",
        "comparison_id": comparison_id,
        "intervention_field": intervention,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "n_pairs": len(seeds),
        "pair_lineage": pair_lineage,
        "bootstrap": {"unit": "paired_seed", "draws": bootstrap_draws, "rng_seed": rng_seed},
        "multiple_testing": {"method": "Holm", "metric_count": len(rows)},
        "metrics": rows,
        "interpretation_limits": [
            "The interval resamples paired seeds; it does not represent within-path temporal uncertainty.",
            "Use the moving-block paired-return analysis separately for temporal-path uncertainty.",
            "This tool does not verify holdout freshness, selection discipline, full-split exposure, or raw metric reconstruction.",
            "A small number of seeds yields coarse intervals and discrete exact-test p-values; report the seed-level differences.",
            "A sign-flip test assumes symmetry of paired differences under its null and does not establish causality by itself.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Matched seed-pair JSON")
    parser.add_argument("--output", required=True, type=Path, help="New statistics JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap/sign-flip RNG seed")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace statistics output: {args.output}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze_paired_seed_metrics(
        payload, rng_seed=args.seed, bootstrap_draws=args.bootstrap_draws
    )
    report["input_path"] = str(args.input.resolve())
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_pairs": report["n_pairs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
