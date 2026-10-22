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
import math
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
EVIDENCE_ROLES = frozenset({"matched_confirmation", "matched_exploratory", "robustness"})


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction (Numerical Recipes form)."""
    max_iterations = 200
    epsilon = 3.0e-14
    minimum = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + aa / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + aa / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_freedom / (degrees_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(x, degrees_freedom / 2.0, 0.5)
    return 1.0 - tail if value > 0.0 else tail


def _student_t_critical_value(probability: float, degrees_freedom: int) -> float:
    """Invert the Student-t CDF without an undeclared SciPy dependency."""
    if not 0.5 < probability < 1.0:
        raise ValueError("Student-t quantile probability must be between 0.5 and 1")
    low = 0.0
    high = 1.0
    while _student_t_cdf(high, degrees_freedom) < probability:
        high *= 2.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _student_t_cdf(midpoint, degrees_freedom) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


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
    evidence_role = payload.get("evidence_role")
    if evidence_role is not None and evidence_role not in EVIDENCE_ROLES:
        raise ValueError(f"unsupported evidence_role: {evidence_role!r}")
    if (
        payload.get("evaluation_exposure") == "exposed_report_only"
        and evidence_role != "matched_exploratory"
    ):
        raise ValueError("exposed report-only results cannot be confirmatory evidence")
    expected_seeds = payload.get("expected_seeds")
    if expected_seeds is not None and (
        not isinstance(expected_seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in expected_seeds)
        or len(expected_seeds) != len(set(expected_seeds))
    ):
        raise ValueError("expected_seeds must be a list of unique integer seed IDs")
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
    seen_roles: set[str] = set()
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
            arm_role = arm.get("evidence_role")
            if arm_role not in EVIDENCE_ROLES:
                raise ValueError(f"run {run_id} has an unsupported evidence role")
            seen_roles.add(arm_role)
            if evidence_role is not None and arm_role != evidence_role:
                raise ValueError(f"run {run_id} evidence role does not match the comparison")
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
    if expected_seeds is not None and seeds != expected_seeds:
        raise ValueError(f"completed seed IDs do not match expected_seeds: {seeds}")
    if len(seen_roles) != 1:
        raise ValueError("all paired runs must use the same evidence role")
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
        degrees_freedom = len(values) - 1
        sample_sd = float(values.std(ddof=1))
        standard_error = sample_sd / float(np.sqrt(len(values)))
        critical_value = _student_t_critical_value(0.975, degrees_freedom)
        mean_difference = float(values.mean())
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
            "mean_treatment_minus_control": mean_difference,
            "median_treatment_minus_control": float(np.median(values)),
            "sd_of_paired_differences": sample_sd,
            "paired_student_t_degrees_freedom": degrees_freedom,
            "paired_student_t_critical_value_95pct": critical_value,
            "paired_student_t_standard_error": standard_error,
            "paired_student_t_ci_95pct": [
                mean_difference - critical_value * standard_error,
                mean_difference + critical_value * standard_error,
            ],
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
        "evidence_role": payload.get("evidence_role")
        or payload["pairs"][0]["control"]["evidence_role"],
        "evaluation_exposure": payload.get("evaluation_exposure"),
        "expected_seeds": payload.get("expected_seeds"),
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
