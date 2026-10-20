from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from scripts.research.thesis.evidence.analyze_paired_seed_results import analyze_paired_seed_metrics

from evaluation.metrics.matched_comparison import MATCH_FIELDS


def _payload() -> dict[str, Any]:
    pairs = []
    for seed, delta in zip((11, 12, 13, 14, 15), (1.0, 2.0, 3.0, 4.0, 5.0), strict=True):
        common = {
            "dataset_sha256": "a" * 64,
            "split_id": "train-v1",
            "features_sha256": "b" * 64,
            "reward_sha256": "c" * 64,
            "costs_sha256": "d" * 64,
            "execution_sha256": "e" * 64,
            "timesteps": 20_000_000,
            "seed": seed,
            "initial_capital": 100_000.0,
            "model_settings_sha256": "f" * 64,
            "environment_sha256": "1" * 64,
            "evaluation_start": "2024-01-01T00:00:00Z",
            "evaluation_end": "2025-01-01T00:00:00Z",
        }
        treatment_provenance = {**common, "features_sha256": "2" * 64}
        arms = {}
        for name, provenance, return_pct in (
            ("control", common, 0.5),
            ("treatment", treatment_provenance, 0.5 + delta),
        ):
            run_id = f"{name}-seed{seed}"
            arms[name] = {
                "run_id": run_id,
                "status": "succeeded",
                "evidence_role": "matched_confirmation",
                "provenance": {**provenance, "run_id": run_id},
                "metrics": {"net_return_pct": return_pct, "trade_count": 40 + seed},
            }
        pairs.append({"seed": seed, **arms})
    return {
        "schema": "aalto.paired_seed_metrics.v1",
        "comparison_id": "rq1-control-v-treatment",
        "intervention_field": "features_sha256",
        "metrics": ["net_return_pct", "trade_count"],
        "pairs": pairs,
    }


def test_paired_seed_analysis_keeps_pairs_and_is_reproducible() -> None:
    payload = _payload()
    first = analyze_paired_seed_metrics(payload, rng_seed=9, bootstrap_draws=500)
    second = analyze_paired_seed_metrics(payload, rng_seed=9, bootstrap_draws=500)

    assert first["schema"] == "aalto.paired_seed_statistics.v1"
    assert first["n_pairs"] == 5
    assert first["metrics"] == second["metrics"]
    assert len(first["pair_lineage"]) == 5
    metrics = first["metrics"]
    assert isinstance(metrics, dict)
    result = metrics["net_return_pct"]
    assert result["mean_treatment_minus_control"] == 3.0
    assert result["sign_flip_p_value"] == 0.0625
    assert result["sign_flip_method"] == "exact two-sided paired sign-flip"
    assert [row["seed"] for row in result["seed_level_values"]] == [11, 12, 13, 14, 15]
    assert result["paired_seed_bootstrap_ci_95pct"][0] <= 3.0
    assert result["paired_seed_bootstrap_ci_95pct"][1] >= 3.0


def test_paired_seed_analysis_rejects_mismatched_match_fields() -> None:
    payload = _payload()
    pairs = payload["pairs"]
    assert isinstance(pairs, list)
    pairs[0]["treatment"]["provenance"]["dataset_sha256"] = "9" * 64

    with pytest.raises(ValueError, match="mismatched provenance: dataset_sha256"):
        analyze_paired_seed_metrics(payload)


def test_paired_seed_analysis_rejects_uncompleted_or_selection_runs() -> None:
    for field, value in (("status", "interrupted"), ("evidence_role", "selection")):
        payload = deepcopy(_payload())
        payload["pairs"][0]["control"][field] = value
        with pytest.raises(ValueError):
            analyze_paired_seed_metrics(payload)


def test_paired_seed_analysis_requires_five_pairs_and_full_identity() -> None:
    payload = _payload()
    payload["pairs"] = payload["pairs"][:4]
    with pytest.raises(ValueError, match="at least 5"):
        analyze_paired_seed_metrics(payload)

    payload = _payload()
    del payload["pairs"][0]["control"]["provenance"][MATCH_FIELDS[0]]
    with pytest.raises(ValueError, match="missing match provenance"):
        analyze_paired_seed_metrics(payload)
