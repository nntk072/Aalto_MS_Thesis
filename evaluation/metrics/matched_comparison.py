"""Paired evaluation with explicit, matching research-run provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .bootstrap import CI, paired_metric_difference_ci

MATCH_FIELDS = (
    "dataset_sha256",
    "split_id",
    "features_sha256",
    "reward_sha256",
    "costs_sha256",
    "execution_sha256",
    "timesteps",
    "seed",
    "initial_capital",
    "model_settings_sha256",
    "environment_sha256",
    "evaluation_start",
    "evaluation_end",
)

# A matched intervention may change exactly one of these declared design
# dimensions. Dataset, split, costs, execution, budget, seed, capital, model
# settings and evaluation window must remain identical.
INTERVENTION_FIELDS = frozenset({"features_sha256", "reward_sha256", "environment_sha256"})


def compare_matched_returns(
    treatment: pd.Series,
    control: pd.Series,
    treatment_provenance: Mapping[str, Any],
    control_provenance: Mapping[str, Any],
    *,
    intervention_field: str | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    rng_seed: int = 42,
) -> CI:
    """Paired mean return difference only when the declared conditions match.

    Requires genuinely timestamp-aligned returns and independently recorded
    provenance. A comparison of a previously inspected holdout is not evidence
    of a new untouched OOS evaluation. When an intervention is declared, its
    provenance field must be one of the supported design dimensions and must
    differ between arms; every other required field still has to match.
    """
    if intervention_field is not None and intervention_field not in INTERVENTION_FIELDS:
        raise ValueError(f"Unsupported comparison intervention field: {intervention_field}")
    for field in MATCH_FIELDS:
        left = treatment_provenance.get(field)
        right = control_provenance.get(field)
        if field == intervention_field:
            if (
                left is None
                or right is None
                or left == right
                or (isinstance(left, str) and not left.strip())
                or (isinstance(right, str) and not right.strip())
            ):
                raise ValueError(
                    f"Declared intervention must have distinct, non-empty provenance: {field}"
                )
            continue
        if (
            left is None
            or right is None
            or left != right
            or (isinstance(left, str) and not left.strip())
            or (isinstance(right, str) and not right.strip())
        ):
            raise ValueError(f"Missing or mismatched comparison provenance: {field}")
    if not isinstance(treatment.index, pd.DatetimeIndex) or not isinstance(
        control.index, pd.DatetimeIndex
    ):
        raise ValueError("Paired series require timestamp indexes")  # noqa: TRY004 - existing API contract
    if treatment.index.hasnans or control.index.hasnans:
        raise ValueError("Paired timestamps must not be missing")
    if not treatment.index.is_monotonic_increasing or not control.index.is_monotonic_increasing:
        raise ValueError("Paired timestamps must be chronologically ordered")
    if not treatment.index.is_unique or not control.index.is_unique:
        raise ValueError("Paired series index must have unique timestamps")
    if not treatment.index.equals(control.index):
        raise ValueError("Paired series index must match exactly")
    if len(treatment) == 0:
        raise ValueError("Paired comparison requires at least one return observation")
    try:
        declared_start = pd.Timestamp(treatment_provenance["evaluation_start"])
        declared_end = pd.Timestamp(treatment_provenance["evaluation_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid declared evaluation period") from exc
    if declared_start != treatment.index[0] or declared_end != treatment.index[-1]:
        raise ValueError("Declared evaluation period must match the return index endpoints")
    if len(treatment) == 1:
        raise ValueError("Paired bootstrap comparison requires at least two return observations")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive for a confidence interval")
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("Confidence level must be finite and strictly between zero and one")
    return paired_metric_difference_ci(
        np.asarray(treatment, dtype=np.float64),
        np.asarray(control, dtype=np.float64),
        np.mean,
        n_boot=n_boot,
        level=level,
        rng_seed=rng_seed,
    )
