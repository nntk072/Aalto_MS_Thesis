"""Minimal leakage detectors for thesis validity (T-03.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def label_shuffle_score(
    y: np.ndarray[Any, Any] | pd.Series,
    pred: np.ndarray[Any, Any] | pd.Series,
    *,
    n_perm: int = 32,
    seed: int = 0,
) -> dict[str, float]:
    """Compare |corr(y, pred)| to a label-permutation null.

    Returns the observed absolute correlation and the fraction of permutations
    with |corr| at least as large (two-sided Monte Carlo p-proxy).
    """
    yt = np.asarray(y, dtype=float).ravel()
    pt = np.asarray(pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(pt)
    yt, pt = yt[mask], pt[mask]
    if len(yt) < 8:
        return {"observed_abs_corr": float("nan"), "null_exceed_frac": float("nan")}

    def _abs_corr(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(abs(np.corrcoef(a, b)[0, 1]))

    observed = _abs_corr(yt, pt)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(yt)
        if _abs_corr(shuffled, pt) >= observed - 1e-15:
            exceed += 1
    return {
        "observed_abs_corr": observed,
        "null_exceed_frac": exceed / n_perm,
    }


def time_shift_feature_scores(
    feature: np.ndarray[Any, Any] | pd.Series,
    target: np.ndarray[Any, Any] | pd.Series,
    *,
    shifts: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Absolute corr of target with feature shifted *forward* by ``k`` bars.

    A causal feature should not improve when shifted forward into the future
    relative to the target (performance should not rise with +k).
    """
    f_arr = np.asarray(feature, dtype=float).ravel()
    y_arr = np.asarray(target, dtype=float).ravel()
    out: dict[str, float] = {}
    n = min(len(f_arr), len(y_arr))
    f = f_arr[:n]
    y = y_arr[:n]

    def _abs_corr(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 8:
            return float("nan")
        aa, bb = a[mask], b[mask]
        if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
            return 0.0
        return float(abs(np.corrcoef(aa, bb)[0, 1]))

    out["shift_0"] = _abs_corr(f, y)
    for k in shifts:
        if k <= 0 or k >= n:
            out[f"shift_{k}"] = float("nan")
            continue
        # feature available k bars earlier than its timestamp ⇒ align f[:-k] with y[k:]
        out[f"shift_{k}"] = _abs_corr(f[:-k], y[k:])
    return out
