"""Rolling train-only z-score normalisation (no data leakage).

The scaler is fit only on training bars (mask=True) and applied to all bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(
    df: pd.DataFrame,
    window: int = 252,
    train_mask: pd.Series | None = None,
    mode: str = "legacy_masked_window",
) -> pd.DataFrame:
    """Apply a rolling z-score to each column.

    Parameters
    ----------
    df:
        Feature DataFrame.
    window:
        Look-back window for mean/std (bars).
    train_mask:
        Boolean Series; if provided, statistics are computed only from rows
        where mask=True (the training fold).  Test rows are normalised using
        the last known training statistics.
    mode:
        ``legacy_masked_window`` retains the historical masked calendar-window
        convention for checkpoint compatibility. ``frozen_fold`` computes
        rolling statistics across training observations and forward-fills the
        final fitted mean/std throughout validation, without validation input.

    Returns
    -------
    Normalised DataFrame (same shape as input).
    """
    if mode not in {"legacy_masked_window", "frozen_fold"}:
        raise ValueError(f"Unknown normalization mode: {mode}")
    if mode == "frozen_fold":
        if train_mask is None:
            raise ValueError("frozen_fold normalization requires a training mask")
        mask = train_mask.reindex(df.index)
        if mask.isna().any() or not bool(mask.any()):
            raise ValueError("frozen_fold requires aligned nonempty training observations")
        train_df = df.loc[mask.astype(bool)]
        # Statistics are computed on the training fold alone, then projected
        # causally onto the full index.  No validation sample can change the fit.
        mu = train_df.rolling(window, min_periods=1).mean().reindex(df.index).ffill()
        sigma = train_df.rolling(window, min_periods=2).std(ddof=0).reindex(df.index).ffill()
    elif train_mask is not None:
        # Forward-fill stats computed from training rows only
        train_df = df.where(train_mask)
        mu = train_df.rolling(window, min_periods=1).mean().ffill()
        sigma = train_df.rolling(window, min_periods=2).std(ddof=0).ffill()
    else:
        mu = df.rolling(window, min_periods=1).mean()
        sigma = df.rolling(window, min_periods=2).std(ddof=0)

    sigma = sigma.replace(0.0, np.nan)
    return (df - mu) / sigma
