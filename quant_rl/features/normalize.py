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

    Returns
    -------
    Normalised DataFrame (same shape as input).
    """
    if train_mask is not None:
        # Forward-fill stats computed from training rows only
        train_df = df.where(train_mask)
        mu = train_df.rolling(window, min_periods=1).mean().ffill()
        sigma = train_df.rolling(window, min_periods=2).std(ddof=0).ffill()
    else:
        mu = df.rolling(window, min_periods=1).mean()
        sigma = df.rolling(window, min_periods=2).std(ddof=0)

    sigma = sigma.replace(0.0, np.nan)
    return (df - mu) / sigma
