"""Train / test date-based split for bars and features.

Prefer passing ``train_mask`` into ``build_features`` so rolling z-score is
fit-scoped (TI-4). Slicing after a fully causal build remains OK for indicators
with short windows; it is not a substitute for train-scoped normalisation under
walk-forward.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


def split_train_test(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    train_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split aligned bars + features into train (≤ *train_end*) and test (≥ *test_start*).

    Parameters
    ----------
    bars, features:
        Must share the same DatetimeIndex (timezone-aware).
    train_end:
        Last date included in the training set (inclusive, full day).
    test_start:
        First date included in the test set (inclusive, midnight).

    Returns
    -------
    train_bars, test_bars, train_features, test_features
    """
    # Align to common index first
    common = bars.index.intersection(features.index)
    b = bars.loc[common]
    f = features.loc[common]

    # Build timezone-aware boundary timestamps matching the index tz
    tz = cast(pd.DatetimeIndex, b.index).tz

    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    if tz is not None:
        if train_end_ts.tzinfo is None:
            train_end_ts = train_end_ts.tz_localize(tz)
        else:
            train_end_ts = train_end_ts.tz_convert(tz)
        if test_start_ts.tzinfo is None:
            test_start_ts = test_start_ts.tz_localize(tz)
        else:
            test_start_ts = test_start_ts.tz_convert(tz)

    # Include the full last training day (end of 2025-12-31)
    train_end_ts = train_end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    train_mask = b.index <= train_end_ts
    test_mask = b.index >= test_start_ts

    return b[train_mask], b[test_mask], f[train_mask], f[test_mask]


def split_bars(
    bars: pd.DataFrame,
    train_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single bar frame on the same date boundaries as ``split_train_test``."""
    train, test, _, _ = split_train_test(bars, bars, train_end, test_start)
    return train, test


def get_split_config(cfg: Any) -> tuple[str, str]:
    """Extract *train_end* / *test_start* from config with fallback defaults."""
    try:
        train_end = str(cfg.data.split.train_end)
        test_start = str(cfg.data.split.test_start)
    except Exception:
        train_end = "2025-12-31"
        test_start = "2026-01-01"
    return train_end, test_start


def make_train_mask(
    index: pd.DatetimeIndex,
    train_end: str,
) -> pd.Series:
    """Boolean mask of bars on or before the inclusive ``train_end`` calendar day."""
    tz = index.tz
    train_end_ts = pd.Timestamp(train_end)
    if tz is not None:
        if train_end_ts.tzinfo is None:
            train_end_ts = train_end_ts.tz_localize(tz)
        else:
            train_end_ts = train_end_ts.tz_convert(tz)
    train_end_ts = train_end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return pd.Series(index <= train_end_ts, index=index)
