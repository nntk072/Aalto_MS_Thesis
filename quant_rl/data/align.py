"""Causal multi-TF alignment and US100/US500 join.

All higher-timeframe data is *forward-filled* into the M1 spine so that
features at bar t use only information available at or before t (causal).
"""

from __future__ import annotations

import pandas as pd


def _prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df


def align_timeframes(
    m1: pd.DataFrame,
    higher: dict[str, pd.DataFrame],
    prefix: str = "",
) -> pd.DataFrame:
    """Merge higher-TF DataFrames into the M1 spine causally.

    Each higher-TF bar value is forward-filled onto the M1 index so that
    the M1 bar at time t only sees the most recent *completed* higher-TF bar
    whose open-time ≤ t.

    Parameters
    ----------
    m1:
        M1 DataFrame (reference spine); must have a tz-aware DatetimeIndex.
    higher:
        Dict mapping timeframe label (e.g. ``"M5"``) to the corresponding
        bar DataFrame.
    prefix:
        Optional column prefix (e.g. symbol name).
    """
    result = m1.copy()
    if prefix:
        result = _prefix_cols(result, prefix)

    for tf, tf_df in higher.items():
        if tf_df is None or tf_df.empty:
            continue
        cols = {c: f"{prefix}_{tf}_{c}" if prefix else f"{tf}_{c}" for c in tf_df.columns}
        tf_renamed = tf_df.rename(columns=cols)
        # reindex onto M1 spine, ffill (causal)
        tf_reindexed = tf_renamed.reindex(result.index, method="ffill")
        result = pd.concat([result, tf_reindexed], axis=1)

    return result


def join_symbols(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    secondary_prefix: str = "sec",
) -> pd.DataFrame:
    """Join secondary (e.g. US500) columns onto the primary (US100) spine.

    Uses ``merge_asof`` (causal: secondary bar must be ≤ primary bar time).
    """
    sec = secondary.copy()
    sec = _prefix_cols(sec, secondary_prefix)
    merged = pd.merge_asof(
        primary.sort_index().reset_index(),
        sec.sort_index().reset_index(),
        on="datetime",
        direction="backward",
        suffixes=("", f"_{secondary_prefix}"),
    )
    merged = merged.set_index("datetime")
    return merged
