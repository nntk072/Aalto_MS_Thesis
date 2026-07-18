"""Clean raw bar DataFrames: dedup, sort, gap flag."""
from __future__ import annotations

import pandas as pd


def clean(df: pd.DataFrame, tz: str = "Etc/GMT-3") -> pd.DataFrame:
    """Deduplicate, sort, localise timezone, and add a ``gap_flag`` column.

    Parameters
    ----------
    df:
        Raw bar DataFrame from :func:`quant_rl.data.loader.load_bars`.
    tz:
        Broker server timezone string (IANA, default UTC+3).
    """
    df = df.copy()
    # Remove duplicate timestamps (keep last – most recent update wins)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    # Localise (timestamps are naive broker-local)
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    # Gap flag: True when gap to previous bar is > 2× the modal bar duration
    delta = df.index.to_series().diff()
    modal = delta.mode().iloc[0]
    df["gap_flag"] = delta > (2 * modal)
    df["gap_flag"] = df["gap_flag"].fillna(False)
    return df
