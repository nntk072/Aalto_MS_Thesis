"""NY-session filter and session-day ID assignment.

The New York session proxy is 16:30–23:00 in broker server time (UTC+3 by
default).  A config flag `dst_flag` documents that ~1 h DST shifts can occur
near DST transitions; we do NOT auto-correct them but flag the caveat.
"""

from __future__ import annotations

from typing import cast

import pandas as pd


def filter_session(
    df: pd.DataFrame,
    start: str = "16:30",
    end: str = "23:00",
) -> pd.DataFrame:
    """Keep only bars whose timestamp falls within the NY session window.

    Parameters
    ----------
    df:
        Timezone-aware bar DataFrame (index must be tz-aware after cleaning).
    start / end:
        Session window boundaries as HH:MM strings, in the DataFrame's tz.
    """
    df = df.copy()
    index = cast(pd.DatetimeIndex, df.index)
    t = index.time
    start_t = pd.Timestamp(f"2000-01-01 {start}").time()
    end_t = pd.Timestamp(f"2000-01-01 {end}").time()
    mask = (t >= start_t) & (t <= end_t)
    result = df[mask]
    return result


def add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``session_id`` integer column: unique per calendar day."""
    df = df.copy()
    # Use tz-naive date for grouping
    dates = cast(pd.DatetimeIndex, df.index).normalize()
    # Map each unique date to an integer ID
    unique_dates = sorted(set(dates))
    date_to_id = {d: i for i, d in enumerate(unique_dates)}
    df["session_id"] = dates.map(date_to_id)
    return df
