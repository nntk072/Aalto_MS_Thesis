"""Causal multi-TF alignment and US100/US500 join.

Higher-timeframe values are shifted by one HTF bar before forward-fill so M1
bar t only sees a *completed* higher-TF candle. Left-labeled bars spanning
``[T, T+period)`` become visible at ``T+period``.
"""

from __future__ import annotations

import pandas as pd

_TIMEFRAME_DURATION = {
    "M1": pd.Timedelta(minutes=1),
    "M3": pd.Timedelta(minutes=3),
    "M5": pd.Timedelta(minutes=5),
    "M10": pd.Timedelta(minutes=10),
    "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}


def timeframe_close_times(
    bars: pd.DataFrame,
    timeframe: str,
    scheduled_closes: dict[str, str] | None = None,
    session_start: str = "16:30",
) -> pd.Series:
    """Return bar close timestamps from open-stamped bars and the session calendar.

    Intraday bars normally close one timeframe after their open. If a bar
    overlaps a date's scheduled NY close, its availability is clipped to that
    known close. Daily bars keep their broker-midnight boundary.
    """
    tf = str(timeframe).upper()
    if tf not in _TIMEFRAME_DURATION:
        raise ValueError(f"unsupported timeframe for close-time calculation: {timeframe}")
    index = pd.DatetimeIndex(bars.index)
    closes = index + _TIMEFRAME_DURATION[tf]
    if tf == "D1" or not scheduled_closes or len(index) == 0:
        return pd.Series(closes, index=bars.index, name="availability_time")
    start = pd.Timestamp(f"2000-01-01 {session_start}").time()
    clipped = list(closes)
    for position, opened in enumerate(index):
        local_date = opened.strftime("%Y-%m-%d")
        scheduled = scheduled_closes.get(local_date)
        if scheduled is None or opened.time() < start:
            continue
        close_time = pd.Timestamp(f"{local_date} {scheduled}")
        if opened.tz is not None:
            close_time = close_time.tz_localize(opened.tz)
        if opened < close_time < closes[position]:
            clipped[position] = close_time
    return pd.Series(pd.DatetimeIndex(clipped), index=bars.index, name="availability_time")


def _prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df


def align_timeframes(
    m1: pd.DataFrame,
    higher: dict[str, pd.DataFrame],
    prefix: str = "",
    completed_bars: bool = True,
    availability_times: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Merge higher-TF DataFrames into the M1 spine causally.

    By default each higher-TF series is shifted one bar so M1 at time t sees
    the last fully closed HTF candle, never the still-forming one. Callers
    that already shift (Chain F FVG/IFVG) must pass ``completed_bars=False``
    or align without this helper.

    Parameters
    ----------
    m1:
        M1 DataFrame (reference spine); must have a tz-aware DatetimeIndex.
    higher:
        Dict mapping timeframe label (e.g. ``"M5"``) to the corresponding
        bar DataFrame.
    prefix:
        Optional column prefix (e.g. symbol name).
    completed_bars:
        If True (default), ``shift(1)`` before ffill.
    availability_times:
        Optional per-timeframe close timestamps indexed like ``higher[tf]``.
        A feature becomes visible on the M1 spine at its close timestamp. This
        handles gaps and shortened final bars without using the next observed
        HTF row as a proxy for the close.
    """
    result = m1.copy()
    if prefix:
        result = _prefix_cols(result, prefix)

    for tf, tf_df in higher.items():
        if tf_df is None or tf_df.empty:
            continue
        cols = {c: f"{prefix}_{tf}_{c}" if prefix else f"{tf}_{c}" for c in tf_df.columns}
        tf_renamed = tf_df.rename(columns=cols)
        close_times = (availability_times or {}).get(tf)
        if close_times is not None:
            close_times = pd.Series(close_times, index=tf_df.index)
            valid = close_times.notna()
            tf_renamed = tf_renamed.loc[valid].copy()
            tf_renamed.index = pd.DatetimeIndex(close_times.loc[valid])
            tf_renamed = tf_renamed.loc[~tf_renamed.index.duplicated(keep="last")]
        elif completed_bars:
            tf_renamed = tf_renamed.shift(1)
        tf_reindexed = tf_renamed.sort_index().reindex(result.index, method="ffill")
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
