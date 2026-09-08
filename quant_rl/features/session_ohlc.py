"""Causal CT-anchored session levels: per-session OHLC, prior-period H/L,
range quadrants.

Session windows are defined in wall-clock ``America/Chicago`` (CT) time and
resolved per bar via tz-aware conversion (``DatetimeIndex.tz_convert``), so
US DST transitions are handled by the tz database on every bar. Never map a
CT ``HH:MM`` boundary into the data timezone once and reuse the result as a
static string: the data feed tz (e.g. fixed ``Etc/GMT-3``) has no DST rules,
so a static mapping drifts by one hour for weeks around each US transition.

Levels follow "live/running" semantics (matching
``features.structure.detect_session_levels``): O/H/L/C update bar-by-bar
while the session forms and are then held until the next session's window
starts. Bars before the first session in the data are NaN.

The CT-anchored levels are additive to the legacy broker-tz levels in
``features.structure`` (``asian_high``/``asian_low``/``london_high``/
``london_low``); new columns carry a ``_ct`` suffix so both can coexist.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd


def _parse_hhmm(value: str) -> dt.time:
    """Parse an ``HH:MM`` string into a ``datetime.time``."""
    return pd.Timestamp(f"2000-01-01 {value}").time()


def session_mask(
    idx: pd.DatetimeIndex,
    start_ct: str,
    end_ct: str,
    session_tz: str = "America/Chicago",
) -> pd.Series:
    """Boolean mask of bars whose local wall-clock time falls in ``[start, end)``.

    Each bar's timestamp is converted into ``session_tz`` so DST rules are
    resolved per bar by the tz database (see module docstring).
    """
    if idx.tz is None:
        raise ValueError("index must be timezone-aware (data tz), got naive index")
    t = idx.tz_convert(session_tz).time
    start_t = _parse_hhmm(start_ct)
    end_t = _parse_hhmm(end_ct)
    if start_t <= end_t:
        mask = (t >= start_t) & (t < end_t)
    else:  # window crosses midnight in session_tz
        mask = (t >= start_t) | (t < end_t)
    return pd.Series(mask, index=idx)


def _session_dates(idx_ct: pd.DatetimeIndex, start_t: dt.time, end_t: dt.time) -> pd.Series:
    """Calendar date in session_tz each bar's session window belongs to.

    For windows crossing midnight (not the case for Asian/London/NY as
    configured), bars before ``end_t`` belong to the previous calendar date.
    """
    dates = pd.Series(idx_ct.date, index=idx_ct)
    if start_t > end_t:
        prev_day = pd.Series([d - dt.timedelta(days=1) for d in idx_ct.date], index=idx_ct)
        belongs_to_next = pd.Series([t >= start_t for t in idx_ct.time], index=idx_ct)
        dates = dates.where(belongs_to_next, prev_day)
    return dates


def session_ohlc(
    df: pd.DataFrame,
    start_ct: str,
    end_ct: str,
    session_tz: str = "America/Chicago",
    prefix: str = "session",
) -> pd.DataFrame:
    """Causal per-session OHLC levels for the ``[start_ct, end_ct)`` window.

    Live/running semantics: within the window, ``high``/``low`` are the
    running max/min of the session so far, ``open`` is the session's first
    open and ``close`` the current bar's close. After the window ends the
    final session O/H/L/C is held (forward-filled) until the next session's
    window starts; bars before the first session are NaN — never 0.

    Parameters
    ----------
    df:
        OHLC bars with a tz-aware DatetimeIndex in the data tz.
    start_ct / end_ct:
        Window boundaries as ``HH:MM`` strings in ``session_tz`` (end exclusive).
    session_tz:
        IANA tz the window is defined in (DST-aware; default America/Chicago).
    prefix:
        Column prefix; returns ``{prefix}_open/_high/_low/_close``.
    """
    idx = pd.DatetimeIndex(df.index)
    mask = session_mask(idx, start_ct, end_ct, session_tz)
    idx_ct = idx.tz_convert(session_tz)
    grp = _session_dates(idx_ct, _parse_hhmm(start_ct), _parse_hhmm(end_ct)).where(mask)

    open_in = df["open"].where(mask)
    high_in = df["high"].where(mask)
    low_in = df["low"].where(mask)
    close_in = df["close"].where(mask)

    return pd.DataFrame(
        {
            f"{prefix}_open": open_in.groupby(grp).transform("first"),
            f"{prefix}_high": high_in.groupby(grp).cummax(),
            f"{prefix}_low": low_in.groupby(grp).cummin(),
            f"{prefix}_close": close_in,
        },
        index=idx,
    ).ffill()


def prior_period_high_low(
    df: pd.DataFrame,
    freq: str,
    tz: str = "America/Chicago",
    prefix: str | None = None,
) -> pd.DataFrame:
    """Previous *completed* calendar-period high/low, causally shifted.

    Periods are calendar periods in ``tz`` (so "yesterday"/"last week" are
    CT calendar days/weeks, not broker-tz days). Weeks are ISO weeks, which
    start on Monday by definition. Every bar of period N sees period N-1's
    completed extremes (``.shift(1)`` on the period-indexed series); bars in
    the first period of the data are NaN.

    ``freq="D"`` returns ``yesterday_high``/``yesterday_low``;
    ``freq="W"`` returns ``lastweek_high``/``lastweek_low``.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError("index must be timezone-aware (data tz), got naive index")
    idx_tz = idx.tz_convert(tz)
    freq_norm = freq.lower()
    if freq_norm.startswith("d"):
        keys = pd.Series(idx_tz.date, index=idx)
        prefix = prefix or "yesterday"
    elif freq_norm.startswith("w"):
        iso = idx_tz.isocalendar()
        keys = pd.Series([f"{y}-W{w:02d}" for y, w in zip(iso.year, iso.week)], index=idx)
        prefix = prefix or "lastweek"
    else:
        raise ValueError(f"unsupported freq {freq!r} (use 'D' or 'W')")

    period_high = df["high"].groupby(keys.to_numpy()).max().sort_index()
    period_low = df["low"].groupby(keys.to_numpy()).min().sort_index()
    return pd.DataFrame(
        {
            f"{prefix}_high": keys.map(period_high.shift(1)),
            f"{prefix}_low": keys.map(period_low.shift(1)),
        },
        index=idx,
    )


def range_quadrants(high: pd.Series, low: pd.Series, prefix: str) -> pd.DataFrame:
    """Quarter-band boundary levels (nearest equilibrium) of a high/low range.

    Pure utility, agnostic of the range source: works for yesterday's range,
    last week's, or any session range. The four columns are the band
    boundaries a price crosses to enter each quartile of ``span = high - low``:

    - ``{prefix}_q_upper_outer``: ``high - 0.25 * span`` (premium extreme band)
    - ``{prefix}_q_upper_inner``: equilibrium ``low + 0.50 * span``
    - ``{prefix}_q_lower_inner``: equilibrium (duplicate, both names requested)
    - ``{prefix}_q_lower_outer``: ``low + 0.25 * span`` (discount extreme band)
    """
    span = high - low
    eq = low + 0.5 * span
    return pd.DataFrame(
        {
            f"{prefix}_q_upper_outer": high - 0.25 * span,
            f"{prefix}_q_upper_inner": eq,
            f"{prefix}_q_lower_inner": eq,
            f"{prefix}_q_lower_outer": low + 0.25 * span,
        },
        index=high.index,
    )
