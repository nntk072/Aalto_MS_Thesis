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

import numpy as np
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


def prev_ny_high_low(
    df: pd.DataFrame,
    start_ct: str = "08:30",
    end_ct: str = "15:00",
    session_tz: str = "America/Chicago",
) -> pd.DataFrame:
    """Latest clock-completed NY session high/low, available without lookahead.

    Bars during an open session see the prior session's observed high/low. At
    the first decision bar at or after the session close, that session becomes
    the reference. Session extrema use available rows; missing-bar quality is
    audited separately and does not silently disqualify a session here. Bars
    before the first session close are NaN.
    """
    idx = pd.DatetimeIndex(df.index)
    mask = session_mask(idx, start_ct, end_ct, session_tz)
    idx_ct = idx.tz_convert(session_tz)
    start_t = _parse_hhmm(start_ct)
    end_t = _parse_hhmm(end_ct)
    mask_values = mask.to_numpy(dtype=bool)
    session_labels = pd.Series(
        _session_dates(idx_ct, start_t, end_t).where(mask).to_numpy(), index=idx
    )
    session_high = df["high"].where(mask_values).groupby(session_labels, sort=True).max()
    session_low = df["low"].where(mask_values).groupby(session_labels, sort=True).min()
    if session_high.empty:
        return pd.DataFrame({"prev_ny_high": np.nan, "prev_ny_low": np.nan}, index=idx)

    session_ends: list[pd.Timestamp] = []
    for session_date in session_high.index:
        end_date = session_date
        if start_t > end_t:
            end_date = end_date + dt.timedelta(days=1)
        end_wall_time = pd.Timestamp.combine(end_date, end_t)
        session_ends.append(end_wall_time.tz_localize(session_tz))

    completed_position = pd.DatetimeIndex(session_ends).searchsorted(idx_ct, side="right") - 1
    valid = completed_position >= 0
    high_values = np.full(len(idx), np.nan, dtype=np.float64)
    low_values = np.full(len(idx), np.nan, dtype=np.float64)
    high_values[valid] = session_high.to_numpy(dtype=np.float64)[completed_position[valid]]
    low_values[valid] = session_low.to_numpy(dtype=np.float64)[completed_position[valid]]
    return pd.DataFrame({"prev_ny_high": high_values, "prev_ny_low": low_values}, index=idx)


def previous_labeled_session_ohlc(
    df: pd.DataFrame,
    sessions: dict[str, dict[str, str]],
    session_tz: str = "America/Chicago",
    *,
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Expose the latest fully completed configured session's range.

    Session labels are considered in timestamp order, so the selected range
    may be Asia, London, or NY depending on which configured window most
    recently ended. A session is well formed only when its final expected
    bar is present; a truncated tail is never promoted to a reference. The
    range becomes visible at the first decision bar whose timestamp is at or
    after the session close (the close of the final in-session bar).

    ``sessions`` maps stable labels to ``{"start": "HH:MM", "end": "HH:MM"}``
    wall-clock windows. The calendar is evaluated in ``session_tz`` for each
    bar, including DST transitions. Codes are assigned by sorted label name
    starting at 1; zero means no completed session is available.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError("index must be timezone-aware (data tz), got naive index")
    if bar_minutes < 1:
        raise ValueError("bar_minutes must be positive")
    labels = sorted(sessions)
    code_for = {label: code for code, label in enumerate(labels, start=1)}
    local = idx.tz_convert(session_tz)
    chosen_high = np.full(len(df), np.nan, dtype=float)
    chosen_low = np.full(len(df), np.nan, dtype=float)
    chosen_code = np.zeros(len(df), dtype=float)
    chosen_close_ns = np.full(len(df), np.nan, dtype=float)

    # Gather only complete sessions, then apply each record from its close
    # onward. Stable sorting by completion time makes the latest one win.
    completed: list[tuple[pd.Timestamp, float, float, int]] = []
    high_values = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low_values = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    for label in labels:
        window = sessions[label]
        start_t = _parse_hhmm(window["start"])
        end_t = _parse_hhmm(window["end"])
        mask = session_mask(idx, window["start"], window["end"], session_tz).to_numpy()
        dates = _session_dates(local, start_t, end_t).to_numpy()
        group_keys = [(dates[i], label) if mask[i] else None for i in range(len(idx))]
        positions: dict[tuple[dt.date, str], list[int]] = {}
        for i, key in enumerate(group_keys):
            if key is not None:
                positions.setdefault(key, []).append(i)
        for (session_date, _), rows in positions.items():
            start_local = pd.Timestamp.combine(session_date, start_t).tz_localize(
                session_tz, nonexistent="shift_forward", ambiguous="raise"
            )
            close_local = pd.Timestamp.combine(session_date, end_t).tz_localize(session_tz)
            expected_last = close_local - pd.Timedelta(minutes=bar_minutes)
            last_bar = idx[rows[-1]]
            expected_count = int(
                (close_local.tz_convert("UTC") - start_local.tz_convert("UTC")).total_seconds()
                // (bar_minutes * 60)
            )
            actual_times = (
                idx[rows].tz_convert("UTC").to_numpy(dtype="datetime64[ns]").view(np.int64)
            )
            contiguous = len(rows) == expected_count and (
                len(rows) <= 1
                or bool((np.diff(actual_times) == pd.Timedelta(minutes=bar_minutes).value).all())
            )
            if not contiguous or last_bar.tz_convert(session_tz) != expected_last:
                continue
            high = float(np.nanmax(high_values[rows]))
            low = float(np.nanmin(low_values[rows]))
            if not np.isfinite(high) or not np.isfinite(low):
                continue
            completed.append((close_local.tz_convert(idx.tz), high, low, code_for[label]))

    completed.sort(key=lambda item: item[0])
    close_ns = np.asarray([item[0].value for item in completed], dtype=np.int64)
    for i, timestamp in enumerate(idx):
        j = int(np.searchsorted(close_ns, timestamp.value, side="right") - 1)
        if j >= 0:
            close_at, high, low, code = completed[j]
            chosen_high[i] = high
            chosen_low[i] = low
            chosen_code[i] = code
            chosen_close_ns[i] = close_at.value

    return pd.DataFrame(
        {
            "previous_session_high": chosen_high,
            "previous_session_low": chosen_low,
            "previous_session_midpoint": (chosen_high + chosen_low) / 2.0,
            "previous_session_code": chosen_code,
            "previous_session_completed_at_ns": chosen_close_ns,
        },
        index=idx,
    )
