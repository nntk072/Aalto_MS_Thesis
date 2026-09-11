"""Broker-tz session labels, NY eligibility mask, and session-day IDs.

Sessions are broker-time definitions in the DataFrame timezone (default
``Etc/GMT-3``), not geographical exchange clocks. CT-anchored levels live in
``quant_rl.features.session_ohlc`` and stay additive (``*_ct`` columns).

Tradable NY is ``[start, end]`` (default 16:30–23:00). Labels:

- asia:   01:05 <= t < 09:00
- london: 09:00 <= t < 16:30
- ny:     16:30 <= t <= 23:00  (matches ``cfg.session``)
- closed: otherwise (broker gap, e.g. 23:01–01:04)

``filter_session`` is a mask helper only. Never use it to drop bars from the
feature-construction dataset. D1 bars are broker-midnight (``resample`` ``1D``,
left-labeled) in this timezone.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

SessionName = Literal["asia", "london", "ny", "closed"]

_ASIA_START = "01:05"
_ASIA_END = "09:00"
_LONDON_END = "16:30"
_NY_END_DEFAULT = "23:00"


def _hhmm(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"2000-01-01 {value}")


def get_session(
    timestamp: pd.Timestamp | str,
    tz: str = "Etc/GMT-3",
    ny_end: str = _NY_END_DEFAULT,
) -> SessionName:
    """Return the broker-tz session label for a timestamp.

    Args:
        timestamp: Timestamp to classify (naive timestamps are localized to ``tz``).
        tz: Broker timezone used when ``timestamp`` is naive.
        ny_end: Inclusive NY session end as HH:MM (default 23:00).
    """
    if isinstance(timestamp, str):
        ts = pd.Timestamp(timestamp)
    else:
        ts = timestamp
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    t = ts.time()
    asia_start = _hhmm(_ASIA_START).time()
    asia_end = _hhmm(_ASIA_END).time()
    london_end = _hhmm(_LONDON_END).time()
    ny_end_t = _hhmm(ny_end).time()
    if asia_start <= t < asia_end:
        return "asia"
    if asia_end <= t < london_end:
        return "london"
    if london_end <= t <= ny_end_t:
        return "ny"
    return "closed"


def ny_session_mask(
    index: pd.DatetimeIndex,
    start: str = "16:30",
    end: str = "23:00",
) -> pd.Series:
    """Boolean mask of bars whose clock time falls in the tradable NY window."""
    t = index.time
    start_t = _hhmm(start).time()
    end_t = _hhmm(end).time()
    return pd.Series((t >= start_t) & (t <= end_t), index=index)


def add_session_labels(
    df: pd.DataFrame,
    tz: str = "Etc/GMT-3",
    ny_end: str = _NY_END_DEFAULT,
) -> pd.DataFrame:
    """Add a ``session`` column (asia/london/ny/closed) using :func:`get_session`."""
    df = df.copy()
    index = cast(pd.DatetimeIndex, df.index)
    if index.tz is None:
        local = index.tz_localize(tz)
    else:
        local = index.tz_convert(tz)
    t = local.time
    asia_start = _hhmm(_ASIA_START).time()
    asia_end = _hhmm(_ASIA_END).time()
    london_end = _hhmm(_LONDON_END).time()
    ny_end_t = _hhmm(ny_end).time()
    labels = np.empty(len(index), dtype=object)
    labels[:] = "closed"
    labels[(t >= asia_start) & (t < asia_end)] = "asia"
    labels[(t >= asia_end) & (t < london_end)] = "london"
    labels[(t >= london_end) & (t <= ny_end_t)] = "ny"
    df["session"] = labels
    return df


def filter_session(
    df: pd.DataFrame,
    start: str = "16:30",
    end: str = "23:00",
) -> pd.DataFrame:
    """Keep only bars inside the NY tradable window.

    This is an eligibility-mask helper. Do not apply it to the feature dataset.
    """
    df = df.copy()
    index = cast(pd.DatetimeIndex, df.index)
    mask = ny_session_mask(index, start=start, end=end)
    return df.loc[mask]


def add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``session_id`` integer column: unique per calendar date."""
    df = df.copy()
    dates = cast(pd.DatetimeIndex, df.index).normalize()
    unique_dates = sorted(set(dates))
    date_to_id = {d: i for i, d in enumerate(unique_dates)}
    df["session_id"] = dates.map(date_to_id)
    return df
