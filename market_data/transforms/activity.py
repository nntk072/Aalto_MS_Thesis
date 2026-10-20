"""Canonical bar activity for CFD feeds (tick count, not exchange volume).

MT5 ``<VOL>`` on US100.cash is always 0. ``<TICKVOL>`` is the tick count in
the bar and is the only full-history activity measure. A zero ``volume`` /
``vol`` column must not win over ``tickvol`` (that path killed VWAP and
``volume_spike``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ZERO_FRAC = 0.99
_CANDIDATES = ("tick_count", "tickvol", "volume", "vol")


def _is_dead_volume(series: pd.Series) -> bool:
    """True when a volume-like column is missing, empty, or ~all zeros."""
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return True
    filled = vals.fillna(0.0)
    return bool((filled == 0.0).mean() >= _ZERO_FRAC)


def activity_series(df: pd.DataFrame) -> pd.Series | None:
    """Return tick-count activity, ignoring all-zero ``vol`` / ``volume``.

    Prefers ``tick_count`` (optional tape overlay) then ``tickvol``. CFD
    ``volume`` / ``vol`` are used only when they are not ~all zeros.
    """
    for col in _CANDIDATES:
        if col not in df.columns:
            continue
        raw = df[col]
        if col in ("volume", "vol") and _is_dead_volume(raw):
            continue
        s = pd.to_numeric(raw, errors="coerce").astype(float)
        if _is_dead_volume(s):
            continue
        return s.rename("activity")
    return None


def attach_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Copy *df* and set ``activity`` from :func:`activity_series` when possible."""
    out = df.copy()
    act = activity_series(out)
    if act is not None:
        out["activity"] = act
    return out


def tick_counts_from_index(
    bars_index: pd.DatetimeIndex,
    tick_index: pd.DatetimeIndex,
) -> pd.Series:
    """Count ticks whose timestamp falls in each M1 bar (left-labeled)."""
    if bars_index.empty:
        return pd.Series(dtype=float, name="tick_count")
    bar_ns = np.asarray(bars_index, dtype="datetime64[ns]").view(np.int64)
    tick_ns = np.asarray(tick_index, dtype="datetime64[ns]").view(np.int64)
    loc = np.searchsorted(bar_ns, tick_ns, side="right") - 1
    valid = loc >= 0
    counts = np.bincount(loc[valid], minlength=len(bars_index))
    return pd.Series(counts.astype(float), index=bars_index, name="tick_count")


def load_tick_index(path: Path | str, tz: str = "Etc/GMT-3") -> pd.DatetimeIndex:
    """DatetimeIndex of ticks from an MT5 tick CSV (date+time only)."""
    raw = pd.read_csv(
        path,
        sep="\t",
        header=0,
        usecols=[0, 1],
        names=["date", "time"],
        dtype=str,
    )
    ts = pd.to_datetime(raw["date"] + " " + raw["time"], format="%Y.%m.%d %H:%M:%S.%f")
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        idx = idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    return idx
