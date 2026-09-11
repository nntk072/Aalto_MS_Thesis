"""Shared equity/drawdown series helpers for matplotlib and Plotly charts."""

from __future__ import annotations

from typing import Any

import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def _date_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Calendar-day key in the series timezone (session-start grouping)."""
    return index.tz_convert(index.tz).normalize() if index.tz is not None else index.normalize()


def session_start_equity(equity: pd.Series) -> pd.Series:
    """Equity at the first bar of each calendar day, forward-filled intra-day."""
    if equity.empty:
        return equity.copy()
    idx = pd.DatetimeIndex(equity.index)
    return equity.groupby(_date_key(idx)).transform("first")


def daily_loss_limit_series(equity: pd.Series, daily_loss_limit: float) -> pd.Series:
    """FTMO daily stop: start-of-day equity minus the dollar daily-loss cap.

    A day that opens at 104_000 with a $5_000 cap is flat at 99_000 all day,
    even if equity later prints 105_000.
    """
    return session_start_equity(equity) - float(daily_loss_limit)


def max_drawdown_pct(equity: pd.Series) -> pd.Series:
    """Peak-to-trough drawdown in percent (0 at highs, negative underwater)."""
    peak = equity.cummax().replace(0, np.nan)
    return (equity - peak) / peak * 100.0


def daily_drawdown_pct(equity: pd.Series) -> pd.Series:
    """Intraday drawdown from that day's opening equity, in percent."""
    start = session_start_equity(equity).replace(0, np.nan)
    return (equity - start) / start * 100.0


def daily_pnl(equity: pd.Series) -> pd.Series:
    """Closed-bar session P&L: last equity of the day minus first."""
    if equity.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(equity.index)
    grouped = equity.groupby(_date_key(idx))
    return grouped.last() - grouped.first()


def drawdown_ylim(values: pd.Series, floor: float) -> tuple[float, float]:
    """Y-limits from ``floor`` to 0, expanding only if the series is worse."""
    observed = float(np.nanmin(np.asarray(values, dtype=float))) if len(values) else 0.0
    if np.isnan(observed):
        observed = 0.0
    return (min(float(floor), observed), 0.0)


def apply_mpl_date_axis(ax: Any, tz: Any | None = None) -> None:
    """Tick dates as month/day (not a lone year) on an equity-style axis."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10, tz=tz)
    formatter = mdates.ConciseDateFormatter(locator, tz=tz)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
