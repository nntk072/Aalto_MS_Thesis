"""Compute chart overlays: EMA50, MACD, RSI, VWAP."""

from __future__ import annotations

from typing import cast

import pandas as pd

from ..features.indicators import _ema


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def compute_ema50(bars: pd.DataFrame) -> pd.Series:
    """Compute EMA(50) on close prices."""
    return _ema(bars["close"], 50)


def compute_macd_sma(
    bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9
) -> pd.DataFrame:
    """Compute MACD with SMA(9) signal line.

    Returns
    -------
    pd.DataFrame with columns: macd, signal, histogram
    """
    close = bars["close"]
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _sma(macd_line, signal_period)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        },
        index=bars.index,
    )


def _with_rsi_vwap(bars: pd.DataFrame, overlays: dict[str, pd.Series]) -> dict[str, pd.Series]:
    from .chart_overlays import compute_vwap_for_chart, rsi_series

    overlays["rsi"] = rsi_series(bars)
    overlays["vwap"] = compute_vwap_for_chart(bars)
    return overlays


def compute_chart_overlays(window: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute EMA50, MACD, RSI and VWAP for a trade window.

    .. warning::
        This computes indicators **from scratch on just the window**, so
        EMA50 (needs ~50 bars) and MACD/signal (needs ~35 bars) are not
        warmed up and will disagree with the strategy's real signals
        computed on full history. Prefer :func:`compute_chart_overlays_full`
        on the full bars, then slice by ``window.index``, for chart
        rendering that must match the actual strategy decisions.

    Parameters
    ----------
    window : pd.DataFrame
        Price bars (OHLC) for the trade window.

    Returns
    -------
    dict with keys: ema50, macd, signal, histogram, rsi, vwap
    """
    ema50 = compute_ema50(window)
    macd_df = compute_macd_sma(window)
    return _with_rsi_vwap(
        window,
        {
            "ema50": ema50,
            "macd": macd_df["macd"],
            "signal": macd_df["signal"],
            "histogram": macd_df["histogram"],
        },
    )


def compute_chart_overlays_full(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute EMA50, MACD, RSI and VWAP over the full bar history.

    Use this once per backtest split, then slice the returned series by
    ``window.index`` for each trade's chart. This ensures the drawn
    indicators are properly warmed up and match the real signals the
    strategy acted on, rather than being recomputed from scratch on a
    short ~60-bar trade window.

    Parameters
    ----------
    bars : pd.DataFrame
        Full M1 price bars (OHLC) for the backtest split.

    Returns
    -------
    dict with keys: ema50, macd, signal, histogram, rsi, vwap
    """
    ema50 = compute_ema50(bars)
    macd_df = compute_macd_sma(bars)
    return _with_rsi_vwap(
        bars,
        {
            "ema50": ema50,
            "macd": macd_df["macd"],
            "signal": macd_df["signal"],
            "histogram": macd_df["histogram"],
        },
    )


def slice_overlays(overlays: dict[str, pd.Series], index: pd.Index) -> dict[str, pd.Series]:
    """Slice a full-history overlays dict down to a trade window's index."""
    return {key: series.reindex(index) for key, series in overlays.items()}


_LINE_KEYS = ("ema50", "macd", "signal", "rsi", "vwap")


def break_overlay_gaps(overlays: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Insert a NaN wherever two plotted candles are more than one bar apart.

    Line series are warmed up on the full M1 spine, then sliced onto the
    window. A gap in that index must not be drawn as a straight slope.
    Histogram bars stay on the candle index.
    """
    out = dict(overlays)
    for key in _LINE_KEYS:
        series = out.get(key)
        if series is not None:
            out[key] = _break_series_gaps(series)
    return out


def _break_series_gaps(series: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(series.index)
    if len(idx) < 2:
        return series
    deltas = idx[1:] - idx[:-1]
    positive = deltas[deltas > pd.Timedelta(0)]
    if len(positive) == 0:
        return series
    step = pd.Timedelta(pd.Series(positive).median())
    if step <= pd.Timedelta(0):
        return series
    pieces: list[pd.Series] = []
    start = 0
    for i, delta in enumerate(deltas, start=1):
        if delta > step:
            pieces.append(series.iloc[start:i])
            mid = pd.Timestamp(idx[i - 1]) + pd.Timedelta(delta) / 2
            pieces.append(
                pd.Series([float("nan")], index=pd.DatetimeIndex([mid]), name=series.name)
            )
            start = i
    pieces.append(series.iloc[start:])
    if len(pieces) == 1:
        return series
    return cast(pd.Series, pd.concat(pieces))
