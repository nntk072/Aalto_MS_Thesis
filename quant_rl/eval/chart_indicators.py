"""Compute chart overlays: EMA50, MACD, signal, histogram."""
from __future__ import annotations

import pandas as pd
from ..features.indicators import _ema, macd as _macd


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def compute_ema50(bars: pd.DataFrame) -> pd.Series:
    """Compute EMA(50) on close prices."""
    return _ema(bars["close"], 50)


def compute_macd_sma(bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
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


def compute_chart_overlays(window: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute EMA50 and MACD for a trade window.
    
    Parameters
    ----------
    window : pd.DataFrame
        Price bars (OHLC) for the trade window.
    
    Returns
    -------
    dict with keys: ema50, macd, signal, histogram (all aligned to window.index)
    """
    ema50 = compute_ema50(window)
    macd_df = compute_macd_sma(window)
    
    return {
        "ema50": ema50,
        "macd": macd_df["macd"],
        "signal": macd_df["signal"],
        "histogram": macd_df["histogram"],
    }
