"""Causal technical indicators computed on bar DataFrames.

All indicators are strictly causal: at index t only rows ≤ t are used.
No pandas-ta / ta-lib dependency – pure pandas/numpy for portability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram},
        index=close.index,
    )


def ema_features(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    """EMA values + slopes (1-bar diff) for each period."""
    out: dict[str, pd.Series] = {}
    for p in periods:
        e = _ema(close, p)
        out[f"ema_{p}"] = e
        out[f"ema_{p}_slope"] = e.diff()
        out[f"ema_{p}_dist"] = (close - e) / e.replace(0, np.nan)
    return pd.DataFrame(out, index=close.index)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean().rename("atr")


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    tr = _true_range(df)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False).mean()
    return pd.DataFrame(
        {"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame(
        {"bb_pct_b": pct_b, "bb_width": width, "bb_mid": mid},
        index=close.index,
    )


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_k = df["low"].rolling(k_period).min()
    high_k = df["high"].rolling(k_period).max()
    stoch_k = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d}, index=df.index)


def vwap_from_session(df: pd.DataFrame) -> pd.Series:
    """Session VWAP using ``session_id`` column."""
    if "session_id" not in df.columns:
        raise ValueError("DataFrame must have 'session_id' column (see data.session)")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["tickvol"].replace(0, np.nan)
    tp_vol = typical * vol
    cum_tp_vol = tp_vol.groupby(df["session_id"]).cumsum()
    cum_vol = vol.groupby(df["session_id"]).cumsum()
    vwap = cum_tp_vol / cum_vol
    vwap.name = "vwap"
    return (df["close"] - vwap) / vwap.replace(0, np.nan)


def returns(close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Log returns at multiple horizons."""
    log_c = np.log(close.replace(0, np.nan))
    out = {}
    for h in horizons:
        out[f"ret_{h}"] = log_c.diff(h)
    return pd.DataFrame(out, index=close.index)


def realized_vol(close: pd.Series, period: int = 20) -> pd.Series:
    lr = np.log(close.replace(0, np.nan)).diff()
    vol = lr.rolling(period).std(ddof=0)
    vol.name = "realized_vol"
    return pd.Series(vol, index=close.index, name="realized_vol")


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical sin/cos encoding of time-of-day and day-of-week."""
    hour_min = index.hour + index.minute / 60.0
    tod_sin = np.sin(2 * np.pi * hour_min / 24.0)
    tod_cos = np.cos(2 * np.pi * hour_min / 24.0)
    dow_sin = np.sin(2 * np.pi * index.dayofweek / 7.0)
    dow_cos = np.cos(2 * np.pi * index.dayofweek / 7.0)
    return pd.DataFrame(
        {"tod_sin": tod_sin, "tod_cos": tod_cos, "dow_sin": dow_sin, "dow_cos": dow_cos},
        index=index,
    )


def atr_normalized_price(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Close distance from previous close, normalised by ATR."""
    atr_s = atr(df, period)
    return df["close"].diff() / atr_s.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------


def build_indicators(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Compute all indicators and return a wide feature DataFrame.

    Parameters
    ----------
    df:
        Cleaned, session-filtered M1 DataFrame with a tz-aware index.
    cfg:
        OmegaConf ``features`` sub-config.
    """
    close = df["close"]
    parts: list[pd.DataFrame | pd.Series] = []

    parts.append(ema_features(close, list(cfg.ema_periods)))
    parts.append(macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal))
    parts.append(rsi(close, cfg.rsi_period).rename("rsi"))
    parts.append(atr(df, cfg.atr_period))
    parts.append(adx(df, cfg.adx_period))
    parts.append(bollinger(close, cfg.bb_period, cfg.bb_std))
    parts.append(stochastic(df, cfg.stoch_k, cfg.stoch_d))
    if "session_id" in df.columns:
        parts.append(vwap_from_session(df))
    parts.append(returns(close, list(cfg.return_horizons)))
    parts.append(realized_vol(close, cfg.realized_vol_period))
    parts.append(time_features(pd.DatetimeIndex(df.index)))
    parts.append(atr_normalized_price(df, cfg.atr_period).rename("atr_norm_price"))

    feat = pd.concat([p for p in parts], axis=1)
    return feat
