"""Tests: causal feature computation (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.indicators import rsi, macd, ema_features, atr, bollinger, returns


def test_rsi_causal(m1_bars):
    """RSI at t must not change when future bars are appended."""
    close = m1_bars["close"]
    rsi_full = rsi(close)
    rsi_trunc = rsi(close.iloc[:250])
    # Values up to bar 249 must match
    pd.testing.assert_series_equal(
        rsi_full.iloc[:249].dropna(),
        rsi_trunc.iloc[:249].dropna(),
        check_names=False,
        rtol=1e-6,
    )


def test_macd_causal(m1_bars):
    close = m1_bars["close"]
    m_full = macd(close)["macd"]
    m_trunc = macd(close.iloc[:300])["macd"]
    pd.testing.assert_series_equal(
        m_full.iloc[:299].dropna(),
        m_trunc.iloc[:299].dropna(),
        check_names=False,
        rtol=1e-6,
    )


def test_ema_causal(m1_bars):
    close = m1_bars["close"]
    feat_full = ema_features(close, [9, 21])
    feat_trunc = ema_features(close.iloc[:200], [9, 21])
    pd.testing.assert_frame_equal(
        feat_full.iloc[:199].dropna(),
        feat_trunc.iloc[:199].dropna(),
        rtol=1e-6,
    )


def test_returns_causal(m1_bars):
    """Log returns must be identical regardless of future data."""
    close = m1_bars["close"]
    r_full = returns(close, [1, 5])
    r_trunc = returns(close.iloc[:100], [1, 5])
    pd.testing.assert_frame_equal(
        r_full.iloc[:99].dropna(),
        r_trunc.iloc[:99].dropna(),
        rtol=1e-9,
    )
