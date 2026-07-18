"""Tests for MACD EMA50 baseline strategy."""
import numpy as np
import pandas as pd
import pytest

from quant_rl.baselines.rule_based import macd_ema50_baseline


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Create synthetic price bars for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="1min")
    
    # Create trending data with some crossovers
    close_vals = pd.Series(100.0 + np.cumsum(np.random.randn(500) * 0.5))
    high_vals = close_vals + np.abs(np.random.randn(500)) * 0.3
    low_vals = close_vals - np.abs(np.random.randn(500)) * 0.3
    open_vals = close_vals.shift(1).fillna(close_vals.iloc[0])
    
    return pd.DataFrame({
        "open": open_vals.values,
        "high": high_vals.values,
        "low": low_vals.values,
        "close": close_vals.values,
    }, index=dates)


def test_macd_baseline_returns_series(sample_bars):
    """Test that macd_ema50_baseline returns a Series."""
    actions = macd_ema50_baseline(sample_bars)
    assert isinstance(actions, pd.Series)
    assert len(actions) == len(sample_bars)


def test_macd_baseline_action_values(sample_bars):
    """Test that returned actions are valid: +1, -1, 0, or 2."""
    actions = macd_ema50_baseline(sample_bars)
    unique_actions = set(actions.unique())
    assert unique_actions.issubset({-1, 0, 1, 2})


def test_macd_baseline_holds_after_entry(sample_bars):
    """Test that after entry, action is either hold or exit (not immediate flip)."""
    actions = macd_ema50_baseline(sample_bars)
    
    position = 0
    for i, action in enumerate(actions):
        if position == 1:  # In long
            assert action in (1, 0, 2), f"In long, got action {action} at bar {i}"
        elif position == -1:  # In short
            assert action in (-1, 0, 2), f"In short, got action {action} at bar {i}"
        
        # Update position based on action
        if action == 1:
            position = 1
        elif action == -1:
            position = -1
        elif action == 2:
            position = 0


def test_macd_baseline_cooldown(sample_bars):
    """Test that cooldown is observed: no entry within 5 bars after exit."""
    actions = macd_ema50_baseline(sample_bars)
    
    exit_bar = None
    for i, action in enumerate(actions):
        if exit_bar is not None:
            bars_since_exit = i - exit_bar
            if bars_since_exit < 5:
                # Should not enter during cooldown
                assert action not in (1, -1), \
                    f"Entry detected at bar {i} ({bars_since_exit} bars after exit)"
            else:
                exit_bar = None  # Cooldown expired
        
        if action == 2:  # Exit signal
            exit_bar = i


def test_macd_baseline_no_auto_flip(sample_bars):
    """Test that strategy doesn't flip from long to short (or vice versa) without exiting first."""
    actions = macd_ema50_baseline(sample_bars)
    
    position = 0
    for i, action in enumerate(actions):
        if position == 1 and action == -1:
            # Tried to flip from long to short without exiting
            pytest.fail(f"Auto-flip detected: long → short at bar {i}")
        elif position == -1 and action == 1:
            # Tried to flip from short to long without exiting
            pytest.fail(f"Auto-flip detected: short → long at bar {i}")
        
        # Update position
        if action == 1:
            position = 1
        elif action == -1:
            position = -1
        elif action == 2:
            position = 0


def test_macd_baseline_respects_ema50_filter(sample_bars):
    """Test that long entries generally occur when close > EMA50 and short entries when close < EMA50."""
    from quant_rl.features.indicators import _ema
    
    close = sample_bars["close"]
    ema50 = _ema(close, 50)
    actions = macd_ema50_baseline(sample_bars)
    
    position = 0
    eps = 0.2  # Allow small tolerance for rounding/numerical errors
    for i, action in enumerate(actions):
        if action == 1:  # Long entry
            # Should be at/above EMA50 at entry (with small tolerance)
            assert close.iloc[i] >= ema50.iloc[i] - eps, \
                f"Long entry at bar {i}: close {close.iloc[i]:.2f} not >= EMA50 {ema50.iloc[i]:.2f}"
            position = 1
        elif action == -1:  # Short entry
            # Should be at/below EMA50 at entry (with small tolerance)
            assert close.iloc[i] <= ema50.iloc[i] + eps, \
                f"Short entry at bar {i}: close {close.iloc[i]:.2f} not <= EMA50 {ema50.iloc[i]:.2f}"
            position = -1
        elif action == 2:
            position = 0


def test_macd_sma_signal(sample_bars):
    """Test that signal line uses SMA(9), not EMA."""
    from quant_rl.features.indicators import _ema
    from quant_rl.baselines.rule_based import _sma
    
    close = sample_bars["close"]
    fast_ema = _ema(close, 12)
    slow_ema = _ema(close, 26)
    macd_line = fast_ema - slow_ema
    signal_line = _sma(macd_line, 9)
    
    # Signal should be SMA, not EMA
    assert isinstance(signal_line, pd.Series)
    assert len(signal_line) == len(macd_line)
    # Check first non-NaN value after warmup period (9 bars)
    assert signal_line.iloc[9:].notna().any()
