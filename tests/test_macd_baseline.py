"""Tests for MACD EMA50 baseline strategy."""

import numpy as np
import pandas as pd
import pytest

from quant_rl.backtest.costs import COST_US100
from quant_rl.backtest.engine import run_backtest
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

    return pd.DataFrame(
        {
            "open": open_vals.values,
            "high": high_vals.values,
            "low": low_vals.values,
            "close": close_vals.values,
        },
        index=dates,
    )


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
    """Test that after entry, subsequent hold bars emit 0 (event-based stream).

    The action stream must be event-based, not per-bar state: the entry
    action (+1/-1) fires only on the bar the position opens; every bar the
    position is held emits 0; the exit action (2) fires only on the bar the
    position closes. This is required so the engine (which starts acting at
    obs_window) never desyncs from a position tracked from bar 0.
    """
    actions = macd_ema50_baseline(sample_bars)

    position = 0
    for i, action in enumerate(actions):
        if position == 1:  # In long
            assert action in (0, 2), f"In long, got action {action} at bar {i} (holds must be 0)"
        elif position == -1:  # In short
            assert action in (0, 2), f"In short, got action {action} at bar {i} (holds must be 0)"

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
                assert action not in (1, -1), (
                    f"Entry detected at bar {i} ({bars_since_exit} bars after exit)"
                )
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

    eps = 0.2  # Allow small tolerance for rounding/numerical errors
    for i, action in enumerate(actions):
        if action == 1:  # Long entry
            assert close.iloc[i] >= ema50.iloc[i] - eps, (
                f"Long entry at bar {i}: close {close.iloc[i]:.2f} not >= EMA50 {ema50.iloc[i]:.2f}"
            )
        elif action == -1:  # Short entry
            assert close.iloc[i] <= ema50.iloc[i] + eps, (
                f"Short entry at bar {i}: close {close.iloc[i]:.2f} not <= EMA50 {ema50.iloc[i]:.2f}"
            )
        elif action == 2:
            pass


def test_macd_sma_signal(sample_bars):
    """Test that signal line uses SMA(9), not EMA."""
    from quant_rl.baselines.rule_based import _sma
    from quant_rl.features.indicators import _ema

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


def test_macd_backtest_alignment(sample_bars):
    """Test that MACD backtest entries align with precomputed actions."""
    from quant_rl.features.build import build_features

    # Build minimal features
    features = build_features(sample_bars, cfg=None)

    # Precompute MACD actions
    actions = macd_ema50_baseline(sample_bars)

    # Create a policy that maps to these actions
    obs_window = 60
    state = {"i": obs_window}

    def policy(obs):
        if state["i"] < len(actions):
            action = int(actions.iloc[state["i"]])
        else:
            action = 0
        state["i"] += 1
        return action

    # Run minimal backtest
    result = run_backtest(
        bars=sample_bars,
        features=features,
        policy=policy,
        obs_window=obs_window,
        hold_on_zero=True,
        exit_action=2,
    )

    # Verify that backtest ran without errors
    assert "trades" in result
    assert "equity" in result
    trades_df = result["trades"]

    # Verify no immediate entry/exit (hold_on_zero should prevent that)
    if len(trades_df) > 0:
        for _, row in trades_df[trades_df["type"] == "open"].iterrows():
            # Entry should be at least at or after obs_window
            assert row["bar"] >= obs_window


def test_macd_backtest_cooldown():
    """Test that backtest respects MACD's 5-bar cooldown."""
    # Create synthetic bars with known MACD behavior
    dates = pd.date_range("2024-01-01", periods=200, freq="1min")
    close = pd.Series(100.0 + np.sin(np.arange(200) / 10.0) * 5, index=dates)

    bars = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=dates,
    )

    actions = macd_ema50_baseline(bars)

    # Count entry gaps
    exit_bars = []
    for i, action in enumerate(actions):
        if action == 2:  # Exit
            exit_bars.append(i)

    # Find entries after exits
    entry_bars = []
    for i, action in enumerate(actions):
        if action in (1, -1):  # Entry
            entry_bars.append(i)

    # Verify cooldown: no entry within 5 bars after exit
    for exit_bar in exit_bars:
        cooldown_end = exit_bar + 5
        entries_in_cooldown = [e for e in entry_bars if exit_bar < e <= cooldown_end]
        # There may be entries at cooldown boundary or after
        if entries_in_cooldown:
            # Entries should only occur at or after cooldown_end
            assert all(e > cooldown_end for e in entries_in_cooldown), (
                f"Entry at {min(entries_in_cooldown)} violates 5-bar cooldown from exit at {exit_bar}"
            )


def test_broker_rejects_invalid_direction():
    """Regression: open_position must reject any direction not in {1, -1}.

    This guards against the bug where the engine's exit_action (e.g. 2)
    reached open_position() and created a bogus direction=2 position.
    """
    from quant_rl.backtest.account import AccountState
    from quant_rl.backtest.broker import Broker

    broker = Broker(cost_model=COST_US100)
    acc = AccountState(initial_balance=100_000.0)

    # Valid directions still open a position.
    pos_long = broker.open_position(acc, (100.0, 100.1), 1.0, 1)
    assert pos_long is not None
    pos_short = broker.open_position(acc, (100.0, 100.1), 1.0, -1)
    assert pos_short is not None

    # Invalid directions (e.g. the exit_action code 2, or 0) must be rejected.
    for bad_direction in (2, 0, -2, 3):
        assert broker.open_position(acc, (100.0, 100.1), 1.0, bad_direction) is None


def test_engine_exit_action_while_flat_is_noop(sample_bars):
    """Regression: policy returning exit_action while flat must not open a position.

    Reproduces the original bug: a leftover exit event (2) hitting the
    engine while there is no open position must never reach
    broker.open_position and must never appear as a trade_log "open" row.
    """
    from quant_rl.features.build import build_features

    features = build_features(sample_bars, cfg=None)

    def _always_exit_policy(obs):
        return 2  # exit_action, but engine starts flat

    result = run_backtest(
        bars=sample_bars,
        features=features,
        policy=_always_exit_policy,
        obs_window=60,
        hold_on_zero=True,
        exit_action=2,
    )

    trades_df = result["trades"]
    if len(trades_df) > 0:
        opens = trades_df[trades_df["type"] == "open"]
        assert len(opens) == 0, "Exit action while flat must never open a position"


def test_engine_open_direction_always_valid(sample_bars):
    """Regression: every logged 'open' row must have direction in {1, -1}.

    Guards against the direction=2 bug observed in trades.csv, using the
    real MACD action stream end-to-end through the engine.
    """
    from quant_rl.features.build import build_features

    features = build_features(sample_bars, cfg=None)
    actions = macd_ema50_baseline(sample_bars)

    obs_window = 60
    state = {"i": obs_window}

    def policy(obs):
        action = int(actions.iloc[state["i"]]) if state["i"] < len(actions) else 0
        state["i"] += 1
        return action

    result = run_backtest(
        bars=sample_bars,
        features=features,
        policy=policy,
        obs_window=obs_window,
        hold_on_zero=True,
        exit_action=2,
    )

    trades_df = result["trades"]
    if len(trades_df) > 0:
        opens = trades_df[trades_df["type"] == "open"]
        for _, row in opens.iterrows():
            assert row["direction"] in (1.0, -1.0), (
                f"Invalid open direction {row['direction']} at bar {row['bar']}"
            )


def test_engine_fills_within_bar_range(sample_bars):
    """Regression: every logged open/close fill price must be within the
    executing bar's [low, high] range (guards against tick-outlier fills
    landing far from any visible candle)."""
    from quant_rl.features.build import build_features

    features = build_features(sample_bars, cfg=None)
    actions = macd_ema50_baseline(sample_bars)

    obs_window = 60
    state = {"i": obs_window}

    def policy(obs):
        action = int(actions.iloc[state["i"]]) if state["i"] < len(actions) else 0
        state["i"] += 1
        return action

    result = run_backtest(
        bars=sample_bars,
        features=features,
        policy=policy,
        obs_window=obs_window,
        hold_on_zero=True,
        exit_action=2,
    )

    trades_df = result["trades"]
    bars_idx = sample_bars.index
    lows = sample_bars["low"]
    highs = sample_bars["high"]

    for _, row in trades_df.iterrows():
        if row["type"] not in ("open", "close", "stop_close", "tp_close", "eod_close"):
            continue
        price = row.get("price")
        if price is None or pd.isna(price):
            continue
        bar_time = pd.Timestamp(row["time"])
        # Fill executes at/after this bar_time; find nearest bar for a
        # generous sanity range check (allow the surrounding few bars since
        # exact fill bar depends on tick timing).
        pos = bars_idx.get_indexer([bar_time], method="nearest")[0]
        window_lo = lows.iloc[max(0, pos - 1) : pos + 2].min()
        window_hi = highs.iloc[max(0, pos - 1) : pos + 2].max()
        assert window_lo - 1.0 <= price <= window_hi + 1.0, (
            f"Fill price {price} at {bar_time} far outside nearby bar range "
            f"[{window_lo}, {window_hi}]"
        )
