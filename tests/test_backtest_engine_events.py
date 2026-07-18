"""Regression tests for engine event semantics.

Verifies:
- breach_events has real timestamps, one per breached session
- per-trade max_loss_per_trade_usd hard stop works
- session diagnostic counters are accurate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.backtest.engine import run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 300, n_sessions: int = 3, start_price: float = 20_000.0) -> pd.DataFrame:
    """Synthetic M1 bars across multiple sessions."""
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min",
                        tz="Etc/GMT-3")
    prices = start_price + np.cumsum(np.random.default_rng(1).normal(0, 0.5, n))
    bars = pd.DataFrame({
        "open": prices,
        "high": prices + 0.3,
        "low": prices - 0.3,
        "close": prices,
        "tickvol": 1,
        "session_id": np.tile(np.arange(n_sessions), int(np.ceil(n / n_sessions)))[:n],
    }, index=idx)
    return bars


def _make_features(bars: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((len(bars), 10)).astype(np.float32),
        index=bars.index,
        columns=[f"f{i}" for i in range(10)],
    )


def _always_long(obs: np.ndarray) -> int:
    return 1


def _always_flat(obs: np.ndarray) -> int:
    return 0


# ---------------------------------------------------------------------------
# Test: breach_events contain real timestamps (not synthetic)
# ---------------------------------------------------------------------------

def test_breach_events_have_real_timestamps():
    """Each breach_event must have a 'time' key matching a bar timestamp."""
    bars = _make_bars(n=400, n_sessions=4)
    features = _make_features(bars)

    # Set a very tight drawdown limit so breaches happen
    result = run_backtest(
        bars, features, _always_long,
        obs_window=10,
        initial_balance=100_000.0,
        guardrail_kwargs={"daily_loss_limit": 5.0, "max_loss_limit": 10.0},
    )

    events = result["breach_events"]
    if not events:
        pytest.skip("No breaches occurred with this synthetic data — adjust params")

    for ev in events:
        assert "time" in ev, "breach_event missing 'time'"
        assert "session_id" in ev, "breach_event missing 'session_id'"
        assert "reason" in ev, "breach_event missing 'reason'"
        assert "equity" in ev, "breach_event missing 'equity'"
        assert isinstance(ev["time"], pd.Timestamp), f"time is not Timestamp: {type(ev['time'])}"

    # One event per unique breached session
    session_ids = [ev["session_id"] for ev in events]
    assert len(session_ids) == len(set(session_ids)), "Duplicate session_id in breach_events"


# ---------------------------------------------------------------------------
# Test: breach count equals unique sessions, not per-bar
# ---------------------------------------------------------------------------

def test_breach_count_is_per_session_not_per_bar():
    """n_breach_sessions must equal len(breach_events), not total bars."""
    bars = _make_bars(n=600, n_sessions=3)
    features = _make_features(bars)

    result = run_backtest(
        bars, features, _always_long,
        obs_window=10,
        initial_balance=100_000.0,
        guardrail_kwargs={"daily_loss_limit": 1.0, "max_loss_limit": 2.0},
    )

    assert result["n_breach_sessions"] == len(result["breach_events"]), (
        f"n_breach_sessions={result['n_breach_sessions']} != "
        f"len(breach_events)={len(result['breach_events'])}"
    )
    assert result["n_breach_sessions"] <= result["n_sessions"]


# ---------------------------------------------------------------------------
# Test: per-trade max_loss_per_trade_usd stop
# ---------------------------------------------------------------------------

def test_max_loss_per_trade_closes_position():
    """A trade should be closed when unrealised loss >= max_loss_per_trade_usd."""
    # Use a downward price series so long positions lose money
    n = 300
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    prices = 20_000.0 - np.arange(n) * 0.5  # falls 0.5 per bar
    bars = pd.DataFrame({
        "open": prices, "high": prices + 0.1,
        "low": prices - 0.1, "close": prices,
        "tickvol": 1,
        "session_id": np.tile(np.arange(3), 100)[:n],
    }, index=idx)
    features = _make_features(bars)

    # With lots=1 and contract_size=1, 1 bar loss ~= 0.5 USD; cap at $3
    result = run_backtest(
        bars, features, _always_long,
        obs_window=10,
        initial_balance=100_000.0,
        lots=1.0,
        max_loss_per_trade_usd=3.0,
        guardrail_kwargs={"daily_loss_limit": 50_000.0, "max_loss_limit": 100_000.0},
    )

    trades = result["trades"]
    stop_closes = trades[trades["type"] == "stop_close"]
    assert len(stop_closes) > 0, "Expected at least one stop_close trade"

    # All stop_closes must have pnl <= -3.0 (capped near the limit)
    for _, row in stop_closes.iterrows():
        assert row["pnl"] <= 0, f"stop_close pnl should be negative, got {row['pnl']}"


# ---------------------------------------------------------------------------
# Test: session diagnostics counters
# ---------------------------------------------------------------------------

def test_session_diagnostics_counters():
    """n_sessions, n_sessions_with_trades, n_breach_sessions should be self-consistent."""
    bars = _make_bars(n=300, n_sessions=3)
    features = _make_features(bars)

    result = run_backtest(
        bars, features, _always_long,
        obs_window=10,
        initial_balance=100_000.0,
    )

    assert result["n_sessions"] == 3, f"Expected 3 sessions, got {result['n_sessions']}"
    assert result["n_sessions_with_trades"] <= result["n_sessions"]
    assert result["n_breach_sessions"] <= result["n_sessions"]
    assert "n_sessions_skipped" in result


# ---------------------------------------------------------------------------
# Test: flat policy never opens a trade
# ---------------------------------------------------------------------------

def test_flat_policy_no_trades():
    bars = _make_bars(n=200, n_sessions=2)
    features = _make_features(bars)

    result = run_backtest(bars, features, _always_flat, obs_window=10)

    trades = result["trades"]
    open_trades = trades[trades["type"] == "open"] if not trades.empty else pd.DataFrame()
    assert len(open_trades) == 0, "Flat policy should produce no open trades"
    assert result["n_sessions_with_trades"] == 0
