"""Tests: NY session filter correctness."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_rl.data.session import add_session_id, filter_session


@pytest.fixture
def sample_bars():
    idx = pd.date_range("2025-01-06 00:00", periods=24 * 60, freq="1min", tz="Etc/GMT-3")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "tickvol": 1, "vol": 0, "spread": 0.6},
        index=idx,
    )
    return df


def test_session_filter_keeps_only_window(sample_bars):
    filtered = filter_session(sample_bars, start="16:30", end="23:00")
    t = filtered.index.time
    start_t = pd.Timestamp("2000-01-01 16:30").time()
    end_t = pd.Timestamp("2000-01-01 23:00").time()
    assert all((t >= start_t) & (t <= end_t))


def test_session_filter_removes_outside(sample_bars):
    filtered = filter_session(sample_bars, start="16:30", end="23:00")
    t = filtered.index.time
    # No bar before 16:30
    assert all(t >= pd.Timestamp("2000-01-01 16:30").time())
    # No bar after 23:00
    assert all(t <= pd.Timestamp("2000-01-01 23:00").time())


def test_session_id_increments_per_day():
    idx = pd.date_range("2025-01-06 16:30", periods=60, freq="1min", tz="Etc/GMT-3")
    idx2 = pd.date_range("2025-01-07 16:30", periods=60, freq="1min", tz="Etc/GMT-3")
    combined = idx.append(idx2)
    df = pd.DataFrame({"close": 1.0}, index=combined)
    df = add_session_id(df)
    sessions = df["session_id"].unique()
    assert len(sessions) == 2
    assert sessions[0] != sessions[1]
