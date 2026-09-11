"""Tests: session labels, NY mask vs feature dataset, cache versioning."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from quant_rl.data.pipeline import BAR_CACHE_VERSION, _cache_path, run_pipeline
from quant_rl.data.resample import resample
from quant_rl.data.session import (
    add_session_id,
    add_session_labels,
    filter_session,
    get_session,
    ny_session_mask,
)


@pytest.fixture
def sample_bars():
    idx = pd.date_range("2025-01-06 00:00", periods=24 * 60, freq="1min", tz="Etc/GMT-3")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "tickvol": 1, "vol": 0, "spread": 0.6},
        index=idx,
    )


def test_session_filter_keeps_only_window(sample_bars):
    filtered = filter_session(sample_bars, start="16:30", end="23:00")
    t = pd.DatetimeIndex(filtered.index).time
    start_t = pd.Timestamp("2000-01-01 16:30").time()
    end_t = pd.Timestamp("2000-01-01 23:00").time()
    assert all((t >= start_t) & (t <= end_t))


def test_session_filter_removes_outside(sample_bars):
    filtered = filter_session(sample_bars, start="16:30", end="23:00")
    t = pd.DatetimeIndex(filtered.index).time
    assert all(t >= pd.Timestamp("2000-01-01 16:30").time())
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


def test_trading_mask_separate_from_features(sample_bars):
    labeled = add_session_labels(sample_bars)
    mask = ny_session_mask(pd.DatetimeIndex(labeled.index))
    assert len(labeled) == len(sample_bars)
    assert mask.sum() < len(labeled)
    assert labeled.loc[mask, "session"].eq("ny").all()


def test_session_labels_timezone_correct():
    assert get_session("2025-01-06 16:30:00+03:00") == "ny"
    assert get_session("2025-01-06 12:00:00+03:00") == "london"
    naive = pd.Timestamp("2025-01-06 12:00:00")
    assert get_session(naive, tz="Etc/GMT-3") == "london"


def test_get_session_closed_label():
    assert get_session("2025-01-06 23:50:00+03:00") == "closed"
    assert get_session("2025-01-06 01:00:00+03:00") == "closed"


def test_feature_pipeline_keeps_non_ny_bars():
    src = inspect.getsource(run_pipeline)
    body = src.split("Parameters", 1)[-1] if "Parameters" in src else src
    assert "filter_session(" not in body


def test_pipeline_cache_bust_on_version(tmp_path):
    path = _cache_path(tmp_path, "US100.cash", "M1")
    assert BAR_CACHE_VERSION in path.name
    assert path.name != "US100.cash_M1.parquet"


def test_resample_uses_full_day_source(sample_bars):
    h1 = resample(sample_bars, "H1")
    hours = pd.DatetimeIndex(h1.index).hour
    assert (hours < 16).any()
    d1 = resample(sample_bars, "D1")
    assert len(d1) >= 1
