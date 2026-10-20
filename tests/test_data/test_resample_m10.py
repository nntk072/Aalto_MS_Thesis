from __future__ import annotations

import pandas as pd

from market_data.transforms.resample import build_all_timeframes, resample


def test_m10_resample_aggregates_ten_left_closed_m1_bars() -> None:
    index = pd.date_range("2025-01-01 00:00", periods=20, freq="min", tz="UTC")
    m1 = pd.DataFrame(
        {
            "open": range(20),
            "high": [value + 2 for value in range(20)],
            "low": [value - 1 for value in range(20)],
            "close": [value + 1 for value in range(20)],
            "tickvol": [1] * 20,
            "vol": [2] * 20,
            "spread": [0.5] * 20,
        },
        index=index,
    )

    m10 = resample(m1, "M10")

    assert m10.index.equals(index[[0, 10]])
    assert m10.iloc[0].to_dict() == {
        "open": 0.0,
        "high": 11.0,
        "low": -1.0,
        "close": 10.0,
        "tickvol": 10.0,
        "vol": 20.0,
        "spread": 0.5,
    }
    assert m10.iloc[1]["open"] == 10
    assert m10.iloc[1]["close"] == 20


def test_build_all_timeframes_includes_m10() -> None:
    index = pd.date_range("2025-01-01", periods=20, freq="min", tz="UTC")
    m1 = pd.DataFrame({"open": range(20)}, index=index)

    frames = build_all_timeframes(m1)

    assert "M10" in frames
    assert len(frames["M10"]) == 2
