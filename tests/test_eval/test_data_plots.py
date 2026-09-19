"""Smoke tests for data/EDA thesis charts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.eval.data_plots import write_data_eda


def test_write_data_eda_creates_core_pngs(tmp_path):
    idx = pd.date_range("2025-01-02 16:30", periods=500, freq="1min")
    rng = np.random.default_rng(0)
    close = 20000.0 + np.cumsum(rng.normal(0, 1, 500))
    bars = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(10, 100, 500),
        },
        index=idx,
    )
    features = pd.DataFrame(
        {
            "rsi": rng.uniform(20, 80, 500),
            "atr": rng.uniform(1, 5, 500),
            "entry_long": rng.integers(0, 2, 500),
            "entry_short": rng.integers(0, 2, 500),
        },
        index=idx,
    )
    out = write_data_eda(
        tmp_path / "data",
        bars,
        features,
        train_end="2025-01-02 20:00",
        test_start="2025-01-02 20:01",
        dpi=72,
        po3_sample=False,
    )
    for name in (
        "coverage_calendar.png",
        "log_return_dist.png",
        "session_profile.png",
        "train_test_split.png",
        "feature_histograms.png",
        "signal_counts.png",
    ):
        path = out / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
