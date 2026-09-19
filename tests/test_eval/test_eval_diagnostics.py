"""Tests for bootstrap CI / calibration artifact writers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_rl.eval.eval_diagnostics import write_bootstrap_cis, write_calibration_artifacts


def _synthetic_split(n: int = 200):
    idx = pd.date_range("2025-01-02 16:30", periods=n, freq="1min")
    rng = np.random.default_rng(1)
    equity = pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0001, 0.001, n)), index=idx)
    rows = []
    for i in range(0, n - 20, 20):
        rows.append(
            {
                "type": "open",
                "direction": 1 if i % 40 == 0 else -1,
                "price": 100.0,
                "time": idx[i],
                "bar": i,
            }
        )
        rows.append(
            {
                "type": "close",
                "pnl": float(rng.normal(5, 30)),
                "time": idx[i + 10],
                "bar": i + 10,
            }
        )
    return equity, pd.DataFrame(rows)


def test_write_bootstrap_and_calibration(tmp_path):
    equity, trades = _synthetic_split()
    cis = write_bootstrap_cis(tmp_path, equity, trades, n_boot=50)
    assert (tmp_path / "bootstrap_ci.json").exists()
    assert "sharpe" in cis
    payload = write_calibration_artifacts(tmp_path, trades, dpi=72)
    assert payload is not None
    assert (tmp_path / "calibration.json").exists()
    assert (tmp_path / "reliability.png").stat().st_size > 0
    data = json.loads((tmp_path / "calibration.json").read_text())
    assert "ece" in data
