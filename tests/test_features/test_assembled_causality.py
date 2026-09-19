"""Assembled-matrix causality harness (T-02.2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from quant_rl.features.build import build_features


def _bars(n: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2025-01-06 00:00", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(1)
    close = 20000.0 + np.cumsum(rng.normal(0, 1.5, n))
    return pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 1.2,
            "low": close - 1.2,
            "close": close,
            "tickvol": rng.integers(10, 200, n),
            "volume": rng.integers(1000, 5000, n),
            "vol": np.zeros(n, dtype=int),
            "spread": np.full(n, 0.6),
            "gap_flag": False,
            "session_id": 0,
        },
        index=idx,
    )


def _thesis_cfg() -> Any:
    return OmegaConf.create(
        {
            "data": {"tz": "Etc/GMT-3"},
            "session": {"tz": "Etc/GMT-3"},
            "strategy": {"session": {"timezone": "Etc/GMT-3"}},
            "features": {
                "ema_periods": [9, 13, 21],
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "adx_period": 14,
                "bb_period": 20,
                "bb_std": 2.0,
                "stoch_k": 14,
                "stoch_d": 3,
                "vwap_session": False,
                "return_horizons": [1, 5, 15],
                "realized_vol_period": 20,
                "smt_swing_period": 5,
                "smt_corr_window": 20,
                "zscore_window": 60,
                "htf_timeframes": ["M5", "M15"],
                "include_po3": True,
                "include_fvg_ifvg": True,
                "include_po3_full": False,
                "include_strategy_state": True,
                "include_session_ohlc": False,
                "include_pd_context": True,
                "structure": {"enabled": True, "timeframes": ["M5", "M15"]},
                "liquidity": {"enabled": True, "timeframes": ["M5"]},
                "smt": {"enabled": False},
            },
        }
    )


@pytest.mark.slow
def test_assembled_matrix_truncation_causality() -> None:
    """Prefix recomputation matches full-run values on the shared index."""
    bars = _bars()
    cfg = _thesis_cfg()
    full = build_features(bars, cfg=cfg, force=True)
    # Cut well after HTF/PD warmup
    cut = bars.index[700]
    prefix = build_features(bars.loc[:cut], cfg=cfg, force=True)
    common = full.index.intersection(prefix.index)
    assert len(common) > 100
    # Compare overlapping columns; NaNs allowed at warmup edges
    cols = [c for c in prefix.columns if c in full.columns]
    a = full.loc[common, cols]
    b = prefix.loc[common, cols]
    # Drop rows that are all-NaN in either frame
    valid = ~(a.isna().all(axis=1) | b.isna().all(axis=1))
    a = a.loc[valid]
    b = b.loc[valid]
    pd.testing.assert_frame_equal(a, b, rtol=1e-5, atol=1e-5, check_freq=False)
