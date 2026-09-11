"""Chain A tests: per-timeframe technical features are causally aligned.

M1 bar t sees only completed HTF candles (one-period shift before ffill).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from quant_rl.data.resample import resample
from quant_rl.features.build import build_features
from quant_rl.features.indicators import build_indicators

HTF_TFS = ["M5", "M15", "H1"]


@pytest.fixture
def feat_cfg():
    """Minimal features config matching config/default.yaml keys."""
    return OmegaConf.create(
        {
            "ema_periods": [9, 21],
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_period": 14,
            "atr_period": 14,
            "adx_period": 14,
            "bb_period": 20,
            "bb_std": 2.0,
            "stoch_k": 14,
            "stoch_d": 3,
            "return_horizons": [1, 5],
            "realized_vol_period": 20,
            "smt_swing_period": 5,
            "smt_corr_window": 20,
            "zscore_window": 100,
            "htf_timeframes": HTF_TFS,
        }
    )


def _expected_htf_value(
    tf: str, col: str, t: pd.Timestamp, m1_bars: pd.DataFrame, feat_cfg
) -> float:
    """Value M1 t should see: last HTF bar that has already completed by t."""
    htf = build_indicators(resample(m1_bars, tf), feat_cfg)  # type: ignore[arg-type]
    completed = htf.shift(1)
    eligible = completed.loc[completed.index <= t]
    assert not eligible.empty, f"no completed {tf} bar at or before {t}"
    return float(eligible[col].iloc[-1])


def test_htf_columns_present(m1_bars, feat_cfg):
    feat = build_features(m1_bars, cfg=OmegaConf.create({"features": feat_cfg}))
    for tf in HTF_TFS:
        htf_cols = [c for c in feat.columns if str(c).startswith(f"{tf}_")]
        assert htf_cols, f"no {tf}_* columns in feature matrix"
        for native in ["ema_9", "rsi"]:
            assert f"{tf}_{native}" in [str(c) for c in feat.columns]


def test_align_timeframes_completed_bar():
    """M1 during a forming H1 bar must not see that H1's final OHLC."""
    from quant_rl.data.align import align_timeframes

    idx = pd.date_range("2025-01-06 16:00", periods=120, freq="1min", tz="Etc/GMT-3")
    close = np.arange(120, dtype=float)
    m1 = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "v": close},
        index=idx,
    )
    h1 = m1.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "v": "last"}
    )
    aligned = align_timeframes(m1[["v"]], {"H1": h1[["close"]]})
    t1630 = pd.Timestamp("2025-01-06 16:30", tz="Etc/GMT-3")
    t17 = pd.Timestamp("2025-01-06 17:00", tz="Etc/GMT-3")
    hour_16 = pd.Timestamp("2025-01-06 16:00", tz="Etc/GMT-3")
    hour_16_close = float(h1["close"].loc[hour_16])
    assert (
        pd.isna(aligned.loc[t1630, "H1_close"]) or aligned.loc[t1630, "H1_close"] != hour_16_close
    )
    assert aligned.loc[t17, "H1_close"] == pytest.approx(hour_16_close)


def test_htf_alignment_causal_open_time(m1_bars, feat_cfg):
    """Feature at bar t equals the HTF value from the last completed HTF bar."""
    from quant_rl.data.align import align_timeframes

    for tf in HTF_TFS:
        htf = build_indicators(resample(m1_bars, tf), feat_cfg)  # type: ignore[arg-type]
        aligned = align_timeframes(m1_bars[["close"]], {tf: htf[["ema_9"]]})[f"{tf}_ema_9"]
        for t in m1_bars.index[300::100]:
            expected = _expected_htf_value(tf, "ema_9", t, m1_bars, feat_cfg)
            if np.isnan(expected):
                continue
            assert abs(aligned.loc[t] - expected) < 1e-9, (
                f"{tf} value at {t} not from last completed bar"
            )


def test_htf_future_invariance(m1_bars, feat_cfg):
    """Truncating future M1 bars must not change past HTF feature values."""
    cfg_full = OmegaConf.create({"features": feat_cfg})
    feat_full = build_features(m1_bars, cfg=cfg_full)
    feat_trunc = build_features(m1_bars.iloc[:400], cfg=cfg_full)

    overlap = feat_trunc.index[:250]
    htf_cols = [c for c in feat_full.columns if any(str(c).startswith(f"{tf}_") for tf in HTF_TFS)]
    pd.testing.assert_frame_equal(
        feat_full.loc[overlap, htf_cols],
        feat_trunc.loc[overlap, htf_cols],
        rtol=1e-9,
    )


def test_htf_disabled_when_empty(m1_bars, feat_cfg):
    """Empty htf_timeframes must reproduce the M1-only feature matrix."""
    cfg = feat_cfg.copy()
    OmegaConf.set_struct(cfg, False)
    cfg.htf_timeframes = []
    feat = build_features(m1_bars, cfg=OmegaConf.create({"features": cfg}))
    assert not any(str(c).startswith(tuple(HTF_TFS)) for c in feat.columns)


def test_mtf_features_use_only_completed_htf_bars(m1_bars, feat_cfg):
    test_htf_alignment_causal_open_time(m1_bars, feat_cfg)  # type: ignore[no-untyped-call]
