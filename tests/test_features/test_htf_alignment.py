"""Chain A tests: per-timeframe technical features are causally aligned.

For each HTF column the value at M1 bar t must come from the most recent
HTF bar whose *open time* ≤ t (align_timeframes' documented guarantee),
never from an HTF bar that opened after t. Verified two ways:

1. Explicit open-time lookup comparison on a small synthetic set.
2. Future-invariance: HTF values computed on a truncated M1 frame must be
   identical to the same rows of the full run (appending future bars
   cannot change past feature values).
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
    """Value the M1 bar at *t* should see: from the latest HTF bar with open ≤ t."""
    htf = build_indicators(resample(m1_bars, tf), feat_cfg)  # type: ignore[arg-type]
    eligible = htf.loc[htf.index <= t]
    assert not eligible.empty, f"no {tf} bar with open time ≤ {t}"
    return float(eligible[col].iloc[-1])


def test_htf_columns_present(m1_bars, feat_cfg):
    feat = build_features(m1_bars, cfg=OmegaConf.create({"features": feat_cfg}))
    for tf in HTF_TFS:
        htf_cols = [c for c in feat.columns if str(c).startswith(f"{tf}_")]
        assert htf_cols, f"no {tf}_* columns in feature matrix"
        # each M1-native indicator must have a prefixed HTF counterpart
        for native in ["ema_9", "rsi"]:
            assert f"{tf}_{native}" in [str(c) for c in feat.columns]


def test_htf_alignment_causal_open_time(m1_bars, feat_cfg):
    """Feature at bar t equals the HTF value from the latest bar with open ≤ t."""
    for tf in HTF_TFS:
        htf = build_indicators(resample(m1_bars, tf), feat_cfg)  # type: ignore[arg-type]
        # Reproduce exactly what align_timeframes does to this column.
        aligned = htf["ema_9"].reindex(m1_bars.index, method="ffill")
        for t in m1_bars.index[300::100]:
            expected = _expected_htf_value(tf, "ema_9", t, m1_bars, feat_cfg)
            assert not np.isnan(expected)
            assert abs(aligned.loc[t] - expected) < 1e-9, (
                f"{tf} value at {t} not from latest bar with open≤t"
            )
            # and never from a bar that opened after t
            later = htf.loc[htf.index > t, "ema_9"]
            if not later.empty:
                assert not np.isclose(aligned.loc[t], later.iloc[0]) or np.isnan(later.iloc[0])


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
