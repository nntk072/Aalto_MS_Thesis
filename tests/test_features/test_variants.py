"""Chain B tests: agent-variant feature blocks (PO3 phase, FVG zones).

The base multi-TF feature matrix (Chain A) must be unchanged when the
variant flags are off; each flag adds exactly its own block. FVG zone
features are validated against a hand-crafted bullish FVG so zone
activity windows are deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from quant_rl.features.build import (
    build_fvg_zone_features,
    build_po3_phase_features,
)


@pytest.fixture
def fvg_bars() -> pd.DataFrame:
    """Bars with a guaranteed bullish FVG: bar1.high < bar3.low.

    Downtrend into bar1, then a sharp up-move leaving a 3-bar imbalance.
    After the gap, price re-enters the zone (stays inside, no fill).
    """
    n = 40
    idx = pd.date_range("2025-01-06 17:00", periods=n, freq="1min", tz="Etc/GMT-3")
    close = np.full(n, 100.0)
    # gentle downtrend bars 0..9
    close[:10] = np.linspace(101.0, 100.0, 10)
    # up-leg bars 10..13 leaving a gap between bar11.high and bar13.low
    close[10] = 100.6
    close[11] = 100.4  # bar1 of the FVG triplet (highest 'high' anchor below)
    close[12] = 102.0
    close[13] = 103.2  # bar3: low must exceed bar11's high
    close[14:] = 102.8  # drift back toward the zone, stay above gap low
    df = pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )
    df.iloc[11, df.columns.get_loc("high")] = 100.8  # bar1 high (gap bottom)
    df.iloc[13, df.columns.get_loc("low")] = 101.5  # bar3 low (gap top)
    return df


def test_po3_phase_known_timestamps():
    idx = pd.date_range("2025-01-06 16:30", periods=395, freq="1min", tz="Etc/GMT-3")
    feats = build_po3_phase_features(idx, session_start="16:30", session_end="23:00")
    # first bar = session open → accumulation (phase 0), progress 0
    assert feats["po3_phase"].iloc[0] == 0
    assert feats["session_progress"].iloc[0] == pytest.approx(0.0)
    # monotonic progress
    assert feats["session_progress"].is_monotonic_increasing
    # 19:30 → 180/390 ≈ 0.46 → manipulation (phase 1)
    assert feats.loc[pd.Timestamp("2025-01-06 19:30", tz="Etc/GMT-3"), "po3_phase"] == 1
    # 22:30 → 360/390 ≈ 0.92 → distribution (phase 2)
    assert feats.loc[pd.Timestamp("2025-01-06 22:30", tz="Etc/GMT-3"), "po3_phase"] == 2


def test_po3_phase_outside_session_nan():
    idx = pd.date_range("2025-01-06 10:00", periods=5, freq="1min", tz="Etc/GMT-3")
    feats = build_po3_phase_features(idx, session_start="16:30", session_end="23:00")
    assert feats["po3_phase"].isna().all()
    assert (feats["session_progress"] == 0.0).all()


def test_fvg_zone_features_in_zone(fvg_bars):
    feats = build_fvg_zone_features(fvg_bars)
    # a bullish zone must exist (up-leg leaves a 3-bar imbalance)
    assert feats["fvg_in_bull"].sum() > 0
    # inside a zone: distance < cap on that side
    assert (feats.loc[feats["fvg_in_bull"] == 1.0, "fvg_bull_dist"] < 5.0).all()
    # before any zone can exist (first 2 bars of the sample): cap on both sides
    assert (feats.iloc[:2][["fvg_bull_dist", "fvg_bear_dist"]] == 5.0).all().all()
    # distances never exceed the cap
    assert feats["fvg_bull_dist"].max() <= 5.0 and feats["fvg_bear_dist"].max() <= 5.0


def test_fvg_zone_features_causal(fvg_bars):
    """Feature values before the gap formation must not depend on later bars."""
    full = build_fvg_zone_features(fvg_bars)
    trunc = build_fvg_zone_features(fvg_bars.iloc[:25])
    overlap = trunc.index[:20]
    pd.testing.assert_frame_equal(full.loc[overlap], trunc.loc[overlap])


def _base_cfg() -> dict:
    return {
        "features": {
            "ema_periods": [9],
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
            "return_horizons": [1],
            "realized_vol_period": 20,
            "smt_swing_period": 5,
            "smt_corr_window": 20,
            "zscore_window": 100,
            "htf_timeframes": ["M5"],
        }
    }


def test_variant_flags_additive(m1_bars):
    from quant_rl.features.build import build_features

    cfg_base = OmegaConf.create(_base_cfg())
    base = build_features(m1_bars, cfg=cfg_base)

    cfg_po3 = OmegaConf.merge(cfg_base, OmegaConf.create({"features": {"include_po3": True}}))
    po3 = build_features(m1_bars, cfg=cfg_po3)
    assert {"po3_phase", "session_progress"} <= set(map(str, po3.columns))
    assert set(map(str, base.columns)) < set(map(str, po3.columns))

    cfg_fvg = OmegaConf.merge(cfg_base, OmegaConf.create({"features": {"include_fvg_ifvg": True}}))
    fvg = build_features(m1_bars, cfg=cfg_fvg)
    assert any(str(c).startswith("M5_fvg_") for c in fvg.columns)
    assert set(map(str, base.columns)) < set(map(str, fvg.columns))


def test_variant_config_files_merge():
    """The three variant YAMLs load and set the right flags over default.yaml."""
    base = OmegaConf.load("quant_rl/config/default.yaml")
    variants = [
        ("config/features_technical_mtf.yaml", False, False),
        ("config/features_po3_mtf.yaml", True, False),
        ("config/features_fvg_ifvg_mtf.yaml", False, True),
    ]
    for fname, want_po3, want_fvg in variants:
        merged = OmegaConf.merge(base, OmegaConf.load(fname))
        assert merged.features.include_po3 is want_po3
        assert merged.features.include_fvg_ifvg is want_fvg
        assert list(merged.features.htf_timeframes) == ["M5", "M15", "H1"]


def test_po3_full_flag_additive(m1_bars):
    """include_po3_full=true adds the full PO3 entry-trigger columns on top of base."""
    from quant_rl.features.build import build_features

    cfg_base = OmegaConf.create(_base_cfg())
    base = build_features(m1_bars, cfg=cfg_base)

    cfg_full = OmegaConf.merge(cfg_base, OmegaConf.create({"features": {"include_po3_full": True}}))
    full = build_features(m1_bars, cfg=cfg_full)

    assert set(map(str, base.columns)) < set(map(str, full.columns))
    for col in ["entry_long", "entry_short", "entry_trigger_type"]:
        assert col in set(map(str, full.columns))
    # matrix must stay homogeneous numeric (TradingEnv casts to float32)
    assert all(pd.api.types.is_numeric_dtype(full[c]) for c in full.columns)
    # entry signals are binary; trigger type is encoded 0..3
    assert set(full["entry_long"].dropna().unique()).issubset({0.0, 1.0})
    assert set(full["entry_short"].dropna().unique()).issubset({0.0, 1.0})
    assert set(full["entry_trigger_type"].dropna().unique()).issubset({0.0, 1.0, 2.0, 3.0})


def test_po3_full_no_lookahead_on_entries(m1_bars):
    """Entry signals at bar t must be unchanged by appending future bars.

    detect_po3_entries consumes only bars <= t when marking bar t (the M1
    loop uses each bar's own close once), so truncation must not alter past
    entry columns.
    """
    from quant_rl.features.build import build_features

    cfg = OmegaConf.merge(
        OmegaConf.create(_base_cfg()), OmegaConf.create({"features": {"include_po3_full": True}})
    )
    full = build_features(m1_bars, cfg=cfg)
    trunc = build_features(m1_bars.iloc[:400], cfg=cfg)

    overlap = trunc.index[:300]
    entry_cols = ["entry_long", "entry_short", "entry_trigger_type"]
    pd.testing.assert_frame_equal(
        full.loc[overlap, entry_cols],
        trunc.loc[overlap, entry_cols],
        rtol=1e-9,
    )


def test_combined_config_merges_all_blocks():
    """config/features_full_po3_mtf.yaml enables all three additive blocks."""
    base = OmegaConf.load("quant_rl/config/default.yaml")
    merged = OmegaConf.merge(base, OmegaConf.load("config/features_full_po3_mtf.yaml"))
    assert merged.features.include_po3 is True
    assert merged.features.include_fvg_ifvg is True
    assert merged.features.include_po3_full is True
    assert merged.features.po3_htf == "M15"
    assert merged.features.po3_ltf == "M5"
    assert list(merged.features.htf_timeframes) == ["M5", "M15", "H1"]
