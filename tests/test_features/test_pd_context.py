"""Unit tests for session-conditional PD context (masking + no lookahead)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from quant_rl.data.session import add_session_labels
from quant_rl.features.build import OBS_RAW_PRICE_COLUMNS, build_features
from quant_rl.features.liquidity import detect_bos, detect_mss
from quant_rl.features.pd_context import (
    PD_CONTEXT_FEATURE_COLUMNS,
    build_pd_context_features,
)
from quant_rl.features.structure import structure_levels


def _two_day_bars() -> pd.DataFrame:
    """Two broker days covering Asia → London → NY with known extremes."""
    # Day 1: 2025-01-06 (Mon), Day 2: 2025-01-07
    chunks: list[pd.DataFrame] = []
    rng = np.random.default_rng(7)
    for day, asia_base, lon_base, ny_base in (
        ("2025-01-06", 100.0, 110.0, 105.0),
        ("2025-01-07", 120.0, 130.0, 125.0),
    ):
        for start, end, base in (
            (f"{day} 01:05", f"{day} 08:59", asia_base),
            (f"{day} 09:00", f"{day} 16:29", lon_base),
            (f"{day} 16:30", f"{day} 22:59", ny_base),
        ):
            idx = pd.date_range(start, end, freq="1min", tz="Etc/GMT-3")
            n = len(idx)
            noise = rng.uniform(-0.5, 0.5, n)
            close = base + noise
            # Plant a unique extreme mid-session so completed H/L are identifiable.
            high = close + 0.2
            low = close - 0.2
            mid = n // 2
            high[mid] = base + 5.0
            low[mid + 1 if mid + 1 < n else mid] = base - 5.0
            chunks.append(
                pd.DataFrame(
                    {
                        "open": close,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": np.full(n, 100.0),
                        "tickvol": np.full(n, 50),
                    },
                    index=idx,
                )
            )
    out = pd.concat(chunks).sort_index()
    out.index.name = "datetime"
    return out


@pytest.fixture
def bars() -> pd.DataFrame:
    return _two_day_bars()


class TestPdContextRouting:
    def test_asia_masks_london_and_asia_ctx(self, bars: pd.DataFrame) -> None:
        atr = pd.Series(1.0, index=bars.index)
        ctx = build_pd_context_features(bars, atr)
        asia = add_session_labels(bars)["session"] == "asia"
        assert asia.any()
        assert (ctx.loc[asia, "ctx_asia_active"] == 0.0).all()
        assert (ctx.loc[asia, "ctx_london_active"] == 0.0).all()
        assert (ctx.loc[asia, "ctx_asia_high"] == 0.0).all()
        assert (ctx.loc[asia, "ctx_london_high"] == 0.0).all()
        assert (ctx.loc[asia, "ctx_asia_high_dist_atr"] == 0.0).all()
        assert (ctx.loc[asia, "ctx_london_high_dist_atr"] == 0.0).all()
        # Prev-day summaries remain available (NaN only on first calendar day).
        day2_asia = asia & (bars.index.date == pd.Timestamp("2025-01-07").date())
        assert day2_asia.any()
        assert ctx.loc[day2_asia, "ctx_prev_day_high"].notna().all()

    def test_london_has_asia_not_completed_self_as_running(self, bars: pd.DataFrame) -> None:
        atr = pd.Series(1.0, index=bars.index)
        ctx = build_pd_context_features(bars, atr)
        london = add_session_labels(bars)["session"] == "london"
        assert london.any()
        assert (ctx.loc[london, "ctx_asia_active"] == 1.0).all()
        assert (ctx.loc[london, "ctx_london_active"] == 1.0).all()
        # Completed Asia for day 1 should equal planted asia extreme 105.
        d1_lon = london & (bars.index.date == pd.Timestamp("2025-01-06").date())
        assert np.allclose(ctx.loc[d1_lon, "ctx_asia_high"].to_numpy(), 105.0)
        # London high is live/running — first London bar cannot yet see the mid-session plant.
        first_lon = ctx.loc[d1_lon].iloc[0]
        assert first_lon["ctx_london_high"] < 115.0
        last_lon = ctx.loc[d1_lon].iloc[-1]
        assert np.isclose(last_lon["ctx_london_high"], 115.0)

    def test_ny_has_completed_asia_and_london(self, bars: pd.DataFrame) -> None:
        atr = pd.Series(1.0, index=bars.index)
        ctx = build_pd_context_features(bars, atr)
        ny = add_session_labels(bars)["session"] == "ny"
        d1_ny = ny & (bars.index.date == pd.Timestamp("2025-01-06").date())
        assert d1_ny.any()
        assert (ctx.loc[d1_ny, "ctx_asia_active"] == 1.0).all()
        assert (ctx.loc[d1_ny, "ctx_london_active"] == 1.0).all()
        assert np.allclose(ctx.loc[d1_ny, "ctx_asia_high"].to_numpy(), 105.0)
        assert np.allclose(ctx.loc[d1_ny, "ctx_london_high"].to_numpy(), 115.0)

    def test_no_lookahead_across_session_boundary(self, bars: pd.DataFrame) -> None:
        """First London bar must not see London's eventual session high."""
        atr = pd.Series(1.0, index=bars.index)
        ctx = build_pd_context_features(bars, atr)
        d1 = bars.index.date == pd.Timestamp("2025-01-06").date()
        london = (add_session_labels(bars)["session"] == "london") & d1
        ny = (add_session_labels(bars)["session"] == "ny") & d1
        first_lon_hi = float(ctx.loc[london, "ctx_london_high"].iloc[0])
        final_lon_hi = float(ctx.loc[ny, "ctx_london_high"].iloc[0])
        assert first_lon_hi < final_lon_hi
        # Asia bars must not see same-day London at all.
        asia = (add_session_labels(bars)["session"] == "asia") & d1
        assert (ctx.loc[asia, "ctx_london_high"] == 0.0).all()

    def test_fixed_width_columns(self, bars: pd.DataFrame) -> None:
        atr = pd.Series(1.0, index=bars.index)
        ctx = build_pd_context_features(bars, atr)
        for col in PD_CONTEXT_FEATURE_COLUMNS:
            assert col in ctx.columns


class TestMssCausal:
    def test_mss_flips_after_opposite_bos(self) -> None:
        idx = pd.date_range("2025-01-06 16:30", periods=40, freq="1min", tz="Etc/GMT-3")
        close = np.concatenate(
            [
                np.linspace(100, 90, 10),   # down
                np.linspace(90, 95, 10),    # bounce
                np.linspace(95, 85, 10),    # continue down → bos_down likely
                np.linspace(85, 100, 10),   # reverse up → bos_up / mss_up
            ]
        )
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
            },
            index=idx,
        )
        struct = structure_levels(bars, swing_period=2)[["last_swing_high", "last_swing_low"]]
        bos = detect_bos(bars, struct)
        mss = detect_mss(bars, struct, bos)
        # If any MSS fires, it must coincide with a BOS on the same bar.
        mss_up_idx = mss.index[mss["mss_up"] == 1]
        for t in mss_up_idx:
            assert bos.loc[t, "bos_up"] == 1
        mss_dn_idx = mss.index[mss["mss_down"] == 1]
        for t in mss_dn_idx:
            assert bos.loc[t, "bos_down"] == 1


def _pd_cfg(**overrides: Any) -> OmegaConf:
    base: dict[str, Any] = {
        "features": {
            "ema_periods": [9, 21],
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
            "return_horizons": [1, 5],
            "realized_vol_period": 20,
            "smt_swing_period": 5,
            "smt_corr_window": 20,
            "zscore_window": 50,
            "htf_timeframes": ["M5", "M15"],
            "include_po3": False,
            "include_fvg_ifvg": False,
            "include_po3_full": False,
            "include_strategy_state": False,
            "include_session_ohlc": False,
            "include_pd_context": True,
            "structure": {"enabled": True, "timeframes": ["M5", "M15"]},
        },
        "session": {"timezone": "Etc/GMT-3"},
    }
    base["features"].update(overrides)
    return OmegaConf.create(base)


class TestBuildFeaturesPdContext:
    def test_pd_columns_and_htf_pd_present(self, bars: pd.DataFrame) -> None:
        feat = build_features(bars, cfg=_pd_cfg())
        for col in ("ctx_asia_active", "bos_up", "mss_up", "active_session_ny"):
            assert col in feat.columns
        assert any(c.startswith("M15_pd_") for c in map(str, feat.columns))
        assert any(c.startswith("M5_bos_") or c == "M5_bos_up" for c in map(str, feat.columns))

    def test_ny_nonzero_asia_zero_mask_in_pipeline(self, bars: pd.DataFrame) -> None:
        feat = build_features(bars, cfg=_pd_cfg())
        labeled = add_session_labels(bars)
        ny = labeled["session"] == "ny"
        asia = labeled["session"] == "asia"
        # Align masks to feat index (dropna may trim leading rows).
        ny = ny.reindex(feat.index).fillna(False).astype(bool)
        asia = asia.reindex(feat.index).fillna(False).astype(bool)
        assert (feat.loc[asia, "ctx_london_high_dist_atr"] == 0.0).all()
        # NY bars on day 2 should have non-zero asia active + filled levels.
        d2_ny = ny & (feat.index.date == pd.Timestamp("2025-01-07").date())
        assert d2_ny.any()
        assert (feat.loc[d2_ny, "ctx_asia_active"] == 1.0).all()
        assert (feat.loc[d2_ny, "ctx_asia_high"] != 0.0).any()

    def test_obs_raw_stems_include_ctx_levels(self) -> None:
        assert "ctx_asia_high" in OBS_RAW_PRICE_COLUMNS
        assert "mss_up_level" in OBS_RAW_PRICE_COLUMNS
        assert "asian_high" in OBS_RAW_PRICE_COLUMNS


class TestConfigYaml:
    def test_idea1_enables_pd_context(self) -> None:
        from quant_rl.config import load_config

        cfg = load_config(config_path="config/idea1_po3_ifvg.yaml")
        assert cfg.features.include_pd_context is True
        assert cfg.features.structure.enabled is True

    def test_features_pd_context_yaml(self) -> None:
        from quant_rl.config import load_config

        cfg = load_config(config_path="config/features_pd_context.yaml")
        assert cfg.features.include_pd_context is True
