"""Full-pipeline build_features test with golden-file regression.

Exercises build_features() end-to-end with the default config on a small
deterministic dataset, verifies the output schema and data quality, and
regresses against a SHA256 golden hash so a silent schema change fails CI.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from features import (
    FEATURE_CACHE_VERSION,
    build_features,
    select_legacy_obs_columns,
    select_obs_columns,
)


@pytest.fixture
def deterministic_bars() -> pd.DataFrame:
    """Deterministic M1 bars (seed=0) spanning Asian+London+NY sessions."""
    n = 600
    idx = pd.date_range("2025-01-06 01:05", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(0)
    close = 20000.0 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, n),
            "high": close + rng.uniform(0, 2, n),
            "low": close - rng.uniform(0, 2, n),
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
    df.index.name = "datetime"
    return df


def _default_cfg() -> dict[str, Any]:
    """Minimal default config matching config/base/default.yaml features block."""
    return {
        "features": {
            "ema_periods": [9, 13, 21, 50, 200],
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
            "return_horizons": [1, 5, 15, 30],
            "realized_vol_period": 20,
            "smt_swing_period": 5,
            "smt_corr_window": 20,
            "zscore_window": 252,
            "htf_timeframes": ["M5", "M15", "H1"],
            "include_po3": False,
            "include_fvg_ifvg": False,
            "include_po3_full": False,
            "include_strategy_state": False,
            "po3_reference_mode": "m1_legacy",
            "include_session_ohlc": False,
        }
    }


class TestBuildFeaturesPipeline:
    """End-to-end build_features pipeline tests."""

    def test_pipeline_produces_frame(self, deterministic_bars: pd.DataFrame) -> None:
        """build_features returns a DataFrame with the same length as input."""
        cfg = OmegaConf.create(_default_cfg())
        feat = build_features(deterministic_bars, cfg=cfg)
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) == len(deterministic_bars)

    def test_expected_base_columns_present(self, deterministic_bars: pd.DataFrame) -> None:
        """Default config produces indicator + structure + session columns."""
        cfg = OmegaConf.create(_default_cfg())
        feat = build_features(deterministic_bars, cfg=cfg)
        cols = set(map(str, feat.columns))
        # Indicator columns
        assert "ret_1" in cols
        assert "rsi" in cols
        assert "atr" in cols
        assert "ema_9" in cols
        assert "ema_200" in cols
        assert "macd" in cols
        # Structure levels
        assert "last_swing_high" in cols
        assert "last_swing_low" in cols
        # Session levels
        assert "asian_high" in cols
        assert "asian_low" in cols
        assert "volume_spike" in cols
        assert "atr_5" in cols
        assert "vwap_dist" not in cols
        # HTF blocks
        assert any(c.startswith("M5_") for c in cols)
        assert any(c.startswith("M15_") for c in cols)
        assert any(c.startswith("H1_") for c in cols)

    def test_no_unexpected_nan_in_body(self, deterministic_bars: pd.DataFrame) -> None:
        """After the warm-up window, features should not be all-NaN."""
        cfg = OmegaConf.create(_default_cfg())
        feat = build_features(deterministic_bars, cfg=cfg)
        body = feat.iloc[252:]  # skip zscore warm-up
        non_nan_cols = body.columns[body.notna().any()]
        assert len(non_nan_cols) > 10, "Too many all-NaN columns in body"

    def test_no_inf_in_output(self, deterministic_bars: pd.DataFrame) -> None:
        """build_features output should not contain Inf values."""
        cfg = OmegaConf.create(_default_cfg())
        feat = build_features(deterministic_bars, cfg=cfg)
        numeric = feat.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_determinism(self, deterministic_bars: pd.DataFrame) -> None:
        """Two calls with the same input produce identical output."""
        cfg = OmegaConf.create(_default_cfg())
        feat1 = build_features(deterministic_bars, cfg=cfg)
        feat2 = build_features(deterministic_bars, cfg=cfg)
        pd.testing.assert_frame_equal(feat1, feat2, rtol=1e-12)

    def test_cache_version_matches(self) -> None:
        """FEATURE_CACHE_VERSION is a non-empty string."""
        assert isinstance(FEATURE_CACHE_VERSION, str)
        assert len(FEATURE_CACHE_VERSION) > 0

    def test_configured_mtf_swing_candidates_are_exposed_but_not_observations(
        self, deterministic_bars: pd.DataFrame
    ) -> None:
        cfg_dict = _default_cfg()
        cfg = cast(
            DictConfig,
            OmegaConf.merge(
                OmegaConf.create(cfg_dict),
                OmegaConf.create({"costs": {"point_size": 0.01}}),
                OmegaConf.load("config/features/pd_context_mtf_swing_candidates.yaml"),
            ),
        )

        features = build_features(deterministic_bars, cfg=cfg)
        candidates = [
            f"{timeframe}_swing_{side}_candidate_{slot}{suffix}"
            for timeframe in ("M5", "M15", "H1")
            for side in ("high", "low")
            for slot in range(1, 4)
            for suffix in ("", "_pivot_time_ms", "_confirmed_at_ms", "_age_bars")
        ]
        assert set(candidates).issubset(features.columns)
        assert features[candidates].notna().any().any()
        assert set(candidates).isdisjoint(select_obs_columns(features).columns)
        decision_ms = features.index.to_numpy(dtype="datetime64[ns]").view(np.int64) // 1_000_000
        for timeframe in ("M5", "M15", "H1"):
            for side in ("high", "low"):
                for slot in range(1, 4):
                    name = f"{timeframe}_swing_{side}_candidate_{slot}"
                    price = features[name]
                    pivot = features[f"{name}_pivot_time_ms"]
                    confirmed = features[f"{name}_confirmed_at_ms"]
                    age = features[f"{name}_age_bars"]
                    valid = price.notna()
                    valid_decisions = decision_ms[valid.to_numpy()]
                    assert (pivot.loc[valid].to_numpy() <= confirmed.loc[valid].to_numpy()).all()
                    assert (confirmed.loc[valid].to_numpy() <= valid_decisions).all()
                    assert (age.loc[valid].to_numpy() >= 0.0).all()

    def test_absolute_event_metadata_is_not_exposed_as_model_features(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="1min")
        features = pd.DataFrame(
            {
                "po3_ifvg_time_ns": [1.7e18, 1.7e18 + 60_000_000_000],
                "M5_sweep_high_pivot_time_ns": [1.7e18, 1.7e18 + 300_000_000_000],
                "sweep_low_confirmed_at_ns": [1.7e18, 1.7e18 + 60_000_000_000],
                "M5_swing_high_candidate_1_pivot_time_ms": [1.7e12, 1.7e12 + 300_000],
                "previous_session_completed_at_ns": [1.7e18, 1.7e18 + 86_400_000_000_000],
                "po3_ifvg_fvg_origin_bar": [1000.0, 1001.0],
                "po3_setup_id": [42.0, 42.0],
                "po3_ifvg_age": [1.0, 2.0],
                "po3_ifvg_active": [1.0, 1.0],
            },
            index=index,
        )

        observed = select_obs_columns(features)
        legacy_observed = select_legacy_obs_columns(features)

        assert list(observed.columns) == ["po3_ifvg_age", "po3_ifvg_active"]
        assert "po3_ifvg_time_ns" in legacy_observed.columns
        assert "M5_sweep_high_pivot_time_ns" not in legacy_observed.columns

    def test_final_mtf_swing_profile_merges_with_default_config(self) -> None:
        base = OmegaConf.load("config/base/default.yaml")
        overlay = OmegaConf.load("config/features/pd_context_mtf_swing_candidates.yaml")

        merged = OmegaConf.merge(base, overlay)

        assert merged.features.structure.enabled is True
        assert list(merged.features.structure.timeframes) == [
            "M3",
            "M5",
            "M10",
            "M15",
            "M30",
            "H1",
            "H4",
            "D1",
        ]
        assert merged.features.structure.swing_candidate_count == 3
        assert merged.features.structure.candidate_price_increment is None
        assert merged.costs.point_size == 0.01

    def test_htf_sweep_metadata_and_audit_sidecar_are_cached(
        self, deterministic_bars: pd.DataFrame, tmp_path: Any
    ) -> None:
        cfg_data = _default_cfg()
        cfg_data["features"]["include_strategy_state"] = True
        cfg_data["features"]["po3_reference_mode"] = "htf_confirmed"
        cfg_data["features"]["po3_reference_timeframes"] = [
            "M5",
            "M15",
            "M30",
            "H1",
            "H4",
            "D1",
        ]
        cfg_data["features"]["po3_phase_mode"] = "reclaim_then_ifvg"
        cfg_data["strategy"] = {
            "ifvg": {
                "max_age_bars": 50,
                "close_through_threshold": 0.0,
                "require_opposing_fvg": True,
            }
        }
        cfg = OmegaConf.create(cfg_data)
        cache_path = tmp_path / "features.parquet"

        features = build_features(deterministic_bars, cfg=cfg, cache_path=cache_path)
        cached = build_features(deterministic_bars, cfg=cfg, cache_path=cache_path)

        assert "sweep_low_ref_tf_code" in features
        assert "sweep_low_confirmed_at_ns" in features
        assert "po3_distribution_started" in features
        assert (tmp_path / "features.parquet.htf_sweeps.parquet").exists()
        pd.testing.assert_frame_equal(features, cached, check_freq=False)

    def test_golden_hash_matches(self, deterministic_bars: pd.DataFrame) -> None:
        """Regression test: feature matrix SHA256 matches golden value."""
        cfg = OmegaConf.create(_default_cfg())
        feat = build_features(deterministic_bars, cfg=cfg)
        hasher = hashlib.sha256()
        hasher.update(json.dumps(list(map(str, feat.columns))).encode())
        hasher.update(str(feat.shape).encode())
        sample = feat.iloc[::10].round(6)
        hasher.update(np.asarray(pd.util.hash_pandas_object(sample).values).tobytes())
        golden = hasher.hexdigest()
        assert golden == _GOLDEN_HASH, (
            f"Feature golden hash mismatch.\n"
            f"  Expected: {_GOLDEN_HASH}\n"
            f"  Got:      {golden}\n"
            f"  If intentional, update _GOLDEN_HASH."
        )


# Structural targets and the PO3 event columns changed the feature matrix.
_GOLDEN_HASH = "c8ed94bff2fd07aabab1e3f0bcead83ffaa5d6d02f0e1ce8ce277a875e364780"
