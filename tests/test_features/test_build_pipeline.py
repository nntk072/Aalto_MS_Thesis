"""Full-pipeline build_features test with golden-file regression.

Exercises build_features() end-to-end with the default config on a small
deterministic dataset, verifies the output schema and data quality, and
regresses against a SHA256 golden hash so a silent schema change fails CI.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from quant_rl.features.build import FEATURE_CACHE_VERSION, build_features


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
    """Minimal default config matching quant_rl/config/default.yaml features block."""
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
            "vwap_session": True,
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


_GOLDEN_HASH = "97b3465f04ec3a781447e974822fb7f83a5a19a5209541c48a26449be88b2d62"
