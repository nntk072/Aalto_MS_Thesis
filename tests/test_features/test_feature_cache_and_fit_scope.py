"""Content-hash feature cache and fit-scoped z-score tests (T-02.1 / T-02.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from quant_rl.features.build import (
    build_features,
    feature_cache_content_hash,
    feature_cache_path,
)
from quant_rl.features.normalize import rolling_zscore


def _bars(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2025-01-06 01:05", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(0)
    close = 20000.0 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
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


def _cfg(**feature_overrides: Any) -> Any:
    base: dict[str, Any] = {
        "data": {"tz": "Etc/GMT-3"},
        "session": {"tz": "Etc/GMT-3"},
        "features": {
            "ema_periods": [9, 13],
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
            "htf_timeframes": [],
            "include_po3": False,
            "include_fvg_ifvg": False,
            "include_po3_full": False,
            "include_strategy_state": False,
            "include_session_ohlc": False,
            "include_pd_context": False,
        },
    }
    base["features"].update(feature_overrides)
    return OmegaConf.create(base)


def test_cache_hash_changes_when_feature_flag_flips() -> None:
    bars = _bars()
    cfg_a = _cfg(include_po3=False)
    cfg_b = _cfg(include_po3=True)
    assert feature_cache_content_hash(cfg_a, bars) != feature_cache_content_hash(cfg_b, bars)


def test_feature_cache_hit_and_miss(tmp_path: Path) -> None:
    bars = _bars()
    cfg = _cfg()
    path = feature_cache_path(tmp_path, "US100", cfg, bars)
    feat1 = build_features(bars, cfg=cfg, cache_path=path)
    mtime1 = path.stat().st_mtime_ns
    feat2 = build_features(bars, cfg=cfg, cache_path=path)
    assert path.stat().st_mtime_ns == mtime1
    pd.testing.assert_frame_equal(feat1, feat2, check_freq=False)

    cfg2 = _cfg(rsi_period=21)
    path2 = feature_cache_path(tmp_path, "US100", cfg2, bars)
    assert path2 != path
    build_features(bars, cfg=cfg2, cache_path=path2)
    assert path2.exists()


def test_fit_scoped_zscore_ignores_future_poison() -> None:
    """With train_mask, future spikes must not change train-window z-scores."""
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    values = np.linspace(0.0, 1.0, n)
    df = pd.DataFrame({"x": values}, index=idx)
    train_mask = pd.Series(np.arange(n) < 100, index=idx)

    clean = rolling_zscore(df, window=20, train_mask=train_mask)
    poisoned = df.copy()
    poisoned.iloc[150:, 0] = 1e6
    scoped = rolling_zscore(poisoned, window=20, train_mask=train_mask)
    pd.testing.assert_series_equal(
        clean.loc[train_mask, "x"],
        scoped.loc[train_mask, "x"],
        check_names=False,
    )
    # Test-side values under train_mask are frozen via ffill of train stats:
    # a late spike must not move the first post-train z-score.
    first_test = ~train_mask
    assert np.isfinite(scoped.loc[first_test, "x"].iloc[0])
    assert scoped.loc[first_test, "x"].iloc[0] == clean.loc[first_test, "x"].iloc[0]
