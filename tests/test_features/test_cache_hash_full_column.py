"""Feature-cache digest must change when any mid-series bar changes (plan B3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.build import feature_cache_content_hash


def _bars(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=n, freq="min", tz="UTC")
    close = 100 + np.cumsum(np.ones(n) * 0.01)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0},
        index=idx,
    )


def test_mid_series_change_changes_digest() -> None:
    base = _bars()
    changed = base.copy()
    changed.loc[changed.index[250], "close"] = (
        float(changed.loc[changed.index[250], "close"]) + 0.5
    )  # interior bar, endpoints unchanged
    assert feature_cache_content_hash(None, base) != feature_cache_content_hash(None, changed)


def test_identical_data_gives_identical_digest() -> None:
    assert feature_cache_content_hash(None, _bars()) == feature_cache_content_hash(None, _bars())


def test_train_mask_interior_change_changes_digest() -> None:
    bars = _bars()
    mask_a = pd.Series(True, index=bars.index)
    mask_b = mask_a.copy()
    mask_b.iloc[250] = False
    assert feature_cache_content_hash(None, bars, mask_a) != feature_cache_content_hash(
        None, bars, mask_b
    )


if __name__ == "__main__":
    pytest.main([__file__])
