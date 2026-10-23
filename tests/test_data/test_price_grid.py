from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from market_data.transforms.price_grid import (
    configured_price_grid_increment,
    quantize_bar_prices,
    quantize_prices,
)


def test_quarter_grid_rounding_matches_requested_example_and_tie_rule() -> None:
    values = quantize_prices([25633.68, 10.125, -10.125], 0.25)

    assert values == pytest.approx([25633.75, 10.25, -10.25])


def test_bar_quantization_changes_only_ohlc_and_preserves_source() -> None:
    source = pd.DataFrame(
        {
            "open": [25633.68],
            "high": [25633.99],
            "low": [25633.61],
            "close": [25633.68],
            "spread": [60.0],
            "tickvol": [123],
        }
    )

    actual = quantize_bar_prices(source, 0.25)

    actual_ohlc = np.array(
        [
            actual["open"].iloc[0],
            actual["high"].iloc[0],
            actual["low"].iloc[0],
            actual["close"].iloc[0],
        ],
        dtype=np.float64,
    )
    assert actual_ohlc == pytest.approx([25633.75, 25634.0, 25633.5, 25633.75])
    assert actual.loc[0, "spread"] == 60.0
    assert actual.loc[0, "tickvol"] == 123
    assert source.loc[0, "close"] == 25633.68


def test_price_grid_is_disabled_by_default_and_validates_enabled_increment() -> None:
    assert configured_price_grid_increment(OmegaConf.create({"data": {}})) is None
    cfg = OmegaConf.create({"data": {"price_discretization": {"enabled": True, "increment": 0.25}}})
    assert configured_price_grid_increment(cfg) == 0.25

    cfg.data.price_discretization.increment = 0.0
    with pytest.raises(ValueError, match="finite positive increment"):
        configured_price_grid_increment(cfg)


def test_quantization_rejects_nonpositive_grid() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        quantize_prices(np.asarray([1.0]), 0.0)
