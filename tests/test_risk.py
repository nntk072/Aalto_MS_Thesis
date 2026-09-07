"""Tests for risk calculations (SL/TP/lots)."""

import numpy as np
import pandas as pd
import pytest

from quant_rl.backtest.risk import (
    compute_lots,
    compute_sl_tp_from_structure,
    compute_sl_tp_long,
    compute_sl_tp_short,
    resolve_tp_target,
)


def test_compute_sl_tp_long() -> None:
    """Test SL/TP computation for long trades."""
    sl, tp = compute_sl_tp_long(
        entry_price=100.0,
        last_swing_low=95.0,
        buffer_pts=1.0,
        rr_ratio=2.0,
    )
    assert sl == 94.0  # 95.0 - 1.0
    # R = entry - SL = 100 - 94 = 6
    # TP = entry + rr * R = 100 + 2*6 = 112
    assert tp == 112.0


def test_compute_sl_tp_short() -> None:
    """Test SL/TP computation for short trades."""
    sl, tp = compute_sl_tp_short(
        entry_price=100.0,
        last_swing_high=105.0,
        buffer_pts=1.0,
        rr_ratio=2.0,
    )
    assert sl == 106.0  # 105.0 + 1.0
    assert tp == 88.0  # 100 - 2*(106-100)


def test_compute_lots_basic() -> None:
    """Test basic lot sizing."""
    lots = compute_lots(
        equity=100_000.0,
        risk_frac=0.01,  # 1%
        entry_price=100.0,
        sl_price=95.0,  # 5-point risk
        contract_size=1.0,
        min_lot=0.01,
        max_lot=100.0,
    )
    # risk_usd = 100_000 * 0.01 = 1_000
    # lots = 1_000 / (5 * 1.0) = 200, clamped to max_lot=100
    assert lots == 100.0


def test_compute_lots_with_cap() -> None:
    """Test lot sizing with max_loss_cap."""
    lots = compute_lots(
        equity=100_000.0,
        risk_frac=0.01,
        entry_price=100.0,
        sl_price=95.0,  # 5-point risk
        contract_size=1.0,
        min_lot=0.01,
        max_lot=100.0,
        max_loss_cap=50.0,  # Cap at $50
    )
    # max_lots_from_cap = 50 / (5 * 1.0) = 10
    # risk would be 100 but cap limits to 10
    assert lots == 10.0


def test_compute_lots_respects_min_max() -> None:
    """Test that min/max lot bounds are respected."""
    # Very wide SL → natural lots within bounds
    lots = compute_lots(
        equity=100_000.0,
        risk_frac=0.01,
        entry_price=100.0,
        sl_price=10.0,  # huge risk (90 pts)
        contract_size=1.0,
        min_lot=0.01,
        max_lot=100.0,
    )
    # Natural: 1_000 / (90 * 1.0) ≈ 11.11, within bounds
    assert 0.01 <= lots <= 100.0
    assert abs(lots - 11.11) < 0.1

    # Very tight SL → natural lots would exceed max_lot
    lots = compute_lots(
        equity=100_000.0,
        risk_frac=0.01,
        entry_price=100.0,
        sl_price=99.99,  # tiny risk (0.01 pts)
        contract_size=1.0,
        min_lot=0.01,
        max_lot=100.0,
    )
    # Natural: 1_000 / (0.01 * 1.0) = 100_000, clamped to max_lot=100
    assert lots == 100.0


def test_compute_lots_with_contract_size() -> None:
    """Test lot sizing with contract multiplier."""
    lots = compute_lots(
        equity=100_000.0,
        risk_frac=0.01,
        entry_price=100.0,
        sl_price=95.0,  # 5-point risk
        contract_size=100.0,  # US100 multiplier
        min_lot=0.01,
        max_lot=100.0,
    )
    # risk_usd = 100_000 * 0.01 = 1_000
    # lots = 1_000 / (5 * 100.0) = 2.0
    assert lots == 2.0


def _row(**overrides: float) -> pd.Series:
    base = {
        "sweep_high_level": 110.0,
        "sweep_low_level": 90.0,
        "london_high": 108.0,
        "london_low": 92.0,
        "prev_day_high": np.nan,
        "prev_day_low": np.nan,
    }
    base.update(overrides)
    return pd.Series(base)


class TestComputeSlTpFromStructure:
    def test_long_exact_mode(self) -> None:
        # Agent.md §27: manipulation_low = 100, entry = 105 -> SL = 100.
        sl, tp = compute_sl_tp_from_structure(
            direction=1, entry_price=105.0, structure_level=100.0, rr_ratio=2.0, buffer_pts=0.0
        )
        assert sl == 100.0
        assert tp == 115.0

    def test_long_buffered_mode(self) -> None:
        sl, _ = compute_sl_tp_from_structure(
            direction=1, entry_price=105.0, structure_level=100.0, rr_ratio=2.0, buffer_pts=1.0
        )
        assert sl == 99.0

    def test_short_exact_mode(self) -> None:
        # Agent.md §27: manipulation_high = 110, entry = 105 -> SL = 110.
        sl, tp = compute_sl_tp_from_structure(
            direction=-1, entry_price=105.0, structure_level=110.0, rr_ratio=2.0, buffer_pts=0.0
        )
        assert sl == 110.0
        assert tp == 95.0

    def test_invalid_geometry_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be below"):
            compute_sl_tp_from_structure(
                direction=1, entry_price=105.0, structure_level=106.0, rr_ratio=2.0
            )
        with pytest.raises(ValueError, match="must be above"):
            compute_sl_tp_from_structure(
                direction=-1, entry_price=105.0, structure_level=104.0, rr_ratio=2.0
            )
        with pytest.raises(ValueError, match="direction"):
            compute_sl_tp_from_structure(
                direction=0, entry_price=105.0, structure_level=100.0, rr_ratio=2.0
            )


class TestResolveTpTarget:
    def test_rr_mode(self) -> None:
        tp = resolve_tp_target(
            direction=1,
            entry_price=105.0,
            sl_price=100.0,
            rr_ratio=2.0,
            target_mode="rr",
            row=_row(),
        )
        assert tp == 115.0

    def test_buyside_liquidity_long(self) -> None:
        tp = resolve_tp_target(
            direction=1,
            entry_price=105.0,
            sl_price=100.0,
            rr_ratio=2.0,
            target_mode="buyside_liquidity",
            row=_row(),
        )
        assert tp == 110.0

    def test_sellside_liquidity_short(self) -> None:
        tp = resolve_tp_target(
            direction=-1,
            entry_price=105.0,
            sl_price=110.0,
            rr_ratio=2.0,
            target_mode="sellside_liquidity",
            row=_row(),
        )
        assert tp == 90.0

    def test_invalid_target_falls_back_to_rr(self) -> None:
        # Buy-side level on the wrong side of a short entry -> rr fallback.
        tp = resolve_tp_target(
            direction=-1,
            entry_price=105.0,
            sl_price=110.0,
            rr_ratio=2.0,
            target_mode="buyside_liquidity",
            row=_row(),
        )
        assert tp == 95.0

    def test_nan_target_falls_back_to_rr(self) -> None:
        tp = resolve_tp_target(
            direction=1,
            entry_price=105.0,
            sl_price=100.0,
            rr_ratio=2.0,
            target_mode="previous_day_high_low",
            row=_row(),
        )
        assert tp == 115.0

    def test_never_creates_invalid_tp(self) -> None:
        # A structural target below a long entry must never be returned.
        tp = resolve_tp_target(
            direction=1,
            entry_price=105.0,
            sl_price=100.0,
            rr_ratio=2.0,
            target_mode="sellside_liquidity",  # 90.0 < entry -> invalid
            row=_row(),
        )
        assert tp == 115.0

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_mode"):
            resolve_tp_target(
                direction=1,
                entry_price=105.0,
                sl_price=100.0,
                rr_ratio=2.0,
                target_mode="moon",
                row=_row(),
            )
