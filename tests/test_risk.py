"""Tests for risk sizing and SL/TP computation."""
import numpy as np
import pytest

from quant_rl.backtest.risk import (
    compute_sl_price,
    compute_tp_price,
    compute_lots,
)


class TestComputeSLPrice:
    """Test SL price computation from swings."""

    def test_long_sl_from_swing_low(self):
        """Long SL should be swing low minus buffer."""
        sl = compute_sl_price(
            entry_price=100.0,
            direction=1,
            last_swing_low=95.0,
            last_swing_high=105.0,
            buffer_pts=1.0,
        )
        assert sl == pytest.approx(94.0)

    def test_short_sl_from_swing_high(self):
        """Short SL should be swing high plus buffer."""
        sl = compute_sl_price(
            entry_price=100.0,
            direction=-1,
            last_swing_low=95.0,
            last_swing_high=105.0,
            buffer_pts=1.0,
        )
        assert sl == pytest.approx(106.0)

    def test_long_sl_fallback_no_swing(self):
        """Long SL fallback when no swing available."""
        sl = compute_sl_price(
            entry_price=100.0,
            direction=1,
            last_swing_low=None,
            last_swing_high=None,
            buffer_pts=1.0,
        )
        assert sl == pytest.approx(98.0)

    def test_short_sl_fallback_no_swing(self):
        """Short SL fallback when no swing available."""
        sl = compute_sl_price(
            entry_price=100.0,
            direction=-1,
            last_swing_low=None,
            last_swing_high=None,
            buffer_pts=1.0,
        )
        assert sl == pytest.approx(102.0)


class TestComputeTPPrice:
    """Test TP price computation from R:R ratio."""

    def test_long_tp_price(self):
        """Long TP = entry + rr_ratio * R."""
        tp = compute_tp_price(
            entry_price=100.0,
            sl_price=95.0,
            direction=1,
            rr_ratio=2.0,
        )
        # R = 100 - 95 = 5
        # TP = 100 + 2.0 * 5 = 110
        assert tp == pytest.approx(110.0)

    def test_short_tp_price(self):
        """Short TP = entry - rr_ratio * R."""
        tp = compute_tp_price(
            entry_price=100.0,
            sl_price=105.0,
            direction=-1,
            rr_ratio=2.0,
        )
        # R = 105 - 100 = 5
        # TP = 100 - 2.0 * 5 = 90
        assert tp == pytest.approx(90.0)


class TestComputeLots:
    """Test lot sizing from equity and risk."""

    def test_basic_lot_calculation(self):
        """lots = risk_usd / (sl_distance * contract_size)."""
        lots = compute_lots(
            equity=100_000.0,
            risk_frac=0.01,  # 1% = $1000
            sl_distance=40.0,
            contract_size=1.0,
            max_loss_usd=None,
        )
        # risk_usd = 100,000 * 0.01 = 1000
        # lots = 1000 / (40 * 1.0) = 25
        assert lots == pytest.approx(25.0)

    def test_lot_capped_by_max_loss(self):
        """Lots should be capped by max_loss_usd."""
        lots = compute_lots(
            equity=100_000.0,
            risk_frac=0.01,  # Would give 25 lots normally
            sl_distance=40.0,
            contract_size=1.0,
            max_loss_usd=100.0,
        )
        # max_loss_usd cap: lots = 100 / (40 * 1.0) = 2.5
        # Result should be min of 25 and 2.5 = 2.5
        assert lots == pytest.approx(2.5)

    def test_lot_clipping_min_max(self):
        """Lots should be clipped to [min_lot, max_lot]."""
        lots_small = compute_lots(
            equity=100.0,
            risk_frac=0.001,  # Very small risk
            sl_distance=100.0,
            contract_size=1.0,
            min_lot=0.1,
            max_lot=100.0,
        )
        # Would calculate to < 0.1, should be clipped to 0.1
        assert lots_small >= 0.1

        lots_large = compute_lots(
            equity=10_000_000.0,
            risk_frac=1.0,  # Huge risk
            sl_distance=0.1,
            contract_size=1.0,
            min_lot=0.01,
            max_lot=100.0,
        )
        # Would be huge, should be clipped to 100.0
        assert lots_large <= 100.0

    def test_zero_sl_distance(self):
        """Zero SL distance should return min_lot."""
        lots = compute_lots(
            equity=100_000.0,
            risk_frac=0.01,
            sl_distance=0.0,
            contract_size=1.0,
            min_lot=0.01,
        )
        assert lots == pytest.approx(0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
