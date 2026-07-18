"""Tests for risk calculations (SL/TP/lots)."""
import pytest

from quant_rl.backtest.risk import compute_lots, compute_sl_tp_long, compute_sl_tp_short


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
