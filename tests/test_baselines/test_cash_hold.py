"""Lot fit and cash-hold simulation for the US100 loss rules."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_rl.baselines.cash_hold import fit_hold_lots, simulate_hold, swap_usd_per_lot


def _bars(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="Etc/GMT-3") for t, _, _ in rows])
    return pd.DataFrame(
        {"close": [c for _, c, _ in rows], "low": [lo for _, _, lo in rows]},
        index=idx,
    )


@pytest.mark.unit
def test_swap_points_convert_at_one_cent() -> None:
    assert swap_usd_per_lot() == pytest.approx(-6.7512)


@pytest.mark.unit
def test_fit_uses_the_tighter_of_max_loss_and_daily_loss() -> None:
    bars = _bars(
        [
            ("2025-01-06 16:30", 100.0, 100.0),
            ("2025-01-07 16:30", 90.0, 40.0),
        ]
    )
    fitted = fit_hold_lots(bars, swap_per_lot=0.0, spread=0.0, obs_window=0)
    # Adverse from entry is 60 points. The session drop is 50, so $5,000 binds.
    assert fitted["lots_from_max_loss"] == pytest.approx(10_000 / 60)
    assert fitted["lots_from_daily_loss"] == pytest.approx(5_000 / 50)
    assert fitted["lots"] == pytest.approx(5_000 / 50)


@pytest.mark.unit
def test_hold_closes_when_the_max_loss_is_reached() -> None:
    bars = _bars(
        [
            ("2025-01-06 16:30", 100.0, 100.0),
            ("2025-01-06 16:31", 80.0, 70.0),
        ]
    )
    out = simulate_hold(
        bars,
        lots=1.0,
        max_loss_usd=10.0,
        daily_loss_usd=5_000.0,
        swap_per_lot=0.0,
        spread=0.0,
        obs_window=0,
    )
    assert out["reason"] == "max_loss"
    assert out["total_pnl"] == pytest.approx(-10.0)
    assert out["margin_ok"] is True
