"""Structural stop menu, target filter, and EMA-21 exit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.backtest.broker import Position
from quant_rl.backtest.entry_levels import (
    OrderGeometry,
    collapse_levels,
    geometry_summary,
    place_strategy_order,
    select_index,
)
from quant_rl.backtest.risk import compute_lots
from quant_rl.envs.feature_row import BarView, FeatureRow
from quant_rl.envs.trading_env import TradingEnv


class _Menu:
    def __init__(self, stops: list[tuple[str, float]] | None = None) -> None:
        self._stops = stops

    def sl_candidates(self, *, direction: int, row: pd.Series) -> list[tuple[str, float]]:
        if self._stops is not None:
            return list(self._stops)
        if direction == 1:
            names = ("last_swing_low", "asian_low")
        else:
            names = ("last_swing_high", "asian_high")
        out: list[tuple[str, float]] = []
        for name in names:
            if name not in row.index or not np.isfinite(float(row[name])):
                continue
            out.append((name, float(row[name])))
        return out

    def sl_reference(self, *, direction: int, row: FeatureRow | pd.Series) -> float | None:
        del direction, row
        return None


def _row(**values: float) -> pd.Series:
    return pd.Series(values)


def _place(
    row: pd.Series,
    *,
    direction: int = 1,
    entry: float = 100.0,
    sl_fraction: float = 0.0,
    tp_fraction: float = 0.0,
    exit_mode: str = "structural",
    min_dist: float = 1.0,
    buffer_pts: float = 0.0,
    max_tp: float = float("nan"),
    stops: list[tuple[str, float]] | None = None,
) -> OrderGeometry:
    return place_strategy_order(
        _Menu(stops),
        direction=direction,
        entry_price=entry,
        feat_row=row,
        min_dist=min_dist,
        sl_fraction=sl_fraction,
        tp_fraction=tp_fraction,
        exit_mode=exit_mode,
        buffer_pts=buffer_pts,
        max_tp_distance=max_tp,
    )


def test_select_index_uses_truncation() -> None:
    assert select_index(0.5, 2) == 1
    assert select_index(0.0, 3) == 0
    assert select_index(0.34, 3) == 1
    assert select_index(0.67, 3) == 2


def test_equal_prices_collapse_and_distinct_prices_stay() -> None:
    one = collapse_levels(
        [
            ("last_swing_high", 100.0),
            ("london_high", 100.0 + 1e-5),
            ("prev_day_high", 100.0),
            ("sweep_high_level", 100.00005),
        ]
    )
    assert len(one) == 1
    assert one[0][0] == "last_swing_high|london_high|prev_day_high|sweep_high_level"
    two = collapse_levels([("a", 100.0), ("b", 110.0)])
    assert len(two) == 2


def test_long_and_short_menus_share_distances() -> None:
    row = _row(
        last_swing_low=98.0,
        asian_low=97.0,
        last_swing_high=102.0,
        asian_high=103.0,
        prev_day_high=110.0,
        prev_day_low=90.0,
    )
    long = _place(row, direction=1, sl_fraction=0.5, min_dist=0.5)
    short = _place(row, direction=-1, sl_fraction=0.5, min_dist=0.5)
    assert long.sl_price is not None and short.sl_price is not None
    assert long.sl_index == 0
    assert short.sl_index == 0
    assert abs(100.0 - long.sl_price) == pytest.approx(abs(short.sl_price - 100.0))
    assert long.planned_rr == pytest.approx(short.planned_rr)


def test_exit_mode_split_at_one_half() -> None:
    row = _row(last_swing_low=90.0, prev_day_high=120.0, ema_21=101.0)
    n = 20
    idx = pd.date_range("2020-01-01 16:30", periods=n, freq="1min")
    bars = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000),
            "session_id": np.zeros(n, dtype=int),
            "session": "ny",
        },
        index=idx,
    )
    feats = pd.DataFrame({"ema_21": np.full(n, 101.0)}, index=idx)
    env = TradingEnv(bars, feats, strategy_actions=True, obs_window=5)
    env._decode_action(np.array([0.0, 0.0, 0.0, 0.0, -0.02], dtype=np.float32), feats.iloc[0])
    assert env._selected_exit_mode == "structural"
    env._decode_action(np.array([0.0, 0.0, 0.0, 0.0, 0.02], dtype=np.float32), feats.iloc[0])
    assert env._selected_exit_mode == "ema_21"
    structural = _place(row, exit_mode="structural")
    ema = _place(row, exit_mode="ema_21")
    assert structural.exit_mode == "structural"
    assert structural.planned_rr is not None and structural.planned_rr >= 1.0
    assert ema.exit_mode == "ema_21"
    assert ema.tp_price is None
    assert ema.planned_rr is None
    assert ema.rejected is False
    missing_ema = _place(
        _row(last_swing_low=90.0, prev_day_high=120.0),
        exit_mode="ema_21",
    )
    assert missing_ema.rejected is True
    assert missing_ema.sl_price is None


def test_buffer_manipulation_and_reach_filters() -> None:
    crossed = _place(
        _row(last_swing_low=100.4),
        entry=100.0,
        buffer_pts=0.5,
        min_dist=0.01,
    )
    assert crossed.rejected is True
    assert crossed.sl_price is None

    against = _place(
        _row(po3_manipulation_active=1.0, po3_manipulation_end=0.0, prev_day_high=120.0),
        stops=[("ifvg_bull_low", 90.0), ("last_swing_low", 95.0)],
        min_dist=1.0,
    )
    assert against.sl_ref == "last_swing_low"
    assert against.rejected is False

    inside = _place(
        _row(prev_day_high=105.0),
        stops=[("last_swing_low", 80.0)],
        min_dist=1.0,
    )
    assert inside.rejected is True
    assert inside.reject_kind == "tp"

    nan_room = _place(
        _row(prev_day_high=130.0),
        stops=[("last_swing_low", 90.0)],
        max_tp=float("nan"),
    )
    assert nan_room.tp_ref == "prev_day_high"
    assert nan_room.planned_rr == pytest.approx(3.0)

    short_room = _place(
        _row(prev_day_high=140.0),
        stops=[("last_swing_low", 90.0)],
        max_tp=15.0,
    )
    assert short_room.rejected is True
    assert short_room.tp_price is None


def test_wider_stop_returns_fewer_lots() -> None:
    equity = 100_000.0
    risk = 0.01
    tight = compute_lots(
        equity, risk, 100.0, 98.0, contract_size=1.0, min_lot=0.01, max_lot=10_000.0
    )
    wide = compute_lots(
        equity, risk, 100.0, 90.0, contract_size=1.0, min_lot=0.01, max_lot=10_000.0
    )
    assert wide < tight


def test_stop_wins_when_the_same_bar_crosses_ema() -> None:
    n = 20
    idx = pd.date_range("2020-01-01 16:30", periods=n, freq="1min")
    bars = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000),
            "session_id": np.zeros(n, dtype=int),
            "session": "ny",
        },
        index=idx,
    )
    feats = pd.DataFrame({"ema_21": np.full(n, 101.0)}, index=idx)
    env = TradingEnv(bars, feats, strategy_actions=True, obs_window=5)
    env.position = Position(
        direction=1,
        size=1.0,
        entry_price=100.0,
        margin_used=0.0,
        sl_price=99.0,
        tp_price=None,
        sl_initial_price=99.0,
        exit_mode="ema_21",
        stop_distance=1.0,
        sl_ref="last_swing_low",
        tp_ref="ema_21",
    )
    env.step_idx = 5
    env._check_sl_tp(BarView(100.0, 102.0, 98.5, 99.5), 100.0, 100.0, bar_idx=5)
    assert env.position is None
    assert env.trade_log[-1]["reason"] == "structure_sl"
    assert env.trade_log[-1]["price"] == pytest.approx(99.0)


def test_geometry_summary_blanks_ema_planned_ratio() -> None:
    trades = pd.DataFrame(
        [
            {
                "type": "open",
                "sl_ref": "last_swing_low",
                "tp_ref": "prev_day_high",
                "planned_rr": 2.5,
                "realized_r": 1.8,
                "exit_mode": "structural",
            },
            {
                "type": "open",
                "sl_ref": "last_swing_low",
                "tp_ref": "ema_21",
                "planned_rr": np.nan,
                "realized_r": 3.4,
                "exit_mode": "ema_21",
            },
        ]
    )
    summary = geometry_summary(trades)
    structural = summary.loc[summary["tp_ref"].eq("prev_day_high")].iloc[0]
    ema = summary.loc[summary["tp_ref"].eq("ema_21")].iloc[0]
    assert structural["n"] == 1
    assert structural["planned_rr_median"] == pytest.approx(2.5)
    assert ema["n"] == 1
    assert pd.isna(ema["planned_rr_median"])
    assert ema["realized_r_median"] == pytest.approx(3.4)


def test_confirmed_manipulation_keeps_the_extreme_and_a_far_target() -> None:
    from quant_rl.envs.strategies import PO3IFVGStrategy

    placed = place_strategy_order(
        PO3IFVGStrategy(enforce_gate=False),
        direction=1,
        entry_price=100.0,
        feat_row=_row(
            po3_manipulation_end=1.0,
            po3_manipulation_active=0.0,
            po3_manipulation_low=95.0,
            ctx_prev_week_high=180.0,
        ),
        min_dist=1.0,
        sl_fraction=0.0,
        tp_fraction=0.0,
        exit_mode="structural",
        buffer_pts=0.0,
        max_tp_distance=float("nan"),
    )
    assert placed.rejected is False
    assert placed.sl_ref == "po3_manipulation_low"
    assert placed.tp_ref == "ctx_prev_week_high"
    assert placed.planned_rr is not None and placed.planned_rr > 1.0


def test_end_bar_restores_gap_stops_and_a_short_leg_drops_them() -> None:
    done = _place(
        _row(po3_manipulation_end=1.0, po3_manipulation_active=0.0, prev_day_high=120.0),
        stops=[("ifvg_bull_low", 90.0)],
        min_dist=1.0,
    )
    assert done.sl_ref == "ifvg_bull_low"
    short = _place(
        _row(po3_manipulation_active=1.0, po3_manipulation_end=0.0, prev_day_low=50.0),
        direction=-1,
        stops=[("fvg_bear_high", 110.0), ("last_swing_high", 105.0)],
        min_dist=1.0,
    )
    assert short.sl_ref == "last_swing_high"
    assert "fvg" not in short.sl_ref
