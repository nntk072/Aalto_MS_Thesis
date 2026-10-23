"""Structural stop menu, target filter, and EMA-21 exit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.reporting.geometry import geometry_summary
from features.technical.swings import recent_unswept_swing_candidates
from simulation.environment.env import TradingEnv
from simulation.environment.market_view import BarView, FeatureRow
from trading.execution.broker import Position
from trading.orders.entries import (
    OrderGeometry,
    place_strategy_order,
    select_index,
)
from trading.orders.stops import stop_menu
from trading.orders.targets import collapse_levels, target_menu
from trading.risk.sizing import compute_lots
from trading.strategies import BaselineStrategy


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
    row: pd.Series | FeatureRow,
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
    price_increment: float | None = None,
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
        price_increment=price_increment,
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


def test_levels_collapse_at_configured_instrument_precision() -> None:
    assert collapse_levels(
        [("first", 100.0), ("same_tick", 100.004), ("next_tick", 100.006)],
        price_increment=0.01,
    ) == [("first|same_tick", 100.0), ("next_tick", 100.006)]


def test_latest_only_stop_menu_keeps_legacy_deduplication_without_candidate_schema() -> None:
    order = _place(
        _row(),
        min_dist=1.0,
        stops=[("legacy_a", 90.0), ("legacy_b", 90.004)],
        price_increment=0.01,
    )

    assert {name for name, _ in order.sl_menu} == {"legacy_a", "legacy_b"}


@pytest.mark.parametrize(
    ("direction", "expected_stops", "expected_targets"),
    [
        (1, {"london_low", "asian_low"}, {"london_high", "asian_high"}),
        (-1, {"london_high", "asian_high"}, {"london_low", "asian_low"}),
    ],
)
def test_session_extremes_are_structural_order_candidates(
    direction: int, expected_stops: set[str], expected_targets: set[str]
) -> None:
    row = _row(london_low=95.0, asian_low=94.0, london_high=105.0, asian_high=106.0)

    menu, _audit = stop_menu(
        BaselineStrategy(),
        direction=direction,
        entry_price=100.0,
        row=row,
        min_dist=1.0,
        buffer=0.0,
        manipulation="none",
        htf=(),
    )

    stop_names = {part for name, _stop, _distance in menu for part in name.split("|")}
    assert expected_stops <= stop_names
    targets = target_menu(
        direction=direction,
        entry_price=100.0,
        row=row,
        stop_dist=1.0,
        max_tp_distance=float("nan"),
        htf=(),
    )
    target_levels = {part for name, _target, _distance in targets for part in name.split("|")}
    assert expected_targets <= target_levels


def test_mtf_swing_candidates_require_causal_provenance_and_are_audited() -> None:
    decision_at = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")
    confirmed_at = pd.Timestamp("2026-01-05 09:55:00", tz="UTC")
    pivot_at = pd.Timestamp("2026-01-05 09:45:00", tz="UTC")
    row = pd.Series(
        {
            "M5_swing_low_candidate_1": 95.0,
            "M5_swing_low_candidate_1_pivot_time_ms": pivot_at.value // 1_000_000,
            "M5_swing_low_candidate_1_confirmed_at_ms": confirmed_at.value // 1_000_000,
            "M5_swing_low_candidate_1_age_bars": 1.0,
        },
        name=decision_at,
    )

    order = _place(row, min_dist=1.0, price_increment=0.01)

    assert any(name == "M5_swing_low_candidate_1" for name, _ in order.sl_menu)
    candidate_audit = next(
        item for item in order.sl_candidate_audit if item["name"] == "M5_swing_low_candidate_1"
    )
    assert candidate_audit["status"] == "eligible"
    assert candidate_audit["provenance"] == [
        {
            "timeframe": "M5",
            "pivot_time_ms": pivot_at.value // 1_000_000,
            "confirmed_at_ms": confirmed_at.value // 1_000_000,
            "age_bars": 1.0,
        }
    ]


@pytest.mark.parametrize(("direction", "side"), [(1, "low"), (-1, "high")])
def test_final_mtf_swing_menu_exposes_and_selects_all_candidates_for_both_sides(
    direction: int, side: str
) -> None:
    decision_at = pd.Timestamp("2026-01-05 12:00:00", tz="UTC")
    timeframe_minutes = {"M5": 5, "M15": 15, "H1": 60}
    expected_distances: dict[str, float] = {}
    values: dict[str, float] = {"ema_21": 101.0}
    candidate_index = 0
    for timeframe in ("M5", "M15", "H1"):
        for slot in range(1, 4):
            candidate_index += 1
            name = f"{timeframe}_swing_{side}_candidate_{slot}"
            distance = 5.0 + 0.5 * (candidate_index - 1)
            level = 100.0 - distance if direction == 1 else 100.0 + distance
            confirmed_at = decision_at - pd.Timedelta(minutes=slot)
            pivot_at = confirmed_at - pd.Timedelta(minutes=timeframe_minutes[timeframe])
            values[name] = level
            values[f"{name}_pivot_time_ms"] = pivot_at.value // 1_000_000
            values[f"{name}_confirmed_at_ms"] = confirmed_at.value // 1_000_000
            values[f"{name}_age_bars"] = float(slot)
            expected_distances[name] = distance

    row = pd.Series(values, name=decision_at)
    menu = _place(
        row,
        direction=direction,
        min_dist=1.0,
        exit_mode="ema_21",
        price_increment=0.01,
    )
    assert {name for name, _price in menu.sl_menu} == set(expected_distances)
    assert all(
        item["status"] == "eligible" and item.get("provenance") for item in menu.sl_candidate_audit
    )

    # Every candidate lies in the reachable stop-coordinate range [nearest, 2x nearest].
    nearest_distance = min(expected_distances.values())
    for name, distance in expected_distances.items():
        selected = _place(
            row,
            direction=direction,
            sl_fraction=distance / nearest_distance,
            min_dist=1.0,
            exit_mode="ema_21",
            price_increment=0.01,
        )
        assert selected.sl_ref == name
        assert any(
            item["name"] == name and item["selected"] for item in selected.sl_candidate_audit
        )


def test_mtf_swing_candidate_with_future_confirmation_fails_closed() -> None:
    decision_at = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")
    future = pd.Timestamp("2026-01-05 10:05:00", tz="UTC")
    row = pd.Series(
        {
            "M5_swing_low_candidate_1": 95.0,
            "M5_swing_low_candidate_1_pivot_time_ms": future.value // 1_000_000,
            "M5_swing_low_candidate_1_confirmed_at_ms": future.value // 1_000_000,
            "M5_swing_low_candidate_1_age_bars": 0.0,
        },
        name=decision_at,
    )

    order = _place(row, min_dist=1.0)

    assert not any(name == "M5_swing_low_candidate_1" for name, _ in order.sl_menu)
    rejected = next(
        item for item in order.sl_candidate_audit if item["name"] == "M5_swing_low_candidate_1"
    )
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "unconfirmed_or_future_reference"


@pytest.mark.parametrize(
    ("price", "min_dist", "reason"),
    [(101.0, 1.0, "wrong_side"), (99.5, 1.0, "below_min_distance")],
)
def test_mtf_swing_candidate_rejects_invalid_long_stop_geometry(
    price: float, min_dist: float, reason: str
) -> None:
    decision_at = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")
    confirmed_at = pd.Timestamp("2026-01-05 09:55:00", tz="UTC")
    pivot_at = pd.Timestamp("2026-01-05 09:45:00", tz="UTC")
    name = "M5_swing_low_candidate_1"
    row = pd.Series(
        {
            name: price,
            f"{name}_pivot_time_ms": pivot_at.value // 1_000_000,
            f"{name}_confirmed_at_ms": confirmed_at.value // 1_000_000,
            f"{name}_age_bars": 1.0,
        },
        name=decision_at,
    )

    order = _place(row, min_dist=min_dist)

    assert not any(candidate == name for candidate, _ in order.sl_menu)
    rejected = next(item for item in order.sl_candidate_audit if item["name"] == name)
    assert rejected["reason"] == reason


def test_mtf_short_swing_candidate_and_ifvg_stops_coexist() -> None:
    decision_at = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")
    confirmed_at = pd.Timestamp("2026-01-05 09:55:00", tz="UTC")
    pivot_at = pd.Timestamp("2026-01-05 09:45:00", tz="UTC")
    name = "H1_swing_high_candidate_1"
    row = pd.Series(
        {
            name: 105.0,
            f"{name}_pivot_time_ms": pivot_at.value // 1_000_000,
            f"{name}_confirmed_at_ms": confirmed_at.value // 1_000_000,
            f"{name}_age_bars": 1.0,
        },
        name=decision_at,
    )

    order = _place(
        row,
        direction=-1,
        min_dist=1.0,
        stops=[("ifvg_bull_high", 104.0), ("po3_ifvg_through_zone_extreme", 106.0)],
    )

    stop_names = {candidate for candidate, _ in order.sl_menu}
    assert name in stop_names
    assert "ifvg_bull_high" in stop_names
    assert "po3_ifvg_through_zone_extreme" in stop_names


def test_recent_swing_window_drops_levels_displaced_by_three_newer_swings() -> None:
    idx = pd.date_range("2026-01-05 09:00", periods=6, freq="5min", tz="UTC")
    prices = [120.0, 115.0, 110.0, 105.0]
    bars = pd.DataFrame({"high": [100.0] * 6, "low": [99.0] * 6}, index=idx)
    swings = pd.DataFrame(
        {
            "swing_high_event": [False, True, True, True, True, False],
            "swing_low_event": [False] * len(idx),
            "swing_high_extreme": [np.nan, *prices, np.nan],
            "swing_low_extreme": [np.nan] * len(idx),
            "swing_high_location": [np.nan, 0.0, 1.0, 2.0, 3.0, np.nan],
            "swing_low_location": [np.nan] * len(idx),
        },
        index=idx,
    )

    result = recent_unswept_swing_candidates(swings, bars, count=3)

    assert result.loc[idx[4], "swing_high_candidate_1"] == 105.0
    assert result.loc[idx[4], "swing_high_candidate_2"] == 110.0
    assert result.loc[idx[4], "swing_high_candidate_3"] == 115.0
    assert 120.0 not in result.loc[idx[4]].to_numpy()


def test_feature_row_candidate_uses_carried_decision_timestamp() -> None:
    decision_ms = pd.Timestamp("2026-01-05 10:00:00", tz="UTC").value // 1_000_000
    confirmed_ms = decision_ms - 300_000
    pivot_ms = confirmed_ms - 600_000
    names = [
        "M5_swing_low_candidate_1",
        "M5_swing_low_candidate_1_pivot_time_ms",
        "M5_swing_low_candidate_1_confirmed_at_ms",
        "M5_swing_low_candidate_1_age_bars",
    ]
    row = FeatureRow(
        np.array([95.0, pivot_ms, confirmed_ms, 1.0]),
        {name: index for index, name in enumerate(names)},
        int(decision_ms),
    )

    order = _place(row, min_dist=1.0)

    assert any(name == names[0] for name, _ in order.sl_menu)


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
    env = TradingEnv(bars, feats, strategy_actions=True, allow_agent_sl_mode=False, obs_window=5)
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
    from trading.strategies import PO3IFVGStrategy

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


@pytest.mark.parametrize(
    ("direction", "zone_low", "zone_high", "through", "expected_edge", "expected_through"),
    [
        (1, 90.0, 95.0, 88.0, 89.0, 87.0),
        (-1, 105.0, 110.0, 112.0, 111.0, 113.0),
    ],
)
def test_revised_ifvg_stop_menu_includes_zone_edge_and_observed_excursion(
    direction: int,
    zone_low: float,
    zone_high: float,
    through: float,
    expected_edge: float,
    expected_through: float,
) -> None:
    from trading.strategies import PO3IFVGStrategy

    feat_row = _row(
        po3_ifvg_low=zone_low,
        po3_ifvg_high=zone_high,
        po3_ifvg_through_zone_extreme=through,
        prev_day_high=130.0,
        prev_day_low=70.0,
    )
    placed = place_strategy_order(
        PO3IFVGStrategy(
            enforce_gate=False,
            entry_mode="confirmed_first_eligible",
            require_price_retest=False,
        ),
        direction=direction,
        entry_price=100.0,
        feat_row=feat_row,
        min_dist=10.0,
        sl_fraction=0.0,
        tp_fraction=0.0,
        exit_mode="structural",
        buffer_pts=1.0,
        max_tp_distance=float("nan"),
    )

    assert ("ifvg_far_edge", expected_edge) in placed.sl_menu
    assert ("ifvg_through_zone_extreme", expected_through) in placed.sl_menu
    assert placed.sl_ref == "ifvg_far_edge"
    excursion_distance = abs(100.0 - expected_through)
    edge_distance = abs(100.0 - expected_edge)
    excursion_placed = place_strategy_order(
        PO3IFVGStrategy(
            enforce_gate=False,
            entry_mode="confirmed_first_eligible",
            require_price_retest=False,
        ),
        direction=direction,
        entry_price=100.0,
        feat_row=feat_row,
        min_dist=10.0,
        sl_fraction=excursion_distance / edge_distance,
        tp_fraction=0.0,
        exit_mode="structural",
        buffer_pts=1.0,
        max_tp_distance=float("nan"),
    )
    assert excursion_placed.sl_ref == "ifvg_through_zone_extreme"
    assert {item["name"] for item in placed.sl_candidate_audit} >= {
        "ifvg_far_edge",
        "ifvg_through_zone_extreme",
    }


@pytest.mark.parametrize(
    ("direction", "zone_low", "zone_high", "through"),
    [
        (1, 95.0, 97.0, 92.0),
        (-1, 103.0, 105.0, 108.0),
    ],
)
def test_revised_ifvg_stop_audit_records_minimum_distance_rejections(
    direction: int, zone_low: float, zone_high: float, through: float
) -> None:
    from trading.strategies import PO3IFVGStrategy

    placed = place_strategy_order(
        PO3IFVGStrategy(
            enforce_gate=False,
            entry_mode="confirmed_first_eligible",
            require_price_retest=False,
        ),
        direction=direction,
        entry_price=100.0,
        feat_row=_row(
            po3_ifvg_low=zone_low,
            po3_ifvg_high=zone_high,
            po3_ifvg_through_zone_extreme=through,
            prev_day_high=130.0,
        ),
        min_dist=10.0,
        sl_fraction=0.0,
        tp_fraction=0.0,
        exit_mode="structural",
        buffer_pts=1.0,
        max_tp_distance=float("nan"),
    )

    audit = {item["name"]: item for item in placed.sl_candidate_audit}
    assert audit["ifvg_far_edge"]["reason"] == "below_min_distance"
    assert audit["ifvg_through_zone_extreme"]["reason"] == "below_min_distance"


@pytest.mark.parametrize(
    ("targets", "expected_target", "expected_rr", "expected_fallback"),
    [
        ([112.0, 116.0], "prev_day_high", 1.6, None),
        ([112.0, 114.0], "asian_high", 1.2, "no_structural_reference_within_rr_window"),
    ],
)
def test_tp_prefers_rr_window_and_logs_nearest_admissible_fallback(
    targets: list[float],
    expected_target: str,
    expected_rr: float,
    expected_fallback: str | None,
) -> None:
    placed = place_strategy_order(
        _Menu([("last_swing_low", 90.0)]),
        direction=1,
        entry_price=100.0,
        feat_row=_row(
            last_swing_low=90.0,
            prev_day_high=max(targets),
            asian_high=min(targets),
        ),
        min_dist=10.0,
        sl_fraction=0.0,
        tp_fraction=0.0,
        exit_mode="structural",
        buffer_pts=0.0,
        max_tp_distance=float("nan"),
        rr_bounds=(1.5, 5.0),
    )

    assert placed.tp_ref == expected_target
    assert placed.planned_rr == pytest.approx(expected_rr)
    assert placed.tp_references
    assert {item.fallback_reason for item in placed.tp_references} == {expected_fallback}
