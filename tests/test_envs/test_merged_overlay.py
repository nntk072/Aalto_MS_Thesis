"""Overlay direction follows distribution, context, then higher-timeframe bias."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_rl.envs.po3_reward import PO3Reward
from quant_rl.envs.strategies.po3_ifvg import PO3IFVGStrategy


def test_sell_side_sweep_sets_the_long() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series({"sweep_low": 1.0, "htf_day_bias": -1.0})
    assert strategy.context_direction(row) == 1


def test_buy_side_sweep_on_a_higher_timeframe_sets_the_short() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series({"H1_sweep_high": 1.0, "ifvg_retest_bull": 1.0})
    assert strategy.context_direction(row) == -1


def test_protected_swing_sets_the_side_when_there_is_no_sweep() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series({"M15_swing_low_event": 1.0, "ny_manip_confirmed": 0.0})
    assert strategy.context_direction(row) == 1


def test_gap_retest_alone_does_not_set_the_side() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series(
        {
            "po3_distribution": 1.0,
            "po3_distribution_direction": -1.0,
            "ny_manip_confirmed": 1.0,
            "htf_day_bias": 1.0,
            "ifvg_retest_bear": 1.0,
            "M15_ifvg_retest_bear": 1.0,
            "H1_fvg_retest_bull": 1.0,
        }
    )
    assert strategy.context_direction(row) == 0


def test_both_sweeps_on_one_bar_hold() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series({"sweep_low": 1.0, "sweep_high": 1.0})
    assert strategy.context_direction(row) == 0


def test_stop_ladder_includes_the_swept_extreme() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series(
        {
            "last_swing_low": 10.0,
            "po3_manipulation_low": 11.0,
            "sweep_low_level": 12.0,
            "asian_low": 13.0,
            "london_low": 14.0,
        }
    )
    names = [name for name, _ in strategy.sl_candidates(direction=1, row=row)]
    assert names == [
        "last_swing_low",
        "po3_manipulation_low",
        "sweep_low_level",
        "asian_low",
        "london_low",
    ]


def test_inside_a_live_gap_does_not_set_the_side() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series(
        {
            "htf_day_bias": 1.0,
            "ny_manip_confirmed": 1.0,
            "price_in_ifvg_bull": 1.0,
            "H1_fvg_in_bull": 1.0,
        }
    )
    assert strategy.context_direction(row) == 0


def test_ifvg_inside_does_not_set_the_side() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False, require_price_retest=False)
    row = pd.Series(
        {
            "htf_day_bias": -1.0,
            "ny_manip_confirmed": 1.0,
            "M5_price_in_ifvg_bear": 1.0,
        }
    )
    assert strategy.context_direction(row) == 0


def test_fvg_retest_on_a_higher_timeframe_does_not_set_the_side() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False, require_price_retest=False)
    row = pd.Series(
        {
            "htf_day_bias": 1.0,
            "ny_manip_confirmed": 1.0,
            "H1_fvg_retest_bull": 1.0,
        }
    )
    assert strategy.context_direction(row) == 0


def test_stop_can_sit_under_the_gap_or_the_first_candle() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series(
        {
            "ifvg_bull_active": 1.0,
            "ifvg_bull_low": 9.0,
            "ifvg_bull_origin": 8.5,
            "M15_fvg_bull_active": 1.0,
            "M15_fvg_bull_low": 8.0,
            "M15_fvg_bull_origin": 7.5,
            "H1_fvg_bull_low": 6.0,
        }
    )
    names = [name for name, _ in strategy.sl_candidates(direction=1, row=row)]
    assert names == [
        "ifvg_bull_low",
        "ifvg_bull_origin",
        "M15_fvg_bull_low",
        "M15_fvg_bull_origin",
    ]


def test_sweep_penalty_only_without_sweep_or_gap() -> None:
    reward = PO3Reward(
        entry_bonus=1.0,
        manipulation_penalty=0.0,
        invalid_ifvg_penalty=0.0,
        distribution_bonus=0.0,
        sweep_penalty=0.5,
    )
    bare = reward(
        position_changed=True,
        direction=1,
        in_ifvg=False,
        manipulation_active=False,
        manipulation_end=False,
        distribution_phase=False,
        liquidity=False,
        in_gap=False,
    )
    swept = reward(
        position_changed=True,
        direction=1,
        in_ifvg=False,
        manipulation_active=False,
        manipulation_end=False,
        distribution_phase=False,
        liquidity=True,
        in_gap=True,
        entry_setup=True,
    )
    assert bare == pytest.approx(-0.5)
    assert swept == pytest.approx(1.0)
