"""Merged overlay: a sweep plus a same-side gap sets the trade direction."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_rl.envs.po3_reward import PO3Reward
from quant_rl.envs.strategies.po3_ifvg import PO3IFVGStrategy


def test_long_after_sellside_sweep_and_bullish_gap() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series({"sweep_low": 1.0, "price_in_ifvg_bull": 1.0})
    assert strategy.context_direction(row) == 1


def test_near_sweep_and_nearby_fvg_is_enough() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series({"manipulation_high_distance_atr": 0.4, "fvg_bear_dist": 0.8})
    assert strategy.context_direction(row) == -1


def test_gap_without_sweep_holds() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series({"price_in_ifvg_bull": 1.0, "htf_day_bias": 1.0})
    assert strategy.context_direction(row) == 0


def test_absent_fvg_cap_is_not_a_nearby_gap() -> None:
    strategy = PO3IFVGStrategy(enforce_gate=False)
    row = pd.Series({"sweep_low": 1.0, "fvg_bull_dist": 5.0})
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
    )
    assert bare == pytest.approx(-0.5)
    assert swept == pytest.approx(1.0)
