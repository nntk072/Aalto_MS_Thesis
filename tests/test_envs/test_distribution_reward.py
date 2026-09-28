"""Tests for DistributionReward and the Idea 2 respect gate."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_rl.envs.distribution_reward import DistributionReward
from quant_rl.envs.strategies.distribution import DistributionStrategy


def reward() -> DistributionReward:
    return DistributionReward(
        entry_bonus=1.0,
        sweep_penalty=0.5,
        distribution_bonus=0.5,
    )


def test_consistent_long_chain_is_rewarded() -> None:
    """A distribution bar after the opposing gap failed is the long entry."""
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        distribution_after_ifvg=True,
        distribution_phase=True,
    )
    # +entry_bonus +distribution_bonus = 1.5
    assert out == pytest.approx(1.5)


def test_consistent_short_chain_is_rewarded() -> None:
    """A distribution bar after the opposing gap failed is the short entry."""
    r = reward()
    out = r(
        position_changed=True,
        direction=-1,
        distribution_after_ifvg=True,
        distribution_phase=False,
    )
    assert out == pytest.approx(1.0)


def test_inconsistent_mapping_is_penalised() -> None:
    """An entry without a respected IFVG is not the distribution chain."""
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        distribution_after_ifvg=False,
        distribution_phase=True,
    )
    # -sweep_penalty +distribution_bonus = 0.0
    assert out == pytest.approx(0.0)


def test_no_context_is_penalised() -> None:
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        distribution_after_ifvg=False,
        distribution_phase=False,
    )
    assert out == pytest.approx(-0.5)


def test_no_position_transition_gives_no_signal() -> None:
    r = reward()
    out = r(
        position_changed=False,
        direction=0,
        distribution_after_ifvg=True,
        distribution_phase=True,
    )
    assert out == 0.0


def test_reset_is_noop() -> None:
    r = reward()
    r(
        position_changed=True,
        direction=1,
        distribution_after_ifvg=True,
        distribution_phase=True,
    )
    r.reset()
    assert (
        r(
            position_changed=False,
            direction=0,
            distribution_after_ifvg=False,
            distribution_phase=False,
        )
        == 0.0
    )


def _long_leg(**extra: float) -> pd.Series:
    row = {
        "po3_distribution": 1.0,
        "po3_distribution_direction": 1.0,
        "htf_day_bias": 1.0,
        "po3_distribution_after_ifvg": 1.0,
    }
    row.update(extra)
    return pd.Series(row)


def test_distribution_after_the_gap_fails_sets_the_side() -> None:
    strategy = DistributionStrategy(enforce_gate=False)
    assert strategy.context_direction(_long_leg(bos_up=1.0)) == 1
    short_row = pd.Series(
        {
            "po3_distribution": 1.0,
            "po3_distribution_direction": -1.0,
            "htf_day_bias": -1.0,
            "po3_distribution_after_ifvg": 1.0,
        }
    )
    assert strategy.context_direction(short_row) == -1


def test_failure_bar_and_a_bias_mismatch_hold() -> None:
    strategy = DistributionStrategy(enforce_gate=True)
    before = _long_leg(po3_distribution_after_ifvg=0.0, bos_up=1.0)
    assert strategy.context_direction(before) == 0
    assert strategy.validate_entry(direction=1, row=before) is False
    against = _long_leg(htf_day_bias=-1.0)
    assert strategy.context_direction(against) == 0


def test_stop_uses_a_respected_lower_gap_and_an_aligned_higher_one() -> None:
    strategy = DistributionStrategy(enforce_gate=False)
    row = pd.Series(
        {
            "last_swing_low": 10.0,
            "sweep_low_level": 11.0,
            "asian_low": 12.0,
            "london_low": 13.0,
            "ifvg_bull_active": 1.0,
            "ifvg_bull_low": 9.0,
            "ifvg_bull_origin": 8.5,
            "M15_ifvg_bull_active": 0.0,
            "M15_ifvg_bull_low": 5.0,
            "M15_ifvg_bull_origin": 4.5,
            "H1_ifvg_bull_active": 1.0,
            "H1_ifvg_bull_low": 7.0,
            "H1_ifvg_bull_origin": 6.5,
        }
    )
    names = [name for name, _ in strategy.sl_candidates(direction=1, row=row)]
    assert names == [
        "last_swing_low",
        "sweep_low_level",
        "asian_low",
        "london_low",
        "ifvg_bull_low",
        "ifvg_bull_origin",
        "H1_ifvg_bull_low",
        "H1_ifvg_bull_origin",
    ]
