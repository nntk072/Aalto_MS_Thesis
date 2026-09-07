"""Tests for DistributionReward (Agent.md §22, §30)."""

from __future__ import annotations

import pytest

from quant_rl.envs.distribution_reward import DistributionReward


def reward() -> DistributionReward:
    return DistributionReward(
        entry_bonus=1.0,
        sweep_penalty=0.5,
        distribution_bonus=0.5,
    )


def test_consistent_long_chain_is_rewarded() -> None:
    """long + sweep_low + BOS_up -> positive alignment (Agent.md §30)."""
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        sweep_high=False,
        sweep_low=True,
        bos_up=True,
        bos_down=False,
        distribution_phase=True,
    )
    # +entry_bonus +distribution_bonus = 1.5
    assert out == pytest.approx(1.5)


def test_consistent_short_chain_is_rewarded() -> None:
    """short + sweep_high + BOS_down -> positive alignment (Agent.md §30)."""
    r = reward()
    out = r(
        position_changed=True,
        direction=-1,
        sweep_high=True,
        sweep_low=False,
        bos_up=False,
        bos_down=True,
        distribution_phase=False,
    )
    assert out == pytest.approx(1.0)


def test_inconsistent_mapping_is_penalised() -> None:
    """long + sweep_high (wrong side) must NOT be rewarded."""
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        sweep_high=True,
        sweep_low=False,
        bos_up=True,
        bos_down=False,
        distribution_phase=True,
    )
    # -sweep_penalty +distribution_bonus = 0.0
    assert out == pytest.approx(0.0)


def test_no_context_is_penalised() -> None:
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        sweep_high=False,
        sweep_low=False,
        bos_up=False,
        bos_down=False,
        distribution_phase=False,
    )
    assert out == pytest.approx(-0.5)


def test_no_position_transition_gives_no_signal() -> None:
    r = reward()
    out = r(
        position_changed=False,
        direction=0,
        sweep_high=False,
        sweep_low=True,
        bos_up=True,
        bos_down=False,
        distribution_phase=True,
    )
    assert out == 0.0


def test_reset_is_noop() -> None:
    r = reward()
    r(
        position_changed=True,
        direction=1,
        sweep_high=False,
        sweep_low=True,
        bos_up=True,
        bos_down=False,
        distribution_phase=True,
    )
    r.reset()
    assert (
        r(
            position_changed=False,
            direction=0,
            sweep_high=False,
            sweep_low=False,
            bos_up=False,
            bos_down=False,
            distribution_phase=False,
        )
        == 0.0
    )
