"""Tests for PO3Reward (Agent.md §21, §30)."""

from __future__ import annotations

import pytest

from quant_rl.envs.po3_reward import PO3Reward


def reward() -> PO3Reward:
    return PO3Reward(
        entry_bonus=1.0,
        manipulation_penalty=0.5,
        invalid_ifvg_penalty=0.5,
        distribution_bonus=0.5,
    )


def test_entry_inside_ifvg_after_distribution_is_a_bonus() -> None:
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        in_ifvg=True,
        manipulation_active=False,
        manipulation_end=True,
        distribution_phase=True,
    )
    # +entry_bonus +distribution_bonus = 1.5
    assert out == pytest.approx(1.5)


def test_entry_during_manipulation_is_a_penalty() -> None:
    r = reward()
    out = r(
        position_changed=True,
        direction=-1,
        in_ifvg=False,
        manipulation_active=True,
        manipulation_end=False,
        distribution_phase=False,
    )
    # -manipulation_penalty -invalid_ifvg_penalty = -1.0
    assert out == pytest.approx(-1.0)


def test_entry_outside_ifvg_is_a_penalty() -> None:
    r = reward()
    out = r(
        position_changed=True,
        direction=1,
        in_ifvg=False,
        manipulation_active=True,
        manipulation_end=True,
        distribution_phase=True,
    )
    # -invalid_ifvg_penalty +distribution_bonus = 0.0
    assert out == pytest.approx(0.0)


def test_no_position_transition_gives_no_entry_signal() -> None:
    r = reward()
    out = r(
        position_changed=False,
        direction=0,
        in_ifvg=False,
        manipulation_active=False,
        manipulation_end=False,
        distribution_phase=True,
    )
    assert out == 0.0


def test_holding_position_not_continuously_rewarded() -> None:
    r = reward()
    out = r(
        position_changed=False,
        direction=1,
        in_ifvg=True,
        manipulation_active=False,
        manipulation_end=True,
        distribution_phase=True,
    )
    assert out == 0.0


def test_reset_clears_state() -> None:
    r = reward()
    r(
        position_changed=True,
        direction=1,
        in_ifvg=True,
        manipulation_active=False,
        manipulation_end=True,
        distribution_phase=True,
    )
    r.reset()
    # reset() is a no-op for a stateless reward; must not raise.
    assert (
        r(
            position_changed=False,
            direction=0,
            in_ifvg=False,
            manipulation_active=False,
            manipulation_end=False,
            distribution_phase=False,
        )
        == 0.0
    )
