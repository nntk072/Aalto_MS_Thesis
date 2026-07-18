"""Tests: DSR reward properties."""

from __future__ import annotations

import pytest

from quant_rl.envs.reward import DSRReward


def test_breach_returns_minus_one():
    r = DSRReward()
    val = r(0.0, breach=True)
    assert val == pytest.approx(-1.0)


def test_positive_pnl_eventually_positive():
    """After consistent positive returns DSR should be positive."""
    r = DSRReward(eta=0.1)
    rewards = [r(100.0, initial_balance=100_000.0) for _ in range(50)]
    assert rewards[-1] > 0.0


def test_negative_pnl_eventually_negative():
    r = DSRReward(eta=0.1)
    rewards = [r(-100.0, initial_balance=100_000.0) for _ in range(50)]
    assert rewards[-1] < 0.0


def test_soft_penalty_near_daily_limit():
    r = DSRReward(eta=0.01)
    # No penalty far from limit
    val_safe = r(0.0, daily_loss=100.0, daily_loss_limit=5000.0, initial_balance=100_000.0)
    r.reset()
    # Close to limit should be more penalised
    val_near = r(0.0, daily_loss=4999.0, daily_loss_limit=5000.0, initial_balance=100_000.0)
    assert val_near < val_safe


def test_reward_clipped():
    r = DSRReward(eta=0.5)
    # Extreme gains
    for _ in range(10):
        val = r(1_000_000.0, initial_balance=100_000.0)
    assert val <= 10.0
    assert val >= -10.0
