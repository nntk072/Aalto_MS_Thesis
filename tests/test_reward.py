"""Tests: DSR reward properties."""

from __future__ import annotations

import pytest

from quant_rl.envs.reward import DSRReward


def test_breach_returns_strong_terminal():
    r = DSRReward()
    val = r(0.0, breach=True)
    assert val == pytest.approx(-10.0)


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
    val_safe = r(0.0, daily_loss=100.0, daily_loss_limit=5000.0, initial_balance=100_000.0)
    r.reset()
    val_near = r(0.0, daily_loss=4999.0, daily_loss_limit=5000.0, initial_balance=100_000.0)
    assert val_near < val_safe


def test_soft_penalty_uses_soft_brick_band():
    r = DSRReward(eta=0.01)
    val_below = r(
        0.0,
        daily_loss=1500.0,
        daily_loss_limit=5000.0,
        soft_daily_loss_limit=2000.0,
        initial_balance=100_000.0,
    )
    r.reset()
    val_in_band = r(
        0.0,
        daily_loss=3500.0,
        daily_loss_limit=5000.0,
        soft_daily_loss_limit=2000.0,
        initial_balance=100_000.0,
    )
    assert val_in_band < val_below


def test_soft_max_loss_shaping_band():
    r = DSRReward(eta=0.01)
    val_safe = r(
        0.0,
        loss_from_initial=1000.0,
        soft_max_loss_limit=5000.0,
        max_loss_limit=10000.0,
        initial_balance=100_000.0,
    )
    r.reset()
    val_near = r(
        0.0,
        loss_from_initial=8000.0,
        soft_max_loss_limit=5000.0,
        max_loss_limit=10000.0,
        initial_balance=100_000.0,
    )
    assert val_near < val_safe


def test_soft_trailing_dd_shaping_band():
    r = DSRReward(eta=0.01)
    val_safe = r(
        0.0,
        trailing_dd=0.02,
        soft_trailing_dd_limit=0.04,
        trailing_dd_limit=0.07,
        initial_balance=100_000.0,
    )
    r.reset()
    val_near = r(
        0.0,
        trailing_dd=0.06,
        soft_trailing_dd_limit=0.04,
        trailing_dd_limit=0.07,
        initial_balance=100_000.0,
    )
    assert val_near < val_safe


def test_reward_clipped():
    r = DSRReward(eta=0.5)
    # Extreme gains
    for _ in range(10):
        val = r(1_000_000.0, initial_balance=100_000.0)
    assert val <= 10.0
    assert val >= -10.0
