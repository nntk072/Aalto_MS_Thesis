"""Accounting contracts for dense fixed-risk incremental-R reward."""

from __future__ import annotations

import numpy as np
import pytest

from quant_rl.envs.reward import EquityDeltaReward


def test_equity_delta_reward_telescopes_to_total_equity_change() -> None:
    equity = np.asarray([100_000.0, 100_125.0, 99_950.0, 100_400.0])
    reward = EquityDeltaReward(500.0)
    rewards = [reward(float(right - left)) for left, right in zip(equity[:-1], equity[1:])]
    assert sum(rewards) == pytest.approx((equity[-1] - equity[0]) / 500.0)


def test_equity_delta_reward_records_dense_mark_to_market_credit() -> None:
    reward = EquityDeltaReward(500.0)
    assert reward(100.0) == pytest.approx(0.2)
    assert reward.last_parts["pnl"] == pytest.approx(0.2)


def test_equity_delta_reward_keeps_breach_terminal_penalty() -> None:
    reward = EquityDeltaReward(500.0, breach_penalty=-10.0)
    assert reward(500.0, breach=True) == -10.0
    assert reward.last_parts["breach"] == -10.0


def test_equity_delta_reward_requires_positive_fixed_risk() -> None:
    with pytest.raises(ValueError, match="fixed_risk_usd"):
        EquityDeltaReward(0.0)
