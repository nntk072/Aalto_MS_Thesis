"""Dashboard aggregation, reward split, and the eval block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.envs.reward import DSRReward, PnLReward
from quant_rl.evaluation.metrics import calculate_metrics
from quant_rl.train.dashboard import (
    action_stats,
    close_shares,
    format_dashboard,
    policy_from_logger,
    position_mix,
    sum_parts,
)

pytestmark = pytest.mark.unit


def test_pnl_parts_match_the_returned_reward() -> None:
    reward_fn = PnLReward(dsr_weight=0.0)
    out = reward_fn(0.0, realized_close_pnl=100.0, initial_balance=100_000.0, breach=False)
    assert out == pytest.approx(0.001)
    assert reward_fn.last_parts["pnl"] == pytest.approx(0.001)
    assert sum(reward_fn.last_parts.values()) == pytest.approx(out)


def test_breach_is_its_own_reward_part() -> None:
    reward_fn = DSRReward()
    assert reward_fn(0.0, breach=True) == -10.0
    assert reward_fn.last_parts["breach"] == -10.0
    assert sum(reward_fn.last_parts.values()) == pytest.approx(-10.0)


def test_position_mix_and_close_shares() -> None:
    mix = position_mix([1, 1, -1, 0])
    assert mix["long"] == pytest.approx(0.5)
    assert mix["short"] == pytest.approx(0.25)
    assert mix["flat"] == pytest.approx(0.25)
    shares = close_shares(["structure_sl", "structure_tp", "structure_sl"])
    assert shares["structure_sl"] == pytest.approx(2 / 3)
    assert shares["structure_tp"] == pytest.approx(1 / 3)
    assert shares["daily_loss"] == 0.0


def test_action_stats_and_part_sums() -> None:
    stats = action_stats([1.0, -1.0, 1.0, -1.0])
    assert stats["mean"] == pytest.approx(0.0)
    assert stats["std"] == pytest.approx(1.0)
    total = sum_parts([{"pnl": 0.1, "dsr": 0.2}, {"pnl": 0.3}])
    assert total["pnl"] == pytest.approx(0.4)
    assert total["dsr"] == pytest.approx(0.2)
    assert total["breach"] == 0.0


def test_eval_block_prints_metrics_from_a_short_equity_curve() -> None:
    equity = pd.Series([100_000.0, 100_050.0, 100_020.0, 100_200.0])
    trades = pd.DataFrame({"pnl": [50.0, -30.0, 180.0]})
    metrics = calculate_metrics(equity, trades=trades, n_sessions=1, n_breach_sessions=0)
    text = format_dashboard(
        {
            "timesteps": 100_000,
            "total_timesteps": 20_000_000,
            "train_reward_mean": 0.01,
            "reward_parts": sum_parts([{"pnl": 0.002, "soft_daily": -0.001}]),
            "position": position_mix([1, 0, -1]),
            "action": action_stats([0.2, -0.1, 0.0]),
            "closes": close_shares(["structure_tp"]),
            "n_closes": 1,
            "critic": {"ret_mean": 0.1, "val_mean": 0.0, "adv_mean": 0.1, "adv_std": 0.2},
            "policy": policy_from_logger(
                {"train/explained_variance": 0.0, "train/entropy_loss": -4.0, "train/std": 0.67},
                grad_norm=0.82,
            ),
            "eval": {
                "return_pct": float(metrics.total_return_pct),
                "sharpe": float(metrics.sharpe),
                "sortino": float(metrics.sortino),
                "max_drawdown": float(metrics.max_drawdown),
                "profit_factor": float(metrics.profit_factor),
                "win_rate": float(metrics.win_rate),
                "avg_trade": float(metrics.avg_trade),
                "n_trades": int(metrics.n_trades),
                "turnover": float(metrics.turnover),
                "reward_mean": 0.004,
            },
        }
    )
    assert "EVAL" in text
    assert "Sharpe" in text
    assert "explained var" in text
    assert "0.0000" in text
    assert "grad norm         0.820" in text
    assert metrics.n_trades == 3
    assert np.isfinite(metrics.sharpe)


def test_step_info_carries_reward_parts_and_direction() -> None:
    from tests.test_envs.test_ny_steps_eod import _make_env

    env = _make_env(max_episode_steps=2)
    env.reset()
    _obs, _reward, _done, _truncated, info = env.step(0)
    assert info["position_direction"] == 0
    assert "pnl" in info["reward_parts"]
    assert "dsr" in info["reward_parts"]
    assert info["close_reason"] == ""
