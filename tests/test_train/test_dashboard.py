"""Dashboard aggregation, reward split, and the eval block."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_rl.envs.reward import REWARD_PART_KEYS, DSRReward, PnLReward
from quant_rl.evaluation.metrics import calculate_metrics
from quant_rl.train.dashboard import (
    TrainingDashboardCallback,
    action_stats,
    close_shares,
    format_dashboard,
    format_dashboard_table,
    policy_from_logger,
    position_mix,
    ppo_log_interval,
    snapshot_row,
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
    assert reward_fn(0.0, breach=True) == -1.0
    assert reward_fn.last_parts["breach"] == -1.0
    assert sum(reward_fn.last_parts.values()) == pytest.approx(-1.0)


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
    assert "eval      " in text
    assert "sharpe" in text
    assert "expl" in text
    assert "+0.0000" in text
    assert "grad 0.820" in text
    assert metrics.n_trades == 3
    assert np.isfinite(metrics.sharpe)


def _sample_snapshot(*, with_eval: bool) -> dict[str, Any]:
    snap = {
        "timesteps": 8192,
        "total_timesteps": 20_000_000,
        "train_reward_mean": 0.01,
        "reward_parts": sum_parts([{"pnl": 0.002, "dsr": -0.1}]),
        "position": position_mix([1, 0, -1]),
        "action": action_stats([0.2, -0.1, 0.0]),
        "closes": close_shares(["structure_sl", "structure_tp"]),
        "n_closes": 2,
        "critic": {"ret_mean": 0.1, "val_mean": 0.0, "adv_mean": 0.1, "adv_std": 0.2},
        "policy": policy_from_logger({"train/std": 0.9}, grad_norm=1.25),
        "eval": None,
    }
    if with_eval:
        snap["eval"] = {
            "return_pct": 1.5,
            "sharpe": 0.8,
            "sortino": 1.1,
            "max_drawdown": 0.04,
            "profit_factor": 1.2,
            "win_rate": 0.5,
            "avg_trade": 12.0,
            "n_trades": 4,
            "turnover": 0.01,
            "reward_mean": 0.002,
        }
    return snap


def test_ppo_log_interval_matches_dashboard_cadence() -> None:
    assert ppo_log_interval(n_steps=64, n_envs=128, log_every=100_000) == 12
    assert ppo_log_interval(n_steps=64, n_envs=128, log_every=0) == 1


def test_dashboard_log_waits_for_the_interval() -> None:
    cb = TrainingDashboardCallback(total_timesteps=1_000_000, log_every=100_000)
    cb.model = type("M", (), {"num_timesteps": 8192})()
    assert cb._should_log()
    cb._advance_log()
    assert cb._next_log == 100_000
    cb.model.num_timesteps = 16_384
    assert not cb._should_log()
    cb._eval_just_ran = True
    assert cb._should_log()


def test_snapshot_row_keeps_behavior_and_omits_eval_until_recorded() -> None:
    row = snapshot_row(_sample_snapshot(with_eval=False))
    for key in REWARD_PART_KEYS:
        assert key in row
    assert row["action_mean"] == pytest.approx(float(np.mean([0.2, -0.1, 0.0])))
    assert row["long"] == pytest.approx(1 / 3)
    assert row["structure_sl"] == pytest.approx(0.5)
    assert row["grad_norm"] == pytest.approx(1.25)
    assert "return_pct" not in row
    assert "eval_reward" not in row

    recorded = snapshot_row(_sample_snapshot(with_eval=True), record_eval=True)
    assert recorded["return_pct"] == pytest.approx(1.5)
    assert recorded["eval_max_drawdown"] == pytest.approx(0.04)
    assert recorded["eval_reward"] == pytest.approx(0.002)
    assert recorded["max_drawdown"] == 0.0


def _prior_book(n: int, **cols: object) -> list[dict[str, object]]:
    row: dict[str, object] = {
        "sl_ref": "swing_low",
        "tp_ref": "swing_high",
        "exit_mode": "structural",
        "direction": 1,
        "stop_u": 0.2,
        "target_u": 0.3,
        "n_sl": 3,
        "n_tp": 3,
        "sl_index": 1,
        "tp_index": 1,
    }
    row.update(cols)
    rows: list[dict[str, object]] = []
    for i in range(n):
        item: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                item[key] = value[i]
            else:
                item[key] = value
        rows.append(item)
    return rows


def test_training_interval_sets_the_prior_and_eval_does_not() -> None:
    class _Policy:
        def __init__(self) -> None:
            self.names: set[str] = {"stale"}

        def set_active_prior(self, names: set[str]) -> None:
            self.names = set(names)

    class _Model:
        def __init__(self) -> None:
            self.policy = _Policy()
            self.action_space = type("Space", (), {"shape": (6,)})()
            self.num_timesteps = 50_000
            self.rollout_buffer = None

    cb = TrainingDashboardCallback(total_timesteps=1_000_000, log_every=100_000)
    cb.model = _Model()  # type: ignore[assignment]
    cb._prior_opens = _prior_book(10, n_sl=5, sl_index=4, stop_u=1.0)
    cb._apply_prior_interval()
    assert cb.model.policy.names == {"stale"}

    cb.num_timesteps = 100_000
    cb.model.num_timesteps = 100_000
    cb._apply_prior_interval()
    assert cb.model.policy.names == set()
    assert cb._prior_opens == []

    spread = np.linspace(0.0, 1.0, 40).tolist()
    cb._prior_opens = _prior_book(40, n_sl=5, sl_index=4, stop_u=spread, target_u=spread)
    cb.num_timesteps = 200_000
    cb.model.num_timesteps = 200_000
    cb._apply_prior_interval()
    assert cb.model.policy.names == {"stop"}
    row = snapshot_row(cb._snapshot())
    assert row["prior_active_stop"] == 1
    assert row["prior_active_target"] == 0
    assert row["prior_active_exit"] == 0
    assert row["prior_active_direction"] == 0
    assert row["sl_rank_median"] == pytest.approx(1.0)

    cb._eval_fn = lambda: {"diversity_text": "eval must not set the prior"}
    cb._next_eval = 200_000
    cb._maybe_eval()
    assert cb.model.policy.names == {"stop"}

    cb.model.action_space = type("Space", (), {"shape": (5,)})()
    one_side = [-1] * 95 + [1] * 5
    wide = np.linspace(0.0, 1.0, 100).tolist()
    cb._prior_opens = _prior_book(
        100,
        direction=one_side,
        exit_mode=["ema_21", "structural"] * 50,
        stop_u=wide,
        target_u=wide,
    )
    cb.num_timesteps = 300_000
    cb.model.num_timesteps = 300_000
    cb._apply_prior_interval()
    assert cb.model.policy.names == set()


def test_empty_opens_still_turn_risk_on_at_the_floor() -> None:
    class _Policy:
        def __init__(self) -> None:
            self.names: set[str] = set()

        def set_active_prior(self, names: set[str]) -> None:
            self.names = set(names)

    class _Model:
        def __init__(self) -> None:
            self.policy = _Policy()
            self.action_space = type("Space", (), {"shape": (6,)})()
            self.num_timesteps = 100_000
            self.rollout_buffer = None

    cb = TrainingDashboardCallback(total_timesteps=1_000_000, log_every=100_000)
    cb.model = _Model()  # type: ignore[assignment]
    cb.num_timesteps = 100_000
    cb._risk_sum = 0.0
    cb._risk_n = 2000
    cb._apply_prior_interval()
    assert cb.model.policy.names == {"risk"}
    assert cb._risk_n == 0

    cb.num_timesteps = 200_000
    cb._risk_sum = 0.5 * 2000
    cb._risk_n = 2000
    cb._apply_prior_interval()
    assert cb.model.policy.names == set()


def test_step_info_carries_reward_parts_and_direction() -> None:
    from tests.test_envs.test_ny_steps_eod import _make_env

    env = _make_env(max_episode_steps=2)
    env.reset()
    _obs, _reward, _done, _truncated, info = env.step(0)
    assert info["position_direction"] == 0
    assert "pnl" in info["reward_parts"]
    assert "dsr" in info["reward_parts"]
    assert info["close_reason"] == ""

def test_format_dashboard_table() -> None:
    """The dashboard table is a flat section/key/value table."""
    snap = {
        "timesteps": 106496,
        "total_timesteps": 20_000_000,
        "train_reward_mean": 0.0,
        "reward_parts": {"pnl": 0.0, "dsr": 0.0, "soft_daily": 0.0, "soft_year": 0.0, "soft_trailing": 0.0, "strategy": 0.0, "breach": 0.0, "sweep": 0.0, "peak_dd": 0.0},
        "reward_per_1k": {"pnl": 0.0, "dsr": 0.0, "soft_daily": 0.0, "soft_year": 0.0, "soft_trailing": 0.0, "strategy": 0.0, "breach": 0.0, "sweep": 0.0, "peak_dd": 0.0},
        "breach_events": 0,
        "position": {"long": 0.0, "short": 0.0, "flat": 1.0},
        "action": {"mean": 0.0, "std": 1.001},
        "closes": {"daily_loss": 0.0, "trailing_dd": 0.0, "max_drawdown": 0.0, "structure_sl": 0.0, "structure_tp": 0.0, "session_end": 0.0, "peak_dd_behavior": 0.0},
        "n_closes": 0,
        "critic": {"ret_mean": 0.0, "val_mean": 0.0, "adv_mean": 0.0, "adv_std": 0.0},
        "policy": {"explained_variance": 0.0, "value_loss": 0.0, "approx_kl": 0.0001, "clip_fraction": 0.0, "entropy": 15.608, "std": 1.0, "grad_norm": "0.003"},
        "eval": {
            "return_pct": 0.26, "sharpe": 4.02, "sortino": 31.09, "max_drawdown": 0.03,
            "profit_factor": 29.39, "win_rate": 0.667, "avg_trade": 87.03, "n_trades": 3,
            "turnover": 0.0006, "reward_mean": -0.0003, "diversity_text": "", "diversity_table": "",
            "prior_names": [],
        },
        "prior_names": [],
        "sl_rank_median": None,
        "tp_rank_median": None,
    }
    text = format_dashboard_table(snap)
    assert "TRAINING" in text
    assert "timesteps" in text
    assert "106,496" in text
    assert "win_rate" in text
    assert "66.7%" in text
    assert "REWARD" in text
    assert "---" in text
    assert "|" in text
    # Every line must share one width so borders close exactly.
    lengths = {len(line) for line in text.split("\n") if line}
    assert len(lengths) == 1
    # The diversity blob must never widen the boxes.
    assert max(lengths) < 400


def test_format_diversity_table() -> None:
    """The diversity table is a DIVERSITY box with --- borders."""
    from quant_rl.eval.decision_diversity import format_diversity_table
    from quant_rl.eval.decision_diversity import DiversityThresholds
    import pandas as pd

    trades = pd.DataFrame({
        "sl_ref": ["htf", "po3_manipulation_low", "po3_manipulation_low"],
        "tp_ref": ["htf", "htf", "htf"],
        "n_sl": [2, 2, 2],
        "sl_index": [0.5, 0.0, 1.0],
        "n_tp": [2, 2, 2],
        "tp_index": [0.5, 0.0, 0.0],
        "exit_mode": ["structural", "structural", "ema_21"],
        "planned_rr": [1.54, 1.54, 1.54],
        "stop_u": [1.0, 1.5, 0.5],
        "direction": [1, -1, 1],
        "strategy": ["po3_ifvg", "po3_manipulation_low", "po3_manipulation_low"],
        "session": ["ny", "ny", "ny"],
        "manipulation": ["done", "against", "against"],
        "volatility_regime": ["high", "high", "high"],
        "trend_regime": ["flat", "flat", "flat"],
    })
    thresholds = DiversityThresholds()
    text = format_diversity_table(trades, thresholds)
    assert "DIVERSITY: SL/TP" in text
    assert "unique" in text
    assert "family" in text
    assert "planned rr" in text
    assert "TP COLLAPSE" in text
    assert "BY DIRECTION" in text
    assert "section" not in text
    # Every line of every box must share one width so borders close exactly.
    lengths = {len(line) for line in text.split("\n") if line}
    assert len(lengths) == 1
    assert max(lengths) < 400
