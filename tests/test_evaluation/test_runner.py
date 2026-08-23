"""Integration-style tests for the episode runner using a fake environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quant_rl.evaluation import run_episode


class _FakeEnv:
    """Minimal gym-like env: equity drifts up, two trades, one breach."""

    def __init__(self) -> None:
        self.initial_balance = 100_000.0
        self.trade_log: list[dict[str, Any]] = [
            {"pnl": 300.0},
            {"pnl": -100.0},
            {"no_pnl_key": True},
        ]
        self.breach_events: list[dict[str, Any]] = [{"reason": "daily_loss"}]
        self._step_idx = 0
        self.actions_seen: list[Any] = []

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step_idx = 0
        return {}, {}

    def step(
        self,
        action: Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.actions_seen.append(action)
        self._step_idx += 1
        done = self._step_idx >= 3
        info = {"equity": 100_000.0 + 50.0 * self._step_idx}
        return {}, 0.0, done, False, info


@pytest.mark.integration
class TestRunEpisode:
    def test_collects_equity_and_trade_stats(self) -> None:
        # Arrange
        env = _FakeEnv()

        # Act
        metrics = run_episode(env, action_fn=lambda obs: 0.5)

        # Assert
        assert metrics.n_trades == 2  # entry without "pnl" key is skipped
        assert metrics.breach_count == 1
        assert metrics.expectancy == pytest.approx(100.0)
        assert metrics.total_pnl == pytest.approx(150.0)

    def test_passes_policy_actions_to_env(self) -> None:
        # Arrange
        env = _FakeEnv()

        # Act
        run_episode(env, action_fn=lambda obs: np.float64(0.25))

        # Assert
        assert len(env.actions_seen) == 3
        assert all(a == np.float64(0.25) for a in env.actions_seen)

    def test_max_steps_cap_stops_runaway_episode(self) -> None:
        # Arrange
        class _EndlessEnv(_FakeEnv):
            def step(
                self,
                action: Any,
            ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
                _, reward, _, _, info = super().step(action)
                return {}, reward, False, False, info

        env = _EndlessEnv()

        # Act — 10 steps of +50 each on a 100k account
        metrics = run_episode(env, action_fn=lambda obs: 0.0, max_steps=10)

        # Assert
        assert metrics.total_return_pct == pytest.approx(500.0 / 100_000.0 * 100.0)
