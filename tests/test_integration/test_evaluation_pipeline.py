"""Integration test: `run_episode` against a real `TradingEnv`.

Locks in the contract between the evaluation pipeline and the environment
that the unit tests (which use a `_FakeEnv`) cannot verify:

- ``env.reset() -> (obs, info)``
- ``env.step(action) -> (obs, reward, done, truncated, info)`` with
  ``"equity"`` in ``info``
- ``env.trade_log`` with a ``"pnl"`` key on closed trades
- ``env.breach_events`` list
- ``env.initial_balance`` float

If ``TradingEnv``'s internals drift from this contract, ``run_episode``
would silently produce wrong or empty metrics — this test catches that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.baselines import BaseStrategy, BuyAndHoldStrategy
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.evaluation import PerformanceMetrics, run_episode

OBS_WINDOW = 60


def _make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Numeric bar columns plus swing levels so the env computes SL/TP.

    Without ``last_swing_low`` / ``last_swing_high`` the env opens
    fallback positions with no SL/TP, which never close in a trending
    synthetic series — adding the swing columns exercises the full
    open → TP/SL close → re-entry loop.
    """
    features = bars.select_dtypes(include=[np.number]).copy()
    features["last_swing_low"] = bars["close"] - 5.0
    features["last_swing_high"] = bars["close"] + 5.0
    return features


@pytest.fixture
def trending_bars() -> pd.DataFrame:
    """600 one-min bars with a steady rise so long trades hit their TP."""
    n = 600
    idx = pd.date_range("2025-06-02 01:05", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(7)
    drift = np.arange(n) * 0.15  # deterministic upward drift
    close = 20_000.0 + drift + rng.normal(0, 0.05, n)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=idx,
    )


class _CountingStrategy(BaseStrategy):
    """Emits its own step counter, so misalignment with env bars is observable."""

    def __init__(self, n_bars: int) -> None:
        super().__init__(n_bars)
        self.seen: list[int] = []

    def act(self, obs: dict) -> float:  # type: ignore[type-arg]
        self.seen.append(self._idx)
        return super().act(obs)

    def _signal(self, idx: int) -> float:
        return 0.0


def _continuous_action(strategy: BaseStrategy):
    """Wrap a strategy's float action for TradingEnv's continuous decoder."""

    def action_fn(obs: dict) -> np.ndarray:  # type: ignore[type-arg]
        return np.array([strategy.act(obs)], dtype=np.float32)

    return action_fn


@pytest.mark.integration
class TestRunEpisodeWithTradingEnv:
    def test_buy_and_hold_produces_sane_metrics(self, trending_bars: pd.DataFrame) -> None:
        # Arrange
        features = _make_features(trending_bars)
        env = TradingEnv(
            bars=trending_bars,
            features=features,
            continuous_actions=True,
            episodic=False,
        )
        strategy = BuyAndHoldStrategy(n_bars=len(trending_bars))
        strategy.fast_forward(OBS_WINDOW)

        # Act
        metrics = run_episode(env, action_fn=_continuous_action(strategy))

        # Assert: the contract values are consumed, not degenerate
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.n_trades > 0  # BuyAndHold enters; TP/SL closes trades
        assert np.isfinite(metrics.sharpe)
        assert np.isfinite(metrics.max_drawdown)
        assert metrics.max_drawdown >= 0.0
        assert 0.0 <= metrics.win_rate <= 1.0
        assert metrics.total_pnl == pytest.approx(env.account.equity - env.initial_balance)

    def test_env_contract_attributes_are_consumed(self, trending_bars: pd.DataFrame) -> None:
        # Arrange
        features = _make_features(trending_bars)
        env = TradingEnv(
            bars=trending_bars,
            features=features,
            continuous_actions=True,
            episodic=False,
        )
        strategy = BuyAndHoldStrategy(n_bars=len(trending_bars))
        strategy.fast_forward(OBS_WINDOW)

        # Act
        run_episode(env, action_fn=_continuous_action(strategy))

        # Assert: run_episode relied on these attributes existing
        assert isinstance(env.initial_balance, float)
        assert env.initial_balance == 100_000.0
        assert isinstance(env.trade_log, list)
        assert any("pnl" in trade for trade in env.trade_log)
        assert isinstance(env.breach_events, list)

    def test_strategy_step_counter_aligns_with_env_bars(self, trending_bars: pd.DataFrame) -> None:
        # Arrange: env starts trading at bar `obs_window`; a strategy that
        # was fast-forwarded must emit signals for bars obs_window..n-1.
        features = trending_bars.select_dtypes(include=[np.number])
        env = TradingEnv(
            bars=trending_bars,
            features=features,
            continuous_actions=True,
            episodic=False,
        )
        strategy = _CountingStrategy(n_bars=len(trending_bars))
        strategy.fast_forward(OBS_WINDOW)

        # Act
        run_episode(env, action_fn=_continuous_action(strategy), max_steps=5)

        # Assert
        assert strategy.seen, "strategy was never called"
        assert strategy.seen[0] == OBS_WINDOW

    def test_run_episode_without_fast_forward_starts_at_zero(
        self, trending_bars: pd.DataFrame
    ) -> None:
        """A fresh (non-fast-forwarded) strategy is misaligned by obs_window."""
        features = trending_bars.select_dtypes(include=[np.number])
        env = TradingEnv(
            bars=trending_bars,
            features=features,
            continuous_actions=True,
            episodic=False,
        )
        strategy = _CountingStrategy(n_bars=len(trending_bars))

        run_episode(env, action_fn=_continuous_action(strategy), max_steps=3)

        assert strategy.seen[0] == 0  # documents the misalignment hazard
