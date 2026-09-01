"""Contract test: ``evaluate_model`` output → ``calculate_metrics`` input.

The two modules live in different subpackages (``quant_rl.eval`` and
``quant_rl.evaluation``), were migrated separately, and have changed
column names and index types at least once. A silent shape mismatch would
not be caught until a real training run tries to plot metrics, by which
point the run is already long and the operator has lost time.

This test pins the contract:

- ``evaluate_model`` returns a ``trades`` DataFrame with a ``"pnl"`` column
  (consumed by ``calculate_metrics``).
- It returns an ``equity`` object with ``.to_numpy()`` (a ``pd.Series``) so
  the metrics layer can convert to a float array.
- The first equity value is positive (the metrics layer raises
  ``ValueError`` on a zero/negative initial balance).
- All session-count keys used by ``train_rl.py`` are present.

We drive ``evaluate_model`` with a random SB3 PPO so the contract is
checked against a *real* trained policy → real trade log, not a stub.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.evaluation import calculate_metrics

pytestmark = pytest.mark.slow


def _make_synthetic_bars(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, n),
            "high": close + rng.uniform(0, 2, n),
            "low": close - rng.uniform(0, 2, n),
            "close": close,
            "tickvol": rng.integers(10, 200, n),
            "volume": rng.integers(1000, 5000, n),
            "vol": np.zeros(n, dtype=int),
            "spread": np.full(n, 0.6),
            "gap_flag": False,
            "session_id": 0,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def _make_features(bars: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(bars)
    return pd.DataFrame(
        {
            "london_high": 20500.0,
            "london_low": 19500.0,
            "asian_high": 20300.0,
            "asian_low": 19700.0,
            "volume_spike": rng.uniform(0.5, 2.5, n),
            "last_swing_high": np.nan,
            "last_swing_low": np.nan,
        },
        index=bars.index,
    )


def test_evaluate_model_output_satisfies_metrics_contract() -> None:
    """End-to-end: drive a tiny PPO through evaluate_model, then feed the
    resulting dict straight into ``calculate_metrics`` without any
    key remapping. Both layers' documented contracts are honored.
    """
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from quant_rl.eval.rollout import evaluate_model

    bars = _make_synthetic_bars()
    features = _make_features(bars)

    # Train a tiny PPO to get a non-trivial policy.
    from quant_rl.envs.trading_env import TradingEnv

    train_env = TradingEnv(bars=bars, features=features, obs_window=10, max_episode_steps=64)
    model = PPO(
        "MultiInputPolicy",
        train_env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        learning_rate=3e-4,
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=128, progress_bar=False)

    result = evaluate_model(
        model,
        bars=bars,
        features=features,
        obs_window=10,
        initial_balance=100_000.0,
    )

    # --- Contract: equity is convertible to numpy and starts positive ---
    assert hasattr(result["equity"], "to_numpy"), (
        "evaluate_model must return a pd.Series for equity "
        "(calculate_metrics relies on .to_numpy())"
    )
    eq = result["equity"].to_numpy(dtype=np.float32)
    assert eq.size > 0
    assert eq[0] > 0, "initial equity must be positive (calculate_metrics rejects <= 0)"

    # --- Contract: trades DataFrame has a 'pnl' column ---
    trades = result["trades"]
    assert isinstance(trades, pd.DataFrame)
    if len(trades) > 0:
        assert "pnl" in trades.columns, (
            "calculate_metrics reads trades['pnl'] — a missing column "
            "crashes the metrics layer. columns=" + str(trades.columns.tolist())
        )

    # --- Contract: session-count keys are present (train_rl.py uses them) ---
    for key in ("n_sessions", "n_breach_sessions"):
        assert key in result, f"evaluate_model must expose '{key}' for train_rl.py"

    # --- Hand the dict to calculate_metrics: it must not raise ---
    metrics = calculate_metrics(
        result["equity"],
        trades=result["trades"],
        n_sessions=result.get("n_sessions", 1),
        n_breach_sessions=result.get("n_breach_sessions", 0),
    )
    # PerformanceMetrics is a frozen dataclass; basic shape checks.
    assert metrics.n_trades >= 0
    assert metrics.sharpe is not None
    assert np.isfinite(metrics.sharpe) or metrics.n_trades == 0
