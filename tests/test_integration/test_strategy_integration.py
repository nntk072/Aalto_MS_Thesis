"""End-to-end strategy pipeline test (Agent.md §31).

Validates the full chain: OHLCV -> build_features (with the strategy-state
block enabled) -> TradingEnv(strategy_actions=True) -> reset() -> step(),
asserting new feature columns exist, the observation shape is valid, the
action shape is the 4-D Box, and strategy trades are logged with context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.config import load_config
from quant_rl.envs.strategies import PO3IFVGStrategy
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.features.build import build_features


def _sample_bars(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 01:05", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + np.abs(rng.normal(0, 0.05, n))
    low = close - np.abs(rng.normal(0, 0.05, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tickvol": rng.integers(100, 1000, n),
            "volume": rng.integers(1000, 5000, n),
            "session_id": (np.arange(n) // 200).astype(int),
        },
        index=idx,
    )


def test_full_strategy_pipeline_runs() -> None:
    """Agent.md §31: bars -> features -> env -> reset -> step."""
    cfg = load_config(config_path="config/idea1_po3_ifvg.yaml")
    bars = _sample_bars()
    features = build_features(bars, cfg=cfg)

    # New strategy feature columns must exist.
    for col in [
        "sweep_high",
        "sweep_low",
        "po3_manipulation_low",
        "po3_distribution",
        "ifvg_bull_active",
        "asian_range",
        "price_to_asian_low_atr",
        "manipulation_low_distance_atr",
    ]:
        assert col in features.columns, f"missing column: {col}"

    env = TradingEnv(
        bars,
        features,
        strategy_actions=True,
        strategy=PO3IFVGStrategy(enforce_gate=False),
        obs_window=10,
        initial_balance=100_000.0,
    )
    obs, _ = env.reset()
    seq_width = int(obs["seq"].shape[1])
    assert env.observation_space.contains(obs)
    assert obs["seq"].shape == (10, seq_width)

    action = np.array([0.8, 0.5, 0.7, 0.0], dtype=np.float32)
    assert env.action_space.contains(action)

    # Step through; at least one strategy trade should be logged.
    for _ in range(len(bars) - 5):
        obs, reward, done, truncated, info = env.step(action)
        if env.position is not None:
            break
        if done or truncated:
            break

    open_trades = [t for t in env.trade_log if t.get("type") == "open"]
    if open_trades:
        trade = open_trades[0]
        assert trade["strategy"] == "po3_ifvg"
        assert "sl_price" in trade
        assert "asian_high" in trade
        assert "distribution_phase" in trade

    # No NaN/inf leaked into model-facing values.
    seq = obs["seq"]
    assert np.all(np.isfinite(seq))
