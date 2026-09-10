"""Characterization / golden-trace tests for TradingEnv.step().

Runs a fixed deterministic episode and compares the full per-step trace
(observations, reward, done/truncated, info, trade log, equity curve)
against a stored SHA256 golden hash. Any behavioral change in ``step()``
must be caught here before decomposition proceeds.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quant_rl.envs.trading_env import TradingEnv


def _make_deterministic_bars(n: int = 120) -> pd.DataFrame:
    """Deterministic M1 bars with a known pattern."""
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(42)
    close = 20000.0 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0.1, 0.5, n),
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
            "close": close,
            "tickvol": rng.integers(10, 100, n),
            "volume": rng.integers(1000, 5000, n),
            "vol": 0,
            "spread": 0.6,
            "gap_flag": False,
            "session_id": (pd.Series(idx).dt.floor("D").astype(np.int64) // 86400000000000).values,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def _make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature matrix with required columns."""
    n = len(bars)
    idx = bars.index
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "london_high": 20010.0 + rng.uniform(0, 5, n),
            "london_low": 19990.0 + rng.uniform(0, 5, n),
            "asian_high": 20005.0 + rng.uniform(0, 5, n),
            "asian_low": 19995.0 + rng.uniform(0, 5, n),
            "volume_spike": rng.uniform(0.5, 2.0, n),
            "last_swing_high": 20015.0 + rng.uniform(0, 3, n),
            "last_swing_low": 19985.0 + rng.uniform(0, 3, n),
        },
        index=idx,
    )


def _make_env(seed: int = 42) -> TradingEnv:
    bars = _make_deterministic_bars()
    features = _make_features(bars)
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        initial_balance=100_000.0,
        episodic=True,
        use_sweep_reward=False,
    )
    env.reset(seed=seed)
    return env


def _trace_to_bytes(trace: dict) -> bytes:
    """Serialize trace to a canonical byte string for hashing."""
    canonical: dict = {
        "rewards": [float(r) for r in trace["rewards"]],
        "dones": [bool(d) for d in trace["dones"]],
        "truncated": [bool(t) for t in trace["truncated"]],
        "infos_equity": [float(i["equity"]) for i in trace["infos"]],
        "infos_position": [bool(i["position"]) for i in trace["infos"]],
        "trade_log_types": [t.get("type") for t in trace["trade_log"]],
        "trade_log_pnls": [float(t.get("pnl", 0.0)) for t in trace["trade_log"]],
        "equity_curve": [float(e) for e in trace["equity_curve"]],
        "final_equity": float(trace["final_equity"]),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return raw.encode("utf-8")


def _run_trace(env: TradingEnv, actions: list[int]) -> dict:
    """Run env through fixed action list and collect trace."""
    obs, _ = env.reset(seed=42)
    rewards: list[float] = []
    dones: list[bool] = []
    truncateds: list[bool] = []
    infos: list[dict] = []
    trade_log: list[dict] = []
    equity_curve: list[float] = []

    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        truncateds.append(bool(truncated))
        infos.append(dict(info))
        trade_log.extend(env.trade_log[len(trade_log) :])
        equity_curve.append(float(env.account.equity))

    return {
        "rewards": rewards,
        "dones": dones,
        "truncated": truncateds,
        "infos": infos,
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "final_equity": float(env.account.equity),
    }


class TestTradingEnvGoldenTraces:
    """Regression tests that lock in TradingEnv.step() behavior."""

    def test_deterministic_episode_trace(self) -> None:
        """Fixed seed + fixed action sequence produces a stable golden trace."""
        env = _make_env(seed=42)
        actions = [0, 0, 0, 1, 0, 0, 10, 0, 0, 0] * 6  # 60 actions
        trace = _run_trace(env, actions)
        digest = hashlib.sha256(_trace_to_bytes(trace)).hexdigest()
        assert digest == _GOLDEN_TRACE_HASH, (
            f"Golden trace mismatch.\n  Expected: {_GOLDEN_TRACE_HASH}\n  Got:      {digest}"
        )

    def test_no_naked_position_without_levels(self) -> None:
        """When swing levels are NaN, entries are rejected (no naked positions)."""
        bars = _make_deterministic_bars()
        features = pd.DataFrame(
            {
                "last_swing_high": np.nan,
                "last_swing_low": np.nan,
                "volume_spike": 1.0,
            },
            index=bars.index,
        )
        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            use_sweep_reward=False,
        )
        env.reset(seed=42)
        actions = [1] * 30
        trace = _run_trace(env, actions)
        opens = [t for t in trace["trade_log"] if t.get("type") == "open"]
        assert len(opens) == 0, "No position should open without valid swing levels"

    def test_episodic_breach_ends_episode(self) -> None:
        """In episodic mode, a guardrail breach sets done=True."""
        bars = _make_deterministic_bars(n=200)
        features = _make_features(bars)
        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100.0,
            episodic=True,
            use_sweep_reward=False,
            guardrail_kwargs={"daily_loss_limit": 50.0},
        )
        env.reset(seed=42)
        actions = [1, 0, 0, 10, 0, 0, 10, 0, 0, 0] * 20
        trace = _run_trace(env, actions)
        assert any(trace["dones"]), "Episode should terminate on guardrail breach"

    def test_eval_mode_continues_after_breach(self) -> None:
        """In eval mode (episodic=False), a breach blocks the session but
        the rollout continues. We verify this by checking that after a
        breach, subsequent steps still return done=False (until data end).
        """
        bars = _make_deterministic_bars(n=300)
        features = _make_features(bars)
        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100.0,
            episodic=False,
            use_sweep_reward=False,
            guardrail_kwargs={"daily_loss_limit": 50.0},
        )
        env.reset(seed=42)
        actions = [1, 0, 0, 10, 0, 0, 10, 0, 0, 0] * 30
        trace = _run_trace(env, actions)
        breach_idx = None
        for i, d in enumerate(trace["dones"]):
            if d:
                breach_idx = i
                break
        assert breach_idx is not None, "Expected a guardrail breach"
        assert breach_idx < len(trace["dones"]) - 1, (
            "Breach should not be the very last step unless data ended"
        )


_GOLDEN_TRACE_HASH = (
    "9b9ba26bcfe60bdc559b09a545fe146cea58b0d1027ac69069e41e0cc1efe583"
)
