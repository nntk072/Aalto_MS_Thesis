"""Train/evaluate helpers shared by walk-forward and ablation experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..envs.trading_env import TradingEnv
from ..evaluation import run_episode, sweep_delay_breakdown
from ..models.agent import build_agent


def load_config(path: str = "quant_rl/config/default.yaml") -> DictConfig:
    """Load and normalise the experiment configuration."""
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise ValueError(f"config root at {path} must be a mapping, got list")
    return loaded


def train_and_score(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    cfg: DictConfig,
    *,
    algo: str = "ppo",
    arch: str = "gru",
    use_vae: bool = False,
    continuous: bool | None = None,
    steps: int = 50_000,
    seed: int = 42,
    eval_bars: pd.DataFrame | None = None,
    eval_features: pd.DataFrame | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], Any]:
    """Train an agent on ``bars`` and score one evaluation episode.

    Args:
        bars: OHLCV DataFrame for the fold.
        features: Feature matrix aligned with ``bars``.
        cfg: Experiment config containing the ``ppo``/``sac`` blocks.
        algo: RL algorithm, ``"ppo"`` or ``"sac"``.
        arch: Sequence encoder, ``"tcn"``, ``"gru"`` or ``"transformer"``.
        use_vae: Condition the policy on the VAE narrative embedding.
        continuous: Force a Box action space; defaults to SAC-only.
        steps: Training timesteps.
        seed: Random seed for reproducibility.
        eval_bars: Separate evaluation bars; defaults to ``bars``.
        eval_features: Feature matrix for ``eval_bars``.

    Returns:
        Tuple of (metrics report dict, environment trade log, trained model).
    """
    if continuous is None:
        continuous = algo == "sac"
    env = TradingEnv(
        bars=bars,
        features=features,
        use_sweep_reward=True,
        continuous_actions=continuous,
    )
    model = build_agent(env, cfg, arch=arch, algo=algo, use_vae=use_vae)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    eval_env = TradingEnv(
        bars=bars if eval_bars is None else eval_bars,
        features=features if eval_features is None else eval_features,
        use_sweep_reward=True,
        continuous_actions=continuous,
    )

    def action_fn(obs: dict[str, Any]) -> Any:
        return model.predict(obs, deterministic=True)[0]

    metrics = run_episode(eval_env, action_fn=action_fn)
    delays = sweep_delay_breakdown(eval_env.trade_log)
    report: dict[str, float] = {
        "sharpe": round(metrics.sharpe, 3),
        "sortino": round(metrics.sortino, 3),
        "calmar": round(metrics.calmar, 3),
        "max_drawdown": round(metrics.max_drawdown, 4),
        "total_return_pct": round(metrics.total_return_pct, 3),
        "breach_count": metrics.breach_count,
        **{k: round(v, 3) for k, v in delays.items()},
    }
    return report, eval_env.trade_log, model
