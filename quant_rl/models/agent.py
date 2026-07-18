"""PPO agent wiring stub.

Fill in ``build_agent`` with your hyperparameters and features extractor after
implementing ``quant_rl/models/encoder.py``.

Usage (after implementing encoder)
-----------------------------------
    from quant_rl.models.agent import build_agent
    model = build_agent(env, cfg)
    model.learn(total_timesteps=cfg.ppo.total_timesteps)
"""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

# Lazy imports so the stub file doesn't break if SB3/torch are absent
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def build_agent(env: Any, cfg: DictConfig) -> Any:
    """Build an SB3 PPO agent wired to the user's SequenceEncoder.

    Parameters
    ----------
    env:
        A :class:`quant_rl.envs.trading_env.TradingEnv` instance.
    cfg:
        Full OmegaConf config.

    Returns
    -------
    ``stable_baselines3.PPO`` model ready to call ``.learn()``.

    Raises
    ------
    NotImplementedError
        Until you implement ``quant_rl/models/encoder.py``.
    """
    if not _SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required: pip install stable-baselines3")

    # TODO: replace with your encoder once implemented
    from .encoder import SequenceEncoder  # noqa: F401
    raise NotImplementedError(
        "Implement quant_rl/models/encoder.py first, then wire it here.\n"
        "Pattern:\n"
        "  policy_kwargs = dict(\n"
        "      features_extractor_class=SequenceEncoder,\n"
        "      features_extractor_kwargs=dict(\n"
        "          seq_len=cfg.env.obs_window,\n"
        "          n_features=<F>,\n"
        "          latent_dim=<D>,\n"
        "      ),\n"
        "  )\n"
        "  return PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, ...cfg.ppo...)\n"
    )
