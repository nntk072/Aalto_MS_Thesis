"""PPO agent wiring: builds an SB3 PPO with the sequence encoder.

Usage
-----
    from quant_rl.models.agent import build_agent
    model = build_agent(env, cfg)                    # default TCN
    model = build_agent(env, cfg, arch="transformer")
    model.learn(total_timesteps=cfg.ppo.total_timesteps)
    model.save("models/ppo_trading")
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def build_agent(env: Any, cfg: DictConfig, arch: str = "tcn") -> Any:
    """Build an SB3 PPO agent wired to the sequence encoder.

    Parameters
    ----------
    env:
        A :class:`quant_rl.envs.trading_env.TradingEnv` instance.
    cfg:
        Full OmegaConf config.
    arch:
        ``"tcn"`` (default) or ``"transformer"``.

    Returns
    -------
    ``stable_baselines3.PPO`` ready to call ``.learn()``.
    """
    if not _SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required: pip install stable-baselines3")

    from .encoder import TCNEncoder, TransformerEncoder

    extractor_cls = TransformerEncoder if arch == "transformer" else TCNEncoder

    # Infer F from the env's observation space
    n_features: int = env.observation_space["seq"].shape[1]

    policy_kwargs: dict[str, Any] = dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=dict(
            seq_len=cfg.env.obs_window,
            n_features=n_features,
            latent_dim=128,
        ),
        # Two hidden layers after the encoder
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    vec_env = DummyVecEnv([lambda: env])

    return PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        verbose=1,
    )
