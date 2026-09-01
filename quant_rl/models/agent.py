"""RL agent wiring: builds SB3 PPO or SAC with sequence encoders.

Usage
-----
    from quant_rl.models.agent import build_agent
    model = build_agent(env, cfg)                    # default PPO with TCN
    model = build_agent(env, cfg, arch="transformer")
    model = build_agent(env, cfg, arch="gru", algo="sac")  # SAC agent
    model.learn(total_timesteps=cfg.ppo.total_timesteps)
    model.save("models/ppo_trading")
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces
from omegaconf import DictConfig

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def build_agent(
    env: Any,
    cfg: DictConfig,
    arch: str = "tcn",
    algo: str = "ppo",
    use_vae: bool = False,
    vae: Any | None = None,
) -> Any:
    """Build an SB3 PPO or SAC agent wired to the sequence encoder.

    Parameters
    ----------
    env:
        A :class:`quant_rl.envs.trading_env.TradingEnv` instance.
    cfg:
        Full OmegaConf config.
    arch:
        ``"tcn"`` (default), ``"transformer"``, or ``"gru"``.
    algo:
        ``"ppo"`` (default) or ``"sac"``.
    use_vae:
        If True, use the VAE feature extractor (default: False).
    vae:
        A trained :class:`quant_rl.models.vae.VAE` instance.  Required when
        ``use_vae`` is ``True``.

    Returns
    -------
    ``stable_baselines3.PPO`` or ``stable_baselines3.SAC`` ready to call
    ``.learn()``.

    Raises
    ------
    ValueError
        If ``use_vae`` is ``True`` but ``vae`` is ``None``.
    """
    if not _SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required: pip install stable-baselines3")

    from .encoder import GRUEncoder, TCNEncoder, TransformerEncoder
    from .vae import VAEFeatureExtractor

    if use_vae and vae is None:
        raise ValueError("use_vae=True requires a constructed VAE instance passed via vae=")

    # Infer F from the env's observation space
    n_features: int = env.observation_space["seq"].shape[1]

    # Choose feature extractor based on architecture / VAE use
    extractor_cls: Any
    if use_vae and "vae_z" in env.observation_space.spaces:
        extractor_cls = VAEFeatureExtractor
        latent_dim = env.observation_space["vae_z"].shape[0]
        extractor_kwargs: dict[str, Any] = {"vae": vae, "freeze": True}
    else:
        if arch == "transformer":
            extractor_cls = TransformerEncoder
        elif arch == "gru":
            extractor_cls = GRUEncoder
        else:
            extractor_cls = TCNEncoder

        latent_dim = 128
        extractor_kwargs = dict(
            seq_len=cfg.env.obs_window,
            n_features=n_features,
            latent_dim=latent_dim,
        )

    # Two hidden layers after the encoder. SAC uses ``qf`` for the critic.
    if algo == "sac":
        net_arch: dict[str, Any] = dict(pi=[256, 128], qf=[256, 128])
    else:
        net_arch = dict(pi=[256, 128], vf=[256, 128])

    policy_kwargs: dict[str, Any] = dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=net_arch,
    )

    vec_env = DummyVecEnv([lambda: env])

    if algo == "sac":
        # SAC requires continuous action space
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("SAC requires continuous action space (Box)")

        sac_cfg = cfg.get("sac", cfg.get("ppo", {}))
        return SAC(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=sac_cfg.get("learning_rate", 3e-4),
            buffer_size=sac_cfg.get("buffer_size", 1_000_000),
            batch_size=sac_cfg.get("batch_size", 256),
            tau=sac_cfg.get("tau", 0.005),
            gamma=sac_cfg.get("gamma", 0.99),
            train_freq=sac_cfg.get("train_freq", 1),
            gradient_steps=sac_cfg.get("gradient_steps", 1),
            ent_coef=sac_cfg.get("ent_coef", "auto"),
            learning_starts=sac_cfg.get("learning_starts", 500),
            verbose=1,
        )

    # Default to PPO
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
