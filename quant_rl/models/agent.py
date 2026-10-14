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

import multiprocessing
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig

from quant_rl.action_schema import ActionSchema
from quant_rl.utils.device import enable_extractor_autocast

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def validate_model_action_schema(
    model: Any,
    expected_schema: ActionSchema,
    *,
    allow_legacy: bool = False,
    legacy_layout_key: str | None = None,
) -> str | None:
    """Validate action semantics before using a checkpoint for evaluation.

    SB3 reconstructs the policy when loading, so its constructor default key
    cannot prove the saved layout. Versioned metadata belongs on the PPO model
    (included in SB3's serialized model data). Unversioned checkpoints require
    an explicit legacy opt-in plus an explicit layout selection. Width alone
    cannot distinguish layouts such as ``6_dir`` and ``6_sl``. The one
    supported width migration is the documented 4D no-exit layout to the
    current 5D base layout, with a structural-exit coordinate appended at
    inference time.
    """
    shape = getattr(getattr(model, "action_space", None), "shape", None)
    actual_width = int(shape[0]) if shape and len(shape) == 1 else None
    policy = getattr(model, "policy", None)
    version = getattr(model, "_action_schema_version", None)
    layout = getattr(model, "_action_layout_key", None)
    # Accept policy-only metadata from lightweight compatibility adapters;
    # a freshly reconstructed SB3 policy's default key has no version and is
    # therefore never trusted as proof of its original layout.
    if version is None:
        version = getattr(policy, "_action_schema_version", None)
        if version is not None and layout is None:
            layout = getattr(policy, "_action_layout_key", None)

    if version is None:
        if not allow_legacy:
            raise ValueError("unversioned action checkpoint needs explicit legacy opt-in")
        if legacy_layout_key is None:
            raise ValueError("unversioned action checkpoint needs an explicit legacy layout key")
        if layout is not None and layout != expected_schema.key:
            raise ValueError(
                f"legacy action layout {layout!r} differs from {expected_schema.key!r}"
            )
        migration: str | None = None
        if legacy_layout_key == "legacy4_no_exit_v1":
            if actual_width != 4 or expected_schema.key != "5":
                raise ValueError(
                    "legacy4_no_exit_v1 requires a 4D checkpoint and the current 5D base layout"
                )
            migration = "legacy4_no_exit_v1_to_overlay-action-v2-5"
        elif legacy_layout_key != expected_schema.key:
            raise ValueError(
                f"explicit legacy layout {legacy_layout_key!r} differs from "
                f"expected {expected_schema.key!r}"
            )
        elif actual_width != expected_schema.width:
            raise ValueError(
                f"legacy action width {actual_width} disagrees with "
                f"selected layout {legacy_layout_key!r} width {expected_schema.width}"
            )
        manifest = expected_schema.to_dict()
        manifest["checkpoint_metadata"] = "legacy_unversioned"
        manifest["legacy_layout_key"] = legacy_layout_key
        manifest["migration"] = migration
        setattr(model, "_action_schema", manifest)
        if policy is not None:
            setattr(policy, "_action_schema", manifest)
        return migration

    if actual_width != expected_schema.width:
        raise ValueError(
            f"checkpoint action width {actual_width} disagrees with "
            f"expected width {expected_schema.width}"
        )

    if version != expected_schema.version:
        raise ValueError(
            f"action schema version {version!r} differs from {expected_schema.version!r}"
        )
    if layout != expected_schema.key:
        raise ValueError(f"action layout {layout!r} differs from {expected_schema.key!r}")
    # The policy rebuilt by SB3 starts with a constructor default. Restore
    # the validated layout for correct mode-prior coordinates after loading.
    if policy is not None:
        policy._action_layout_key = layout
        policy._action_schema_version = version
    # The validated decoder contract is useful to dashboards and exports;
    # it does not change the serialized policy's action coordinates.
    manifest = expected_schema.to_dict()
    manifest["checkpoint_metadata"] = "versioned"
    setattr(model, "_action_schema", manifest)
    if policy is not None:
        setattr(policy, "_action_schema", manifest)
    return None


class LegacyFourToFiveActionAdapter:
    """Append the explicit structural-exit coordinate for a legacy 4D policy."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def predict(self, observation: Any, *, deterministic: bool = True) -> tuple[Any, Any]:
        action, state = self.model.predict(observation, deterministic=deterministic)
        values = np.asarray(action)
        if values.shape[-1:] != (4,):
            raise ValueError(f"legacy 4D policy returned unexpected action shape {values.shape}")
        structural_exit = np.full((*values.shape[:-1], 1), -1.0, dtype=values.dtype)
        return np.concatenate((values, structural_exit), axis=-1), state


def encoder_for(arch: str) -> type[Any]:
    """Feature extractor selected by ``agent.arch``. Unknown names stay on the TCN."""
    from .encoder import GRUEncoder, MTFEncoder, TCNEncoder, TransformerEncoder

    if arch == "transformer":
        return TransformerEncoder
    if arch == "gru":
        return GRUEncoder
    if arch == "mtf":
        return MTFEncoder
    return TCNEncoder


def build_agent(
    env: Any,
    cfg: DictConfig,
    arch: str = "tcn",
    algo: str = "ppo",
    use_vae: bool = False,
    vae: Any | None = None,
    device: str | torch.device | None = None,
    env_fn: Callable[[], Any] | None = None,
) -> Any:
    """Build an SB3 PPO or SAC agent wired to the sequence encoder.

    Parameters
    ----------
    env:
        A :class:`quant_rl.envs.trading_env.TradingEnv` instance.
    cfg:
        Full OmegaConf config.
    arch:
        ``"tcn"`` (default), ``"transformer"``, ``"gru"``, or ``"mtf"``.
    algo:
        ``"ppo"`` (default) or ``"sac"``.
    use_vae:
        If True, use the VAE feature extractor (default: False).
    vae:
        A trained :class:`quant_rl.models.vae.VAE` instance.  Required when
        ``use_vae`` is ``True``.
    device:
        Torch device to place the policy on (e.g. ``"cpu"``, ``"cuda"``).
        ``None`` lets SB3 auto-detect.
    env_fn:
        Optional factory ``() -> TradingEnv`` used when ``cfg.env.n_envs > 1``
        to spawn parallel environments via ``SubprocVecEnv``.  When omitted or
        when ``n_envs == 1``, falls back to ``DummyVecEnv`` (sequential,
        single-env — the original behaviour).

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

    from .vae import VAEFeatureExtractor

    if use_vae and vae is None:
        raise ValueError("use_vae=True requires a constructed VAE instance passed via vae=")

    # Infer F from the env's observation space
    n_features: int = env.observation_space["seq"].shape[1]
    account_dim: int = int(env.observation_space["account"].shape[0])

    # Choose feature extractor based on architecture / VAE use
    extractor_cls: Any
    if use_vae and "vae_z" in env.observation_space.spaces:
        extractor_cls = VAEFeatureExtractor
        latent_dim = env.observation_space["vae_z"].shape[0]
        extractor_kwargs: dict[str, Any] = {
            "vae": vae,
            "freeze": True,
            "account_dim": account_dim,
        }
    else:
        if arch == "mtf" and "seq_m5" not in env.observation_space.spaces:
            raise ValueError("arch 'mtf' requires an environment built with mtf=True")
        extractor_cls = encoder_for(arch)

        latent_dim = 128
        if arch == "mtf":
            extractor_kwargs = dict(latent_dim=latent_dim, account_dim=account_dim)
        else:
            extractor_kwargs = dict(
                seq_len=cfg.env.obs_window,
                n_features=n_features,
                latent_dim=latent_dim,
                account_dim=account_dim,
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

    n_envs: int = int(getattr(cfg.get("env", {}), "n_envs", 1))
    vec_env: Any
    if n_envs > 1 and env_fn is not None:
        bundle_workers = str(cfg.env.get("data_source", "frames")) == "bundle"
        use_forkserver = bundle_workers and "forkserver" in multiprocessing.get_all_start_methods()
        if use_forkserver:
            multiprocessing.set_forkserver_preload(
                [
                    "stable_baselines3.common.vec_env.subproc_vec_env",
                    "quant_rl.envs.worker",
                    "quant_rl.envs.trading_env",
                ]
            )

        # Stage 3: thread caps. Cap each worker to 1 thread to prevent
        # OMP/MKL/torch thread-pool oversubscription (64 workers × default pools).
        def _thread_capped_env_fn() -> Any:
            import os as _os

            for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                _os.environ[_var] = "1"
            _torch = sys.modules.get("torch")
            if _torch is not None:
                _torch.set_num_threads(1)
            return env_fn()

        start_method = "forkserver" if use_forkserver else None
        vec_env = SubprocVecEnv([_thread_capped_env_fn] * n_envs, start_method=start_method)
    else:
        vec_env = DummyVecEnv([lambda: env])

    if algo == "sac":
        # SAC requires continuous action space
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("SAC requires continuous action space (Box)")

        sac_cfg = cfg.get("sac", cfg.get("ppo", {}))
        sac_kwargs: dict[str, Any] = dict(
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
        if device is not None:
            sac_kwargs["device"] = device
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            **sac_kwargs,
        )
        return _finalize_cuda_policy(model, device)

    # Default to PPO
    ppo_n_steps = cfg.ppo.n_steps
    if n_envs > 1 and env_fn is not None:
        ppo_n_steps = max(1, ppo_n_steps // n_envs)

    ppo_kwargs: dict[str, Any] = dict(
        n_steps=ppo_n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        verbose=1,
    )
    if device is not None:
        ppo_kwargs["device"] = device
    policy, policy_kwargs, ppo_kwargs["ent_coef"] = _ppo_policy_and_ent_coef(
        cfg, policy_kwargs, env.action_space
    )
    ppo_model = PPO(
        policy,
        vec_env,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs,
    )
    from .ppo_policy import _layout_key

    if isinstance(env.action_space, spaces.Box):
        width = int(env.action_space.shape[0])
        schema = getattr(env, "action_schema", None)
        if isinstance(schema, ActionSchema):
            if width != schema.width:
                raise ValueError(
                    f"action space width {width} disagrees with declared schema width "
                    f"{schema.width}"
                )
            layout_key = schema.key
        elif width == 1 and not bool(getattr(env, "strategy_actions", False)):
            # The scalar Box belongs to the continuous directional action mode,
            # not to the strategy-overlay coordinate schema.
            layout_key = _layout_key(width)
        else:
            raise ValueError(
                "Box strategy actions require a declarative ActionSchema; "
                "action width alone cannot identify coordinate meanings"
            )
        setattr(ppo_model.policy, "_action_layout_key", layout_key)
        from quant_rl.action_schema import SCHEMA_VERSION

        setattr(ppo_model.policy, "_action_schema_version", SCHEMA_VERSION)
        # The policy itself is excluded from SB3's model data during save;
        # persist the semantics on the model so save/load can recover them.
        setattr(ppo_model, "_action_layout_key", layout_key)
        setattr(ppo_model, "_action_schema_version", SCHEMA_VERSION)
        if isinstance(schema, ActionSchema):
            schema_manifest = schema.to_dict()
            setattr(ppo_model, "_action_schema", schema_manifest)
            setattr(ppo_model.policy, "_action_schema", schema_manifest)
    from .rollout_buffer import install_single_copy_add

    install_single_copy_add(ppo_model.rollout_buffer)
    return _finalize_cuda_policy(ppo_model, device)


def _ppo_policy_and_ent_coef(
    cfg: DictConfig,
    policy_kwargs: dict[str, Any],
    action_space: spaces.Space[Any],
) -> tuple[Any, dict[str, Any], float]:
    """Box overlay: clamp Gaussian std. Discrete: keep categorical entropy coef."""
    ent_coef = float(cfg.ppo.ent_coef)
    if not isinstance(action_space, spaces.Box):
        return "MultiInputPolicy", policy_kwargs, ent_coef
    from .ppo_policy import ClampedStdMultiInputPolicy

    kwargs = dict(policy_kwargs)
    kwargs["log_std_init"] = float(cfg.ppo.get("log_std_init", 0.0))
    kwargs["log_std_min"] = float(cfg.ppo.get("log_std_min", -0.7))
    kwargs["log_std_max"] = float(cfg.ppo.get("log_std_max", 0.0))
    kwargs["mode_prior_coef"] = float(cfg.ppo.get("mode_prior_coef", 1e-3))
    return ClampedStdMultiInputPolicy, kwargs, float(cfg.ppo.get("ent_coef_continuous", 0.001))


def _finalize_cuda_policy(model: Any, device: str | torch.device | None) -> Any:
    """Enable bf16 encoder autocast when the policy lives on CUDA."""
    if device is None:
        return model
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        return model
    enable_extractor_autocast(model.policy.features_extractor)
    return model
