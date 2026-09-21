"""PPO Gaussian log_std clamp for Box overlay actions."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from quant_rl.models.agent import build_agent
from quant_rl.models.ppo_policy import clip_policy_log_std
from quant_rl.train.callbacks import ClipLogStdCallback
from tests.test_models.test_sac_agent import make_env


def _log_std_range(log_std: Any) -> tuple[float, float]:
    detached = log_std.detach()
    return float(detached.min().item()), float(detached.max().item())


def _ppo_cfg() -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "env": {"obs_window": 10},
            "ppo": {
                "n_steps": 64,
                "batch_size": 32,
                "n_epochs": 1,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "ent_coef_continuous": 0.001,
                "log_std_init": 0.0,
                "log_std_min": -2.0,
                "log_std_max": 0.0,
            },
        }
    )


def test_clip_policy_log_std_clamps_and_skips_discrete() -> None:
    class _Pol:
        def __init__(self) -> None:
            self.log_std = torch.nn.Parameter(torch.tensor([5.0, -4.0]))

    pol = _Pol()
    assert clip_policy_log_std(pol, -2.0, 0.0)
    lo, hi = _log_std_range(pol.log_std)
    assert hi <= 0.0
    assert lo >= -2.0
    assert clip_policy_log_std(object(), -2.0, 0.0) is False


def test_clip_log_std_callback_clamps_exploded_std() -> None:
    class _Pol:
        def __init__(self) -> None:
            self.log_std = torch.nn.Parameter(torch.tensor([5.0, -4.0]))

    class _Model:
        policy = _Pol()

    cb = ClipLogStdCallback(log_std_min=-2.0, log_std_max=0.0)
    cb.model = _Model()  # type: ignore[assignment]
    cb._on_rollout_start()
    lo, hi = _log_std_range(cb.model.policy.log_std)
    assert hi <= 0.0
    assert lo >= -2.0


def test_build_agent_box_uses_clamped_policy() -> None:
    from quant_rl.models.ppo_policy import ClampedStdMultiInputPolicy

    env = make_env(continuous=True)
    model = build_agent(env, _ppo_cfg(), arch="tcn", algo="ppo")
    assert isinstance(model.policy, ClampedStdMultiInputPolicy)
    assert model.ent_coef == pytest.approx(0.001)
    with torch.no_grad():
        model.policy.log_std.fill_(5.0)
    obs, _ = env.reset()
    model.predict(obs, deterministic=False)
    _, hi = _log_std_range(model.policy.log_std)
    assert hi <= 0.0 + 1e-6


def test_build_agent_discrete_keeps_categorical_ent_coef() -> None:
    env = make_env(continuous=False)
    model = build_agent(env, _ppo_cfg(), arch="tcn", algo="ppo")
    assert model.ent_coef == pytest.approx(0.01)
    assert getattr(model.policy, "log_std", None) is None
