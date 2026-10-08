"""PPO Gaussian log_std clamp for Box overlay actions."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from quant_rl.envs.trading_env import TradingEnv
from quant_rl.models.agent import build_agent
from quant_rl.models.ppo_policy import (
    attach_mode_prior,
    clip_policy_log_std,
    install_post_update_std_clip,
    mode_index,
)
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


def _mode_grad(mean: torch.Tensor, coef: float, dims: tuple[int, ...]) -> torch.Tensor:
    attach_mode_prior(mean, coef, dims)
    mean.sum().backward()  # type: ignore[no-untyped-call]
    assert mean.grad is not None
    return mean.grad


def _six() -> torch.Tensor:
    # direction, intensity, stop, risk, target, exit
    return torch.tensor(
        [[0.0, 0.2, 1.0, 0.1, 2.0, 3.0], [0.0, -0.2, 3.0, -0.1, 4.0, 5.0]],
        requires_grad=True,
    )


def test_mode_index_map_rejects_unknown_width_and_name() -> None:
    assert mode_index("stop", 5) == 1
    assert mode_index("target", 5) == 3
    assert mode_index("exit", 5) == 4
    assert mode_index("direction", 6, has_direction=True) == 0
    assert mode_index("stop", 6, has_direction=True) == 2
    assert mode_index("target", 6, has_direction=True) == 4
    assert mode_index("exit", 6, has_direction=True) == 5
    assert mode_index("risk", 5) == 2
    assert mode_index("risk", 6, has_direction=True) == 3
    with pytest.raises(ValueError):
        mode_index("stop", 4)
    with pytest.raises(ValueError):
        mode_index("intensity", 6, has_direction=True)
    with pytest.raises(ValueError):
        mode_index("direction", 5)


def test_mode_prior_on_stop_and_target_only() -> None:
    coef = 1e-3
    mean = _six()
    grad = _mode_grad(mean, coef, (2, 4))
    stop_extra = 2.0 * coef * 2.0 / 2.0
    target_extra = 2.0 * coef * 3.0 / 2.0
    assert grad[:, 2].tolist() == pytest.approx([1.0 + stop_extra, 1.0 + stop_extra])
    assert grad[:, 4].tolist() == pytest.approx([1.0 + target_extra, 1.0 + target_extra])
    for column in (0, 1, 3, 5):
        assert grad[:, column].tolist() == pytest.approx([1.0, 1.0])


def test_mode_prior_on_stop_only() -> None:
    coef = 1e-3
    mean = _six()
    grad = _mode_grad(mean, coef, (2,))
    stop_extra = 2.0 * coef * 2.0 / 2.0
    assert grad[:, 2].tolist() == pytest.approx([1.0 + stop_extra, 1.0 + stop_extra])
    assert grad[:, 5].tolist() == pytest.approx([1.0, 1.0])


def test_mode_prior_on_exit_only() -> None:
    coef = 1e-3
    mean = _six()
    grad = _mode_grad(mean, coef, (5,))
    exit_extra = 2.0 * coef * 4.0 / 2.0
    assert grad[:, 5].tolist() == pytest.approx([1.0 + exit_extra, 1.0 + exit_extra])
    assert grad[:, 2].tolist() == pytest.approx([1.0, 1.0])


def test_mode_prior_empty_set_adds_nothing() -> None:
    mean = _six()
    grad = _mode_grad(mean, 1e-3, ())
    assert grad.reshape(-1).tolist() == pytest.approx([1.0] * 12)


def test_mode_prior_coef_zero_matches_the_unhooked_backward() -> None:
    values = [[1.0, 0.0, 1.0, 0.0, 1.0, 2.0]]
    hooked = torch.tensor(values, requires_grad=True)
    plain = torch.tensor(values, requires_grad=True)
    attach_mode_prior(hooked, 0.0, (0, 2, 4, 5))
    hooked.sum().backward()  # type: ignore[no-untyped-call]
    plain.sum().backward()  # type: ignore[no-untyped-call]
    assert hooked.grad is not None and plain.grad is not None
    assert hooked.grad.reshape(-1).tolist() == pytest.approx(plain.grad.reshape(-1).tolist())


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


def test_post_update_clip_logs_the_clamped_std() -> None:
    class _Pol:
        def __init__(self) -> None:
            self.log_std = torch.nn.Parameter(torch.tensor([0.5]))

    class _Logger:
        def __init__(self) -> None:
            self.values: dict[str, float] = {}

        def record(self, key: str, value: float) -> None:
            self.values[key] = float(value)

    class _Model:
        def __init__(self) -> None:
            self.policy = _Pol()
            self.logger = _Logger()

        def train(self) -> None:
            self.logger.record("train/std", 9.0)

    model = _Model()
    install_post_update_std_clip(model, -0.7, 0.0)
    model.train()
    assert float(model.policy.log_std.detach().item()) == pytest.approx(0.0)
    assert model.logger.values["train/std"] == pytest.approx(1.0)


def test_post_update_clip_logs_mode_means() -> None:
    class _Pol:
        def __init__(self) -> None:
            self.log_std = torch.nn.Parameter(torch.tensor([0.0]))
            self.reset_called = False

        def reset_mode_prior_stats(self) -> None:
            self.reset_called = True

        def mode_prior_metrics(self) -> dict[str, float]:
            return {
                "train/mean_exit": 0.25,
                "train/mean_direction": -0.5,
                "train/mode_prior": 1e-4,
            }

    class _Logger:
        def __init__(self) -> None:
            self.values: dict[str, float] = {}

        def record(self, key: str, value: float) -> None:
            self.values[key] = float(value)

    class _Model:
        def __init__(self) -> None:
            self.policy = _Pol()
            self.logger = _Logger()

        def train(self) -> None:
            return None

    model = _Model()
    install_post_update_std_clip(model, -0.7, 0.0)
    model.train()
    assert model.policy.reset_called
    assert model.logger.values["train/mean_exit"] == pytest.approx(0.25)
    assert model.logger.values["train/mean_direction"] == pytest.approx(-0.5)
    assert model.logger.values["train/mode_prior"] == pytest.approx(1e-4)


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


def test_policy_terms_follow_the_active_names() -> None:
    env = TradingEnv(
        make_env(continuous=True).bars,
        make_env(continuous=True).features,
        strategy_actions=True,
        agent_direction_control=True,
        allow_agent_sl_mode=False,
        obs_window=10,
        episodic=True,
    )
    model = build_agent(env, _ppo_cfg(), arch="tcn", algo="ppo")
    policy = model.policy
    assert policy._prior_dims(6) == ()
    policy.set_active_prior({"stop", "target"})
    assert policy._prior_dims(6) == (2, 4)
    policy.reset_mode_prior_stats()
    mean = torch.zeros(4, 6)
    mean[:, 2] = 2.0
    mean[:, 4] = -1.0
    policy._note_mode_means(mean)
    metrics = policy.mode_prior_metrics()
    coef = float(policy.mode_prior_coef)
    assert metrics["prior_active_stop"] == 1.0
    assert metrics["prior_active_target"] == 1.0
    assert metrics["prior_active_exit"] == 0.0
    assert metrics["prior_active_direction"] == 0.0
    assert "train/mode_prior_exit" not in metrics
    assert "train/mode_prior_direction" not in metrics
    assert metrics["train/mode_prior_stop"] == pytest.approx(coef * 4.0)
    assert metrics["train/mode_prior_target"] == pytest.approx(coef * 1.0)
    assert metrics["train/mode_prior"] == pytest.approx(coef * 5.0)
    policy.set_active_prior({"risk"})
    assert policy._prior_dims(6) == (3,)
    with pytest.raises(ValueError):
        policy.set_active_prior({"intensity"})


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


def test_build_agent_width_11_multi_tp_sl_tp_simplex() -> None:
    """A4b/A4a entry-FSM variants produce action width 11.

    5 base + sl_mode + tp_mode + 2*multi_tp + 2*simplex = 11. The env step
    decoder handles it, but the policy layout table did not, so build_agent
    raised ValueError: unsupported action width 11.
    """
    env = make_env(continuous=True)
    env = TradingEnv(
        env.bars,
        env.features,
        obs_window=10,
        strategy_actions=True,
        allow_agent_sl_mode=True,
        allow_agent_tp_mode=True,
        allow_multi_tp=True,
        allow_simplex=True,
        entry_state_machine=True,
        entry_state_observation=True,
        episodic=True,
    )
    assert env.action_space.shape == (11,)
    model = build_agent(env, _ppo_cfg(), arch="tcn", algo="ppo")
    assert model.policy._action_layout_key == "11"
    assert mode_index("stop", 11) == 1
    assert mode_index("risk", 11) == 2
    assert mode_index("tp1_sel", 11) == 3
    assert mode_index("tp2_sel", 11) == 4
    assert mode_index("tp3_sel", 11) == 5
    assert mode_index("sl_mode", 11) == 6
    assert mode_index("tp_mode", 11) == 7
    assert mode_index("z1", 11) == 8
    assert mode_index("z2", 11) == 9
    assert mode_index("exit", 11) == 10


def test_policy_account_value_weight_matches_observation_width() -> None:
    env = make_env(continuous=True)
    env = TradingEnv(
        env.bars,
        env.features,
        obs_window=10,
        strategy_actions=True,
        entry_state_machine=True,
        entry_state_observation=True,
        episodic=True,
    )
    model = build_agent(env, _ppo_cfg(), arch="tcn", algo="ppo")
    account_dim = env.observation_space["account"].shape[0]  # type: ignore[index]
    assert model.policy.account_value_weight.shape == (account_dim,)
