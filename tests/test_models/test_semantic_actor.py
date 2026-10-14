"""Contracts for the semantic-head Gaussian PPO treatment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from quant_rl.action_schema import build_action_schema
from quant_rl.models.agent import validate_model_observation_schema
from quant_rl.models.encoder import TCNEncoder
from quant_rl.models.semantic_actor import (
    SemanticActorHeads,
    SemanticClampedStdMultiInputPolicy,
    semantic_action_groups,
)
from quant_rl.observation_schema import OBSERVATION_SCHEMA_V2, build_observation_schema


class _OverlayEnv(gym.Env[dict[str, npt.NDArray[np.float32]], npt.NDArray[np.float32]]):
    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "seq": spaces.Box(-1.0, 1.0, (8, 3), dtype=np.float32),
                "seq_mask": spaces.Box(0.0, 1.0, (8,), dtype=np.float32),
                "account": spaces.Box(-10.0, 10.0, (6,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (12,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, Any]]:
        super().reset(seed=seed)
        return {
            "seq": np.zeros((8, 3), dtype=np.float32),
            "seq_mask": np.ones(8, dtype=np.float32),
            "account": np.zeros(6, dtype=np.float32),
        }, {}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[dict[str, npt.NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        observation, _ = self.reset()
        return observation, 0.0, True, False, {}


COORDINATES = build_action_schema(
    has_direction=True,
    has_sl_mode=True,
    has_tp_mode=True,
    has_multi_tp=True,
    has_simplex=True,
).names


def test_semantic_groups_cover_action_schema_without_reordering() -> None:
    groups = semantic_action_groups(COORDINATES)
    flattened = [index for indices in groups.values() for index in indices]
    assert sorted(flattened) == list(range(len(COORDINATES)))
    assert groups["entry"] == (0, 1)
    assert groups["management"] == (11,)
    heads = SemanticActorHeads(16, COORDINATES)
    output = heads(torch.randn(5, 16))
    assert output.shape == (5, 12)


def test_each_semantic_head_receives_gradients() -> None:
    actor = SemanticActorHeads(16, COORDINATES)
    actor(torch.randn(4, 16)).square().mean().backward()
    for head in actor.heads.values():
        assert any(parameter.grad is not None for parameter in head.parameters())


def test_semantic_ppo_has_finite_distribution_and_roundtrips(tmp_path) -> None:
    env = _OverlayEnv()
    model = PPO(
        SemanticClampedStdMultiInputPolicy,
        env,
        policy_kwargs={
            "semantic_action_coordinates": COORDINATES,
            "net_arch": {"pi": [32], "vf": [32]},
            "features_extractor_class": TCNEncoder,
            "features_extractor_kwargs": {"seq_len": 8, "n_features": 3, "account_dim": 6},
        },
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        device="cpu",
    )
    obs, _ = env.reset()
    tensor_obs = {key: torch.as_tensor(value).unsqueeze(0) for key, value in obs.items()}
    distribution = model.policy.get_distribution(tensor_obs)
    log_prob = distribution.log_prob(torch.zeros((1, 12)))
    entropy = distribution.entropy()
    assert entropy is not None
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()

    checkpoint = tmp_path / "semantic_policy.zip"
    observation_schema = build_observation_schema(OBSERVATION_SCHEMA_V2)
    setattr(model, "_observation_schema", observation_schema.to_dict())
    model.save(checkpoint)
    loaded = PPO.load(checkpoint, device="cpu")
    assert isinstance(loaded.policy, SemanticClampedStdMultiInputPolicy)
    assert loaded.policy.semantic_action_coordinates == COORDINATES
    validate_model_observation_schema(loaded, observation_schema)
    predicted, _ = loaded.predict(obs, deterministic=True)
    assert predicted.shape == (12,)
