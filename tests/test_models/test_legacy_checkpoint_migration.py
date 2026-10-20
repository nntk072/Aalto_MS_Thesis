"""Explicitly load the archived 4D policy and add its structural-exit action."""

import importlib

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from stable_baselines3 import PPO

from decisioning.actions import build_action_schema
from evaluation.reanalysis.runner import load_checkpoint
from learning.rl.agent import LegacyFourToFiveActionAdapter, validate_model_action_schema
from learning.rl.checkpoint import _install_pre_reorganization_checkpoint_modules
from learning.rl.ppo import LegacyClampedStdMultiInputPolicy
from models.encoders import LegacyTCNEncoderV1


class _LegacyObservationEnv(gym.Env[dict[str, npt.NDArray[np.float32]], npt.NDArray[np.float32]]):
    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "seq": spaces.Box(-1.0, 1.0, (8, 6), dtype=np.float32),
                "account": spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, object]]:
        super().reset(seed=seed)
        return {
            "seq": np.zeros((8, 6), dtype=np.float32),
            "account": np.zeros(6, dtype=np.float32),
        }, {}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[dict[str, npt.NDArray[np.float32]], float, bool, bool, dict[str, object]]:
        obs, info = self.reset()
        return obs, 0.0, True, False, info


def test_legacy4_checkpoint_loads_and_adapter_appends_structural_exit(tmp_path):
    env = _LegacyObservationEnv()
    model = PPO(
        LegacyClampedStdMultiInputPolicy,
        env,
        n_steps=2,
        batch_size=2,
        device="cpu",
        policy_kwargs={
            "features_extractor_class": LegacyTCNEncoderV1,
            "features_extractor_kwargs": {
                "seq_len": 8,
                "n_features": 6,
                "latent_dim": 4,
                "channels": (4, 4, 4),
                "dropout": 0.0,
            },
        },
    )
    path = tmp_path / "legacy4.zip"
    model.save(path)

    loaded = load_checkpoint(path, legacy_layout_key="legacy4_no_exit_v1")
    migration = validate_model_action_schema(
        loaded,
        build_action_schema(),
        allow_legacy=True,
        legacy_layout_key="legacy4_no_exit_v1",
    )
    assert migration == "legacy4_no_exit_v1_to_overlay-action-v2-5"

    obs, _ = env.reset()
    action, _ = LegacyFourToFiveActionAdapter(loaded).predict(obs, deterministic=True)
    assert action.shape == (5,)
    assert action[-1] == -1.0


def test_pre_reorganization_checkpoint_modules_resolve_to_current_classes():
    _install_pre_reorganization_checkpoint_modules()

    legacy_encoder = importlib.import_module("quant_rl.models.encoder")
    legacy_policy = importlib.import_module("quant_rl.models.ppo_policy")
    legacy_buffer = importlib.import_module("quant_rl.models.rollout_buffer")

    from learning.rl.ppo import ClampedStdMultiInputPolicy
    from learning.rl.rollout_buffer import DeviceDictRolloutBuffer
    from models.encoders import TCNEncoder

    assert getattr(legacy_encoder, "TCNEncoder") is TCNEncoder
    assert getattr(legacy_policy, "ClampedStdMultiInputPolicy") is ClampedStdMultiInputPolicy
    assert getattr(legacy_buffer, "DeviceDictRolloutBuffer") is DeviceDictRolloutBuffer
