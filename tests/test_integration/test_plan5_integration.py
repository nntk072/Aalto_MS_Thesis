"""Integration tests for plan 5 (SAC agent + auxiliary loss + env wiring)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


class TestPlan5SACIntegration:
    """End-to-end SAC rollout against TradingEnv with time-decay reward."""

    @pytest.fixture
    def sac_setup(self) -> Any:
        """Build a continuous-action env and an SAC agent wired to it."""
        from omegaconf import OmegaConf

        from quant_rl.models.agent import build_agent
        from tests.test_models.test_sac_agent import make_env

        cfg = OmegaConf.create(
            {
                "env": {"obs_window": 10},
                "sac": {
                    "learning_rate": 3e-4,
                    "buffer_size": 2000,
                    "batch_size": 16,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "ent_coef": 0.01,
                    "learning_starts": 100,
                },
                "ppo": {},
            }
        )
        env = make_env(continuous=True, n_bars=160)
        model = build_agent(env, cfg, arch="tcn", algo="sac")
        return model

    def test_sac_rollout_produces_rewards_and_equity(self, sac_setup: Any) -> None:
        """A short deterministic rollout yields finite rewards and equity."""
        model = sac_setup
        vec_env = model.get_env()
        assert vec_env is not None

        obs = vec_env.reset()
        rewards = []
        for _ in range(20):
            action, _ = model.predict(obs, deterministic=True)
            action_arr = np.asarray(action).reshape(-1)
            obs, reward, done, _ = vec_env.step(
                np.array([[float(action_arr[0])]], dtype=np.float32)
            )
            rewards.append(float(reward[0]))
            if done.any():
                break

        assert len(rewards) > 0
        assert all(np.isfinite(r) for r in rewards)

    def test_sweep_reward_receives_time_decay(self) -> None:
        """CompositeReward accepts minutes_since_open for the decay penalty."""
        from quant_rl.envs.sweep_reward import CompositeReward, SweepConfirmationReward

        composite = CompositeReward(
            sweep_reward=SweepConfirmationReward(alpha=0.1, beta=0.01, hold_bars=1),
            dsr_weight=0.5,
            sweep_weight=0.5,
        )
        # No sweep context -> DSR fallback path, but the kwarg must be accepted.
        r_no_ctx = composite(
            0.001,
            daily_loss=10.0,
            daily_loss_limit=5000.0,
            initial_balance=100_000.0,
            breach=False,
            minutes_since_open=35.0,
        )
        assert np.isfinite(r_no_ctx)

        # Full sweep context with a late entry triggers the time-decay penalty.
        r_ctx = composite(
            0.001,
            cost=0.0,
            price=110.0,
            london_high=112.0,
            london_low=92.0,
            asian_high=108.0,
            asian_low=96.0,
            minutes_since_open=45.0,
            position_changed=False,
            daily_loss=10.0,
            daily_loss_limit=5000.0,
            initial_balance=100_000.0,
            breach=False,
        )
        assert np.isfinite(r_ctx)
        # Late entries must score lower than early ones via beta * T_t.
        r_early = composite(
            0.001,
            cost=0.0,
            price=110.0,
            london_high=112.0,
            london_low=92.0,
            asian_high=108.0,
            asian_low=96.0,
            minutes_since_open=5.0,
            position_changed=False,
            daily_loss=10.0,
            daily_loss_limit=5000.0,
            initial_balance=100_000.0,
            breach=False,
        )
        assert r_early > r_ctx

    def test_auxiliary_loss_integrates_with_encoder_latent(self) -> None:
        """AuxiliaryLoss consumes encoder latents of the documented size."""
        import torch

        from quant_rl.models.auxiliary import AuxiliaryLoss, ReturnPredictionHead

        latent_dim, account_dim = 128, 5
        features_dim = latent_dim + account_dim

        head = ReturnPredictionHead(latent_dim=features_dim, prediction_horizon=5)
        aux = AuxiliaryLoss(head, aux_weight=0.1)

        rl_loss = torch.tensor(0.5)
        latent = torch.randn(8, features_dim, requires_grad=True)
        targets = torch.randn(8, 5)

        total = aux(rl_loss, latent, targets)
        total.backward()
        assert latent.grad is not None and torch.isfinite(latent.grad).all()
