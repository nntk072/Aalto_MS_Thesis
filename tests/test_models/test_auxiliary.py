"""Tests for the auxiliary self-supervised loss (plan 5)."""

from __future__ import annotations

import pytest
import torch

from quant_rl.models.auxiliary import (
    AuxiliaryLoss,
    CumulativeReturnPredictor,
    EntropyBonus,
    JointEntropyLoss,
    ReturnPredictionHead,
)


class TestReturnPredictionHead:
    """Tests for ReturnPredictionHead."""

    def test_output_shape(self) -> None:
        head = ReturnPredictionHead(latent_dim=32, prediction_horizon=5)
        latent = torch.randn(8, 32)
        pred = head(latent)
        assert pred.shape == (8, 5)

    def test_backward_pass(self) -> None:
        head = ReturnPredictionHead(latent_dim=16, hidden_dim=32, prediction_horizon=3)
        pred = head(torch.randn(4, 16))
        loss = torch.nn.functional.mse_loss(pred, torch.zeros_like(pred))
        loss.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestAuxiliaryLoss:
    """Tests for the combined RL + auxiliary loss."""

    def test_combined_loss_adds_aux_term(self) -> None:
        head = ReturnPredictionHead(latent_dim=16, prediction_horizon=5)
        aux = AuxiliaryLoss(head, aux_weight=0.1)

        rl_loss = torch.tensor(1.0)
        latent = torch.randn(4, 16, requires_grad=True)
        targets = torch.randn(4, 5)

        total = aux(rl_loss, latent, targets)
        # Total must differ from the raw RL loss by the weighted aux term.
        assert not torch.allclose(total, rl_loss)

    def test_zero_weight_returns_rl_loss_only(self) -> None:
        head = ReturnPredictionHead(latent_dim=16, prediction_horizon=5)
        aux = AuxiliaryLoss(head, aux_weight=0.0)

        rl_loss = torch.tensor(2.0)
        total = aux(rl_loss, torch.randn(4, 16), torch.randn(4, 5))
        assert torch.allclose(total, rl_loss)


class TestCumulativeReturnPredictor:
    """Tests for cumulative return target computation."""

    def test_shapes(self) -> None:
        predictor = CumulativeReturnPredictor(horizon=3)
        returns = torch.randn(4, 20)
        targets = predictor.compute_targets(returns)
        assert targets.shape[0] == 4
        assert targets.shape[-1] == 3


class TestEntropyBonus:
    """Tests for entropy bonus utilities."""

    def test_log_prob_bonus_shape(self) -> None:
        bonus_fn = EntropyBonus(action_dim=1, entropy_coef=0.01)
        log_probs = torch.log(torch.full((4, 1), 0.5))
        bonus = bonus_fn(log_probs)
        assert bonus.shape == (4,)
        assert torch.isfinite(bonus).all()

    def test_gaussian_bonus_shape(self) -> None:
        bonus_fn = EntropyBonus(action_dim=2, entropy_coef=0.01)
        mean = torch.zeros(6, 2)
        log_std = torch.zeros(6, 2)
        bonus = bonus_fn.from_action_distribution(mean, log_std)
        assert bonus.shape == (6,)
        assert torch.isfinite(bonus).all()

    def test_higher_std_gives_larger_entropy_bonus(self) -> None:
        bonus_fn = EntropyBonus(action_dim=1, entropy_coef=0.01)
        low = bonus_fn.from_action_distribution(torch.zeros(1, 1), torch.full((1, 1), -2.0))
        high = bonus_fn.from_action_distribution(torch.zeros(1, 1), torch.full((1, 1), 2.0))
        assert high.item() > low.item()


class TestJointEntropyLoss:
    """Tests for the joint RL + entropy loss."""

    def test_combines_with_gaussian_params(self) -> None:
        joint = JointEntropyLoss(EntropyBonus(action_dim=1))
        rl_loss = torch.ones(4)
        mean = torch.zeros(4, 1)
        log_std = torch.zeros(4, 1)

        total = joint(rl_loss, mean=mean, log_std=log_std)
        assert total.shape == (4,)
        # Entropy bonus is subtracted, so total < rl_loss for positive entropy.
        assert (total <= rl_loss + 1e-9).all()

    def test_no_distribution_args_yields_rl_loss(self) -> None:
        joint = JointEntropyLoss(EntropyBonus(action_dim=1))
        rl_loss = torch.ones(4)
        total = joint(rl_loss)
        assert torch.allclose(total, rl_loss)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
