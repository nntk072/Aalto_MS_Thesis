"""Auxiliary self-supervised loss and entropy regularization for RL agents.

This module provides:
1. Auxiliary prediction heads for joint training with RL loss
2. Entropy regularization utilities for improved exploration

The auxiliary task forecasts the cumulative return over the next N bars,
forcing the encoder to capture meaningful market dynamics even when
the RL reward is sparse.

The entropy regularization encourages the policy to maintain diversity
in its actions, improving exploration and robustness.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ReturnPredictionHead(nn.Module):
    """Auxiliary head for predicting future returns.

    This head takes the encoder's latent representation and predicts
    the cumulative return over the next N bars.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent representation from the encoder.
    hidden_dim : int
        Dimension of the hidden layer in the prediction head.
    prediction_horizon : int
        Number of bars ahead to predict (default: 5).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        prediction_horizon: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_horizon),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict future returns from latent representation.

        Parameters
        ----------
        latent : torch.Tensor
            Latent representation of shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Predicted returns of shape (batch_size, prediction_horizon).
        """
        pred: torch.Tensor = self.net(latent)
        return pred


class AuxiliaryLoss:
    """Auxiliary loss combiner for joint training.

    Combines the RL loss with the auxiliary prediction loss.

    Parameters
    ----------
    prediction_head : ReturnPredictionHead
        The auxiliary prediction head.
    aux_weight : float
        Weight for the auxiliary loss (default: 0.1).
    loss_fn : nn.Module
        Loss function for the auxiliary task (default: MSE).
    """

    def __init__(
        self,
        prediction_head: ReturnPredictionHead,
        aux_weight: float = 0.1,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.prediction_head = prediction_head
        self.aux_weight = aux_weight
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def __call__(
        self,
        rl_loss: torch.Tensor,
        latent: torch.Tensor,
        target_returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        rl_loss : torch.Tensor
            The primary RL loss.
        latent : torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        target_returns : torch.Tensor
            Target returns of shape (batch_size, prediction_horizon).

        Returns
        -------
        torch.Tensor
            Combined loss = RL loss + aux_weight * auxiliary loss.
        """
        # Predict future returns
        pred_returns: torch.Tensor = self.prediction_head(latent)

        # Compute auxiliary loss
        aux_loss: torch.Tensor = self.loss_fn(pred_returns, target_returns)

        # Combined loss
        total_loss: torch.Tensor = rl_loss + self.aux_weight * aux_loss

        return total_loss


class CumulativeReturnPredictor:
    """Helper class to compute cumulative returns for auxiliary training.

    This class computes the cumulative return over the next N bars
    from the current state, which can be used as the target for
    the auxiliary prediction head.

    Parameters
    ----------
    horizon : int
        Number of bars ahead to compute cumulative return for.
    """

    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon

    def compute_targets(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute cumulative returns for each position in the sequence.

        Parameters
        ----------
        returns : torch.Tensor
            Per-bar returns of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Cumulative returns of shape (batch_size, seq_len - horizon + 1, horizon).
        """
        batch_size, seq_len = returns.shape

        # Compute cumulative returns using cumsum
        cum_returns = torch.cumsum(returns, dim=1)

        # Get returns for each horizon
        targets = []
        for h in range(1, self.horizon + 1):
            # Cumulative return over h bars: sum of returns[t:t+h]
            # This is cum_returns[t+h] - cum_returns[t]
            if seq_len >= h:
                target_h = cum_returns[:, h:] - cum_returns[:, :-h]
                targets.append(target_h)

        # Stack to get (batch_size, seq_len - horizon, horizon)
        # Pad if needed
        min_len = min(t.shape[1] for t in targets)
        padded_targets = [t[:, :min_len] for t in targets]
        result = torch.stack(padded_targets, dim=-1)

        return result


class EntropyBonus:
    """Entropy-based exploration bonus for RL policies.

    Adds an entropy bonus to the reward to encourage the policy to
    maintain diversity in its action distribution. This is particularly
    useful for continuous action spaces where the policy might converge
    to a deterministic solution.

    The entropy bonus is computed as the negative entropy of the action
    distribution, scaled by a temperature parameter.

    Parameters
    ----------
    action_dim : int
        Dimension of the action space.
    entropy_coef : float
        Weight for the entropy bonus (default: 0.01).
    temperature : float
        Temperature parameter for the softmax (default: 1.0).
    """

    def __init__(
        self,
        action_dim: int = 1,
        entropy_coef: float = 0.01,
        temperature: float = 1.0,
    ) -> None:
        self.action_dim = action_dim
        self.entropy_coef = entropy_coef
        self.temperature = temperature

    def __call__(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy bonus from log probabilities.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities of actions of shape (batch_size, action_dim).

        Returns
        -------
        torch.Tensor
            Entropy bonus of shape (batch_size,).
        """
        # Compute entropy: -sum(p * log(p))
        # With log_probs: -sum(exp(log_p) * log_p) = -sum(p * log_p)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Scale by temperature and coefficient
        bonus = self.entropy_coef * entropy / self.temperature

        return bonus

    def from_action_distribution(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Compute entropy bonus from Gaussian action distribution parameters.

        For a Gaussian distribution, the entropy is:
        H = 0.5 * log(2 * pi * e * sigma^2) = 0.5 * (1 + log(2 * pi) + 2 * log(sigma))

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the Gaussian distribution of shape (batch_size, action_dim).
        log_std : torch.Tensor
            Log standard deviation of shape (batch_size, action_dim).

        Returns
        -------
        torch.Tensor
            Entropy bonus of shape (batch_size,).
        """
        # Entropy of Gaussian: 0.5 * (1 + log(2 * pi) + 2 * log(std))
        # log_std is log(std), so 2 * log(std) = 2 * log_std
        log_2pi = torch.log(2 * torch.tensor(math.pi))
        entropy = 0.5 * (1 + log_2pi + 2 * log_std)

        # Sum over action dimensions and scale
        entropy = torch.sum(entropy, dim=-1)
        bonus = self.entropy_coef * entropy / self.temperature

        return bonus


class JointEntropyLoss:
    """Combined RL loss with entropy regularization.

    This combines the RL policy loss with an entropy bonus to encourage
    exploration. This is particularly useful for continuous action spaces.

    Parameters
    ----------
    entropy_bonus : EntropyBonus
        The entropy bonus calculator.
    """

    def __init__(self, entropy_bonus: EntropyBonus) -> None:
        self.entropy_bonus = entropy_bonus

    def __call__(
        self,
        rl_loss: torch.Tensor,
        log_probs: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
        log_std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined loss with entropy regularization.

        Parameters
        ----------
        rl_loss : torch.Tensor
            The primary RL loss.
        log_probs : torch.Tensor | None
            Log probabilities for discrete actions.
        mean : torch.Tensor | None
            Mean for Gaussian actions.
        log_std : torch.Tensor | None
            Log std for Gaussian actions.

        Returns
        -------
        torch.Tensor
            Combined loss = RL loss - entropy bonus.
        """
        # Compute entropy bonus
        if log_probs is not None:
            entropy_bonus = self.entropy_bonus(log_probs)
        elif mean is not None and log_std is not None:
            entropy_bonus = self.entropy_bonus.from_action_distribution(mean, log_std)
        else:
            entropy_bonus = torch.zeros(rl_loss.shape[0], device=rl_loss.device)

        # Combined loss (note: we add because entropy_bonus is already positive)
        total_loss = rl_loss - entropy_bonus

        return total_loss
