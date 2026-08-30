"""Chain E Step 0 — wire the (previously unused) auxiliary return-prediction head.

SB3 has no native support for extra loss terms, so instead of subclassing
PPO/SAC's ``train()`` (fragile across versions), the auxiliary head and the
policy encoder are co-trained by a *separate* optimizer inside an SB3
callback at rollout boundaries: the encoder receives the auxiliary gradient
through the shared latent, while SB3's own update handles the RL loss.

Supervised windows are built **directly from the env's feature matrix and
closes** — not from the rollout buffer — so window↔target alignment stays
exact even across env resets: the observation window ending at bar ``t``
(``features[t-T:t]``, matching ``TradingEnv._get_obs``) is paired with the
per-bar log returns of bars ``t..t+N-1`` (strictly after the window, hence
causal as a supervised target). ``aux_weight`` scales the auxiliary MSE
gradient; the default 0.0 keeps the feature fully opt-in.

Gate (see PLANS/05_chain_e_capacity_ladder.md): keep the auxiliary loss only
if held-out Sharpe improves vs. an ``aux_weight=0`` control at the same seed
count — a falling aux loss alone is necessary, not sufficient.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray
from stable_baselines3.common.callbacks import BaseCallback

from quant_rl.models.auxiliary import AuxiliaryLoss, ReturnPredictionHead

logger = logging.getLogger(__name__)


def build_supervised_windows(
    features: NDArray[np.float64],
    closes: NDArray[np.float64],
    obs_window: int,
    horizon: int,
    max_windows: int = 256,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Build aligned (window, future-return) pairs from the feature matrix.

    Pair ``i`` is ``(features[i-obs_window:i], log_ret[i:i+horizon])`` — the
    same window layout ``TradingEnv`` produces at bar ``i``, with targets
    strictly after the window end. Windows/targets containing NaNs (indicator
    warmup, horizon past the sample end) are dropped.

    Returns
    -------
    (seq, y):
        ``seq`` has shape ``(B, obs_window, F)``; ``y`` has shape
        ``(B, horizon)`` (per-bar log returns, matching the head output).
    """
    if rng is None:
        rng = np.random.default_rng()
    features = np.asarray(features, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    if n <= obs_window + horizon or features.shape[0] != n:
        empty_seq = np.zeros((0, obs_window, features.shape[1]), np.float32)
        return empty_seq, np.zeros((0, horizon), np.float32)

    log_ret = np.log(closes)
    candidates: NDArray[np.int64] = np.arange(obs_window, n - horizon)
    if len(candidates) > max_windows:
        candidates = rng.choice(candidates, size=max_windows, replace=False).astype(np.int64)
    candidates = np.sort(candidates)

    seq = np.stack([features[i - obs_window : i] for i in candidates])
    y = np.stack([log_ret[i : i + horizon] for i in candidates])

    valid = np.isfinite(seq).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    return seq[valid].astype(np.float32), y[valid].astype(np.float32)


class AuxiliaryTrainerCallback(BaseCallback):
    """Co-train ``ReturnPredictionHead`` + encoder at each rollout end.

    Parameters
    ----------
    prediction_horizon:
        Bars ahead for the cumulative-return target (N).
    aux_weight:
        Weight on the auxiliary MSE term (0.0 disables training entirely).
    lr:
        Learning rate for the auxiliary optimizer (head + encoder params).
    grad_steps:
        Gradient steps taken per rollout.
    """

    def __init__(
        self,
        prediction_horizon: int = 5,
        aux_weight: float = 0.1,
        lr: float = 1e-4,
        grad_steps: int = 4,
        batch_windows: int = 256,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.prediction_horizon = prediction_horizon
        self.aux_weight = aux_weight
        self.lr = lr
        self.grad_steps = grad_steps
        self.batch_windows = batch_windows
        self.aux_loss_fn: AuxiliaryLoss | None = None
        self.last_aux_loss: float | None = None
        self.rng = np.random.default_rng()

    def _on_step(self) -> bool:
        """Per-step hook — the auxiliary training happens at rollout end."""
        return True

    # -- setup ------------------------------------------------------------
    def _init_callback(self) -> None:
        encoder = self.model.policy.features_extractor
        # Encoder output dim is features_dim (= latent_dim + ACCOUNT_DIM);
        # the head sits on the full extractor output, not latent_dim alone.
        out_dim = int(encoder.features_dim)
        head = ReturnPredictionHead(latent_dim=out_dim, prediction_horizon=self.prediction_horizon)
        head.to(self.model.device)
        self.head = head
        self.aux_loss_fn = AuxiliaryLoss(head, aux_weight=self.aux_weight)
        params = list(head.parameters()) + list(encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        logger.info(
            "Auxiliary trainer attached: horizon=%d aux_weight=%.3f features_dim=%d",
            self.prediction_horizon,
            self.aux_weight,
            out_dim,
        )

    # -- data ---------------------------------------------------------------
    def _env_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """(features, closes) from the first underlying TradingEnv."""
        env = self.model.get_env()
        if env is None:
            return None
        inner = getattr(env, "envs", [None])[0]
        while inner is not None and hasattr(inner, "env"):
            inner = inner.env
        inner = getattr(inner, "unwrapped", inner)
        features = getattr(inner, "features", None)
        bars = getattr(inner, "bars", None)
        if features is None or bars is None or len(bars) != len(features):
            return None
        return features.to_numpy(dtype=np.float64), bars["close"].to_numpy(dtype=np.float64)

    # -- training -----------------------------------------------------------
    def _on_rollout_end(self) -> None:
        if self.aux_weight <= 0.0 or self.aux_loss_fn is None:
            return
        data = self._env_data()
        if data is None:
            return
        feats, closes = data

        obs_space = self.model.observation_space
        seq_shape = obs_space["seq"].shape  # type: ignore[index]
        acc_shape = obs_space["account"].shape  # type: ignore[index]
        seq, y = build_supervised_windows(
            feats,
            closes,
            obs_window=int(seq_shape[0]),
            horizon=self.prediction_horizon,
            max_windows=self.batch_windows,
            rng=self.rng,
        )
        if len(seq) < 2:
            return

        seq_t = torch.as_tensor(seq, device=self.model.device)
        # Account state is path-dependent (unreconstructible offline); zeros
        # are the neutral input for the auxiliary task's account branch.
        acc_t = torch.zeros(
            (len(seq), int(acc_shape[0])),
            device=self.model.device,
        )

        policy = self.model.policy
        loss = None
        for _ in range(self.grad_steps):
            self.optimizer.zero_grad()
            obs_dict: dict[str, NDArray[np.float32]] = {
                "seq": seq_t.cpu().numpy(),
                "account": acc_t.cpu().numpy(),
            }
            obs_tensor, _ = policy.obs_to_tensor(obs_dict)
            latent = policy.extract_features(obs_tensor, policy.features_extractor)
            zero_rl = torch.zeros((), device=self.model.device)
            loss = self.aux_loss_fn(zero_rl, latent, torch.as_tensor(y, device=self.model.device))
            loss.backward()  # type: ignore[no-untyped-call]
            self.optimizer.step()

        if loss is not None:
            self.last_aux_loss = float(loss.item())
        if self.model.logger is not None and self.last_aux_loss is not None:
            self.model.logger.record("aux/mse_loss", self.last_aux_loss)
