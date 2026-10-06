"""PPO Gaussian policy with clamped log_std for Box overlay actions."""

from __future__ import annotations

from typing import Any

import torch

try:
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy

    _SB3_AVAILABLE = True
except ImportError:
    MultiInputActorCriticPolicy = object  # type: ignore[assignment,misc]
    _SB3_AVAILABLE = False


# Layout keys: width plus which optional dims are present.
#   5      -> [intensity, stop, risk, target, exit]
#   6_dir  -> [direction, intensity, stop, risk, target, exit]
#   6_sl   -> [intensity, stop, risk, target, sl_mode, exit]
#   7      -> [direction, intensity, stop, risk, target, sl_mode, exit]
_MODE_INDEX: dict[str, dict[str, int]] = {
    "5": {"stop": 1, "risk": 2, "target": 3, "exit": 4},
    "6_dir": {"direction": 0, "stop": 2, "risk": 3, "target": 4, "exit": 5},
    "6_sl": {"stop": 1, "risk": 2, "target": 3, "sl_mode": 4, "exit": 5},
    "6_tp": {"stop": 1, "risk": 2, "target": 3, "tp_mode": 4, "exit": 5},
    "7": {"direction": 0, "stop": 2, "risk": 3, "target": 4, "sl_mode": 5, "exit": 6},
    "7_tp": {"direction": 0, "stop": 2, "risk": 3, "target": 4, "tp_mode": 5, "exit": 6},
    "8": {
        "direction": 0,
        "stop": 2,
        "risk": 3,
        "target": 4,
        "sl_mode": 5,
        "tp_mode": 6,
        "exit": 7,
    },
    "10": {
        "direction": 0,
        "stop": 2,
        "risk": 3,
        "tp1_sel": 4,
        "tp2_sel": 5,
        "tp3_sel": 6,
        "target": 4,
        "sl_mode": 7,
        "tp_mode": 8,
        "exit": 9,
    },
    "12": {
        "direction": 0,
        "stop": 2,
        "risk": 3,
        "tp1_sel": 4,
        "tp2_sel": 5,
        "tp3_sel": 6,
        "target": 4,
        "sl_mode": 7,
        "tp_mode": 8,
        "z1": 9,
        "z2": 10,
        "exit": 11,
    },
}
_PRIOR_ORDER: tuple[str, ...] = (
    "direction",
    "stop",
    "risk",
    "target",
    "sl_mode",
    "tp_mode",
    "exit",
)


def _layout_key(
    width: int,
    has_direction: bool = False,
    has_sl_mode: bool = False,
    has_tp_mode: bool = False,
    has_multi_tp: bool = False,
    has_simplex: bool = False,
) -> str:
    if width == 1:
        return "1"
    if width == 5:
        return "5"
    if width == 6:
        if has_direction:
            return "6_dir"
        if has_tp_mode:
            return "6_tp"
        return "6_sl"
    if width == 7:
        if has_tp_mode and not has_sl_mode:
            return "7_tp"
        return "7"
    if width == 8:
        return "8"
    if width == 10:
        return "10"
    if width == 12:
        return "12"
    raise ValueError(f"unsupported action width {width}")


def mode_index(
    name: str,
    width: int,
    has_direction: bool = False,
    has_sl_mode: bool = False,
    has_tp_mode: bool = False,
    has_multi_tp: bool = False,
    has_simplex: bool = False,
) -> int:
    """Column of one prior name."""
    key = _layout_key(
        width,
        has_direction,
        has_sl_mode,
        has_tp_mode,
        has_multi_tp,
        has_simplex,
    )
    table = _MODE_INDEX.get(key)
    if table is None or name not in table:
        raise ValueError(f"no mode prior index for {name!r} at width {width} (layout {key})")
    return table[name]


def attach_mode_prior(
    mean_actions: torch.Tensor,
    coef: float,
    dims: tuple[int, ...],
) -> None:
    """Add the gradient of ``coef * sum(mean(mu_d)^2)`` on ``dims`` only.

    The penalty is the square of each batch mean, so a state that wants -1
    and a state that wants +1 can cancel. An empty ``dims`` or a zero coef
    adds nothing.
    """
    scale_coef = float(coef)
    if scale_coef == 0.0 or not dims or not mean_actions.requires_grad:
        return
    if mean_actions.ndim != 2 or mean_actions.shape[0] == 0 or mean_actions.shape[1] == 0:
        return
    width = int(mean_actions.shape[1])
    columns = tuple(int(dim) for dim in dims)
    if any(dim < 0 or dim >= width for dim in columns):
        raise ValueError(f"mode prior dims {columns} do not fit width {width}")
    batch = float(mean_actions.shape[0])

    def _hook(grad: torch.Tensor | None) -> torch.Tensor | None:
        if grad is None:
            return None
        extra = torch.zeros_like(grad)
        values = mean_actions.detach()
        scale = 2.0 * scale_coef / batch
        for dim in columns:
            extra[:, dim] = scale * values[:, dim].mean()
        return grad + extra

    mean_actions.register_hook(_hook)  # type: ignore[no-untyped-call]


def install_post_update_std_clip(model: Any, log_std_min: float, log_std_max: float) -> None:
    """Clamp ``log_std`` after ``model.train`` and log the post-clamp std.

    Rollout samples are already clamped inside the distribution build. SB3
    records ``train/std`` from the parameter immediately after the optimizer
    step, before that next forward. This clip runs when ``train`` returns, so
    the entropy bonus does not keep a value outside the band, and the logged
    number is the one the next rollout will use.

    Mode-mean stats accumulated during ``train`` are recorded on the same return.
    """
    orig = model.train

    def _train() -> None:
        policy = model.policy
        reset = getattr(policy, "reset_mode_prior_stats", None)
        if callable(reset):
            reset()
        orig()
        logger = getattr(model, "logger", None)
        if clip_policy_log_std(policy, log_std_min, log_std_max) and logger is not None:
            log_std = policy.log_std.detach()
            std = float(torch.exp(log_std).mean().item())
            logger.record("train/std", std)
        metrics = getattr(policy, "mode_prior_metrics", None)
        if logger is not None and callable(metrics):
            for key, value in metrics().items():
                logger.record(key, float(value))

    model.train = _train


def clip_policy_log_std(policy: Any, log_std_min: float, log_std_max: float) -> bool:
    """Clamp ``policy.log_std`` in-place. Returns False when the param is absent."""
    log_std = getattr(policy, "log_std", None)
    if log_std is None:
        return False
    with torch.no_grad():
        log_std.clamp_(float(log_std_min), float(log_std_max))
    return True


class ClampedStdMultiInputPolicy(MultiInputActorCriticPolicy):
    """Multi-input PPO policy that clamps Gaussian ``log_std`` on every dist build.

    SB3's entropy bonus on ``Box[-1, 1]`` overlay actions otherwise drives
    ``std`` to thousands (seed 50 at 20M) and saturates tanh/env clipping.

    ``account_value_weight`` is a zero-start linear read of the raw account
    vector, added to the value MLP. Open PnL tracks the return target, and the
    shared embedding plus value MLP were collapsing to a constant.
    """

    def __init__(
        self,
        *args: Any,
        log_std_min: float = -0.7,
        log_std_max: float = 0.0,
        mode_prior_coef: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for ClampedStdMultiInputPolicy")
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.mode_prior_coef = float(mode_prior_coef)
        self._active_prior: frozenset[str] = frozenset()
        self._action_layout_key = "5"
        super().__init__(*args, **kwargs)
        self.reset_mode_prior_stats()
        # Optimizer is built inside super().__init__, so a parameter created
        # after that has to join the existing Adam group or it never moves.
        self.account_value_weight = torch.nn.Parameter(torch.zeros(6, device=self.device))
        group = {
            key: value for key, value in self.optimizer.param_groups[0].items() if key != "params"
        }
        group["params"] = [self.account_value_weight]
        self.optimizer.add_param_group(group)

    def _account_value(self, obs: Any) -> torch.Tensor:
        account = obs["account"] if isinstance(obs, dict) else obs.account
        if not isinstance(account, torch.Tensor):
            account = torch.as_tensor(account, dtype=torch.float32, device=self.device)
        extra = account.float() @ self.account_value_weight
        return extra.unsqueeze(-1)

    def forward(
        self, obs: Any, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, values, log_prob = super().forward(obs, deterministic=deterministic)
        return actions, values + self._account_value(obs), log_prob

    def predict_values(self, obs: Any) -> torch.Tensor:
        return super().predict_values(obs) + self._account_value(obs)

    def evaluate_actions(
        self, obs: Any, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        values, log_prob, entropy = super().evaluate_actions(obs, actions)
        return values + self._account_value(obs), log_prob, entropy

    def set_active_prior(self, names: set[str] | frozenset[str]) -> None:
        """Replace the names the next update will pull toward zero."""
        cleaned = {str(name) for name in names}
        unknown = cleaned.difference(_PRIOR_ORDER)
        if unknown:
            raise ValueError(f"unknown prior names: {sorted(unknown)}")
        self._active_prior = frozenset(cleaned)

    def reset_mode_prior_stats(self) -> None:
        """Clear the prior terms accumulated since the last PPO update."""
        self._mode_term_sum = {name: 0.0 for name in _PRIOR_ORDER}
        self._mode_prior_sum = 0.0
        self._mode_batches = 0

    def mode_prior_metrics(self) -> dict[str, float]:
        """Prior value, 0/1 active flags, and a term for each active name."""
        batches = int(self._mode_batches)
        if batches <= 0:
            return {}
        metrics = {"train/mode_prior": self._mode_prior_sum / batches}
        for name in _PRIOR_ORDER:
            metrics[f"prior_active_{name}"] = 1.0 if name in self._active_prior else 0.0
            if name in self._active_prior:
                metrics[f"train/mode_prior_{name}"] = self._mode_term_sum[name] / batches
        return metrics

    def _prior_dims(self, width: int) -> tuple[int, ...]:
        if not self._active_prior:
            return ()
        key = getattr(self, "_action_layout_key", "5")
        table = _MODE_INDEX.get(key)
        if table is None:
            return ()
        return tuple(
            table[name] for name in _PRIOR_ORDER if name in self._active_prior and name in table
        )

    def _note_mode_means(self, mean_actions: torch.Tensor) -> None:
        width = int(mean_actions.shape[1])
        if width < 1 or mean_actions.shape[0] == 0:
            return
        key = getattr(self, "_action_layout_key", "5")
        table = _MODE_INDEX.get(key)
        if table is None:
            return
        coef = float(self.mode_prior_coef)
        prior = 0.0
        for name in _PRIOR_ORDER:
            if name not in self._active_prior or name not in table:
                continue
            bar = float(mean_actions[:, table[name]].detach().mean())
            term = coef * bar * bar
            self._mode_term_sum[name] += term
            prior += term
        self._mode_prior_sum += prior
        self._mode_batches += 1

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            mode_prior_coef=self.mode_prior_coef,
        )
        return data

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Any:
        clip_policy_log_std(self, self.log_std_min, self.log_std_max)
        dist = super()._get_action_dist_from_latent(latent_pi)
        mean = getattr(getattr(dist, "distribution", None), "loc", None)
        if isinstance(mean, torch.Tensor) and mean.ndim == 2:
            self._note_mode_means(mean)
            attach_mode_prior(mean, self.mode_prior_coef, self._prior_dims(int(mean.shape[1])))
        return dist
