"""Terminal dashboard for a PPO training run.

Rollout behavior, reward parts, and critic stats print every update.
A short deterministic slice adds return and risk metrics on a slower cadence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from quant_rl.envs.reward import REWARD_PART_KEYS, empty_reward_parts

log = logging.getLogger(__name__)

_CLOSE_KEYS: tuple[str, ...] = (
    "daily_loss",
    "trailing_dd",
    "max_drawdown",
    "structure_sl",
    "structure_tp",
    "session_end",
    "peak_dd_behavior",
)

_PART_LABELS: tuple[tuple[str, str], ...] = (
    ("pnl", "PnL"),
    ("dsr", "differential Sharpe"),
    ("soft_daily", "daily penalty"),
    ("soft_year", "year penalty"),
    ("soft_trailing", "trailing penalty"),
    ("strategy", "entry shaping"),
    ("breach", "breach"),
    ("sweep", "sweep"),
    ("peak_dd", "peak-dd penalty"),
)


def position_mix(directions: list[int]) -> dict[str, float]:
    """Share of steps that were long, short, or flat."""
    n = len(directions)
    if n == 0:
        return {"long": 0.0, "short": 0.0, "flat": 0.0}
    arr = np.asarray(directions, dtype=int)
    return {
        "long": float(np.mean(arr > 0)),
        "short": float(np.mean(arr < 0)),
        "flat": float(np.mean(arr == 0)),
    }


def action_stats(actions: list[float]) -> dict[str, float]:
    """Mean and std of the actions taken in one rollout."""
    if not actions:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(actions, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def close_shares(reasons: list[str]) -> dict[str, float]:
    """Each close reason as a fraction of closes in the window."""
    n = len(reasons)
    shares = {key: 0.0 for key in _CLOSE_KEYS}
    if n == 0:
        return shares
    for reason in reasons:
        if reason in shares:
            shares[reason] += 1.0
    return {key: value / n for key, value in shares.items()}


def critic_stats(advantages: Any, returns: Any, values: Any) -> dict[str, float]:
    """Mean return, value prediction, and advantage from one rollout buffer."""
    adv = np.asarray(advantages, dtype=float).reshape(-1)
    ret = np.asarray(returns, dtype=float).reshape(-1)
    val = np.asarray(values, dtype=float).reshape(-1)
    if adv.size == 0:
        return {"adv_mean": 0.0, "adv_std": 0.0, "ret_mean": 0.0, "val_mean": 0.0}
    return {
        "adv_mean": float(np.mean(adv)),
        "adv_std": float(np.std(adv)),
        "ret_mean": float(np.mean(ret)) if ret.size else 0.0,
        "val_mean": float(np.mean(val)) if val.size else 0.0,
    }


def sum_parts(rows: list[dict[str, float]]) -> dict[str, float]:
    """Add reward splits across the steps of one rollout."""
    total = empty_reward_parts()
    for row in rows:
        for key in REWARD_PART_KEYS:
            total[key] += float(row.get(key, 0.0))
    return total


def format_dashboard(snapshot: dict[str, Any]) -> str:
    """One text block for the terminal."""
    steps = int(snapshot["timesteps"])
    total = int(snapshot["total_timesteps"])
    lines = [
        "================================================================",
        f"PPO TRAINING | {steps:,} / {total:,}",
        "================================================================",
    ]
    ev = snapshot.get("eval")
    if ev:
        lines.extend(
            [
                "EVAL",
                f"return            {ev['return_pct']:+.2f}%",
                f"Sharpe            {ev['sharpe']:.2f}",
                f"Sortino           {ev['sortino']:.2f}",
                f"max DD            {-100.0 * ev['max_drawdown']:.2f}%",
                f"profit factor     {ev['profit_factor']:.2f}",
                f"win rate          {100.0 * ev['win_rate']:.1f}%",
                f"avg trade         {ev['avg_trade']:+.2f}",
                f"trades            {int(ev['n_trades'])}",
                f"turnover          {ev['turnover']:.4f}",
                f"train reward      {snapshot['train_reward_mean']:+.4f}",
                f"eval reward       {ev['reward_mean']:+.4f}",
            ]
        )
    else:
        lines.append(f"TRAIN reward      {snapshot['train_reward_mean']:+.4f}   (eval not run yet)")

    parts = snapshot["reward_parts"]
    lines.append("REWARD")
    for key, label in _PART_LABELS:
        lines.append(f"{label:<18}{parts.get(key, 0.0):+.4f}")
    lines.append(f"{'total':<18}{sum(parts.values()):+.4f}")

    mix = snapshot["position"]
    act = snapshot["action"]
    lines.extend(
        [
            "BEHAVIOR",
            f"action mean       {act['mean']:+.3f}",
            f"action std        {act['std']:.3f}",
            f"long              {100.0 * mix['long']:.1f}%",
            f"short             {100.0 * mix['short']:.1f}%",
            f"flat              {100.0 * mix['flat']:.1f}%",
        ]
    )

    risk = snapshot["closes"]
    lines.append("RISK")
    lines.append(f"closes            {int(snapshot['n_closes'])}")
    for key in _CLOSE_KEYS:
        lines.append(f"{key:<18}{100.0 * risk.get(key, 0.0):.1f}%")

    critic = snapshot["critic"]
    policy = snapshot["policy"]
    lines.extend(
        [
            "CRITIC",
            f"explained var     {policy['explained_variance']:.4f}",
            f"value loss        {policy['value_loss']:.4f}",
            f"mean return       {critic['ret_mean']:+.4f}",
            f"mean value        {critic['val_mean']:+.4f}",
            f"advantage mean    {critic['adv_mean']:+.4f}",
            f"advantage std     {critic['adv_std']:.4f}",
            "POLICY",
            f"approx KL         {policy['approx_kl']:.4f}",
            f"clip fraction     {policy['clip_fraction']:.4f}",
            f"entropy           {policy['entropy']:.3f}",
            f"std               {policy['std']:.3f}",
            f"grad norm         {policy['grad_norm']}",
            "================================================================",
        ]
    )
    return "\n".join(lines)


def _logger_float(values: dict[str, Any], key: str) -> float:
    try:
        return float(values.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def policy_from_logger(values: dict[str, Any], grad_norm: float | None) -> dict[str, Any]:
    """PPO figures Stable-Baselines3 already records, plus the captured grad norm."""
    entropy_loss = _logger_float(values, "train/entropy_loss")
    grad = "n/a" if grad_norm is None else f"{grad_norm:.3f}"
    return {
        "explained_variance": _logger_float(values, "train/explained_variance"),
        "value_loss": _logger_float(values, "train/value_loss"),
        "approx_kl": _logger_float(values, "train/approx_kl"),
        "clip_fraction": _logger_float(values, "train/clip_fraction"),
        "entropy": -entropy_loss,
        "std": _logger_float(values, "train/std"),
        "grad_norm": grad,
    }


try:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback as _Base

    _SB3_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _Base = object  # type: ignore[assignment,misc]
    _SB3_AVAILABLE = False


if _SB3_AVAILABLE:

    class TrainingDashboardCallback(_Base):
        """Print the training dashboard once per PPO update."""

        def __init__(
            self,
            total_timesteps: int,
            eval_every: int = 100_000,
            eval_fn: Any | None = None,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose=verbose)
            self.total_timesteps = int(total_timesteps)
            self.eval_every = int(eval_every)
            self._eval_fn = eval_fn
            self._next_eval = self.eval_every if self.eval_every > 0 else 0
            self._parts: list[dict[str, float]] = []
            self._directions: list[int] = []
            self._actions: list[float] = []
            self._closes: list[str] = []
            self._reward_sum = 0.0
            self._reward_steps = 0
            self._since_eval_reward = 0.0
            self._since_eval_steps = 0
            self._last_eval: dict[str, Any] | None = None
            self._grad_norm: float | None = None

        def _on_training_start(self) -> None:
            orig = getattr(self.model, "train")
            callback = self

            def _train_and_report() -> None:
                snapshot = callback._snapshot()
                callback._maybe_eval()
                norms: list[float] = []
                real_clip = torch.nn.utils.clip_grad_norm_

                def _cap(parameters: Any, max_norm: float, *args: Any, **kwargs: Any) -> Any:
                    total = real_clip(parameters, max_norm, *args, **kwargs)
                    try:
                        norms.append(float(total))
                    except (TypeError, ValueError):
                        pass
                    return total

                torch.nn.utils.clip_grad_norm_ = _cap
                try:
                    orig()
                finally:
                    torch.nn.utils.clip_grad_norm_ = real_clip
                if norms:
                    callback._grad_norm = float(np.mean(norms))
                snapshot["policy"] = policy_from_logger(
                    dict(callback.model.logger.name_to_value),
                    callback._grad_norm,
                )
                if callback._last_eval is not None:
                    snapshot["eval"] = callback._last_eval
                print(format_dashboard(snapshot))

            setattr(self.model, "train", _train_and_report)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            rewards = self.locals.get("rewards")
            actions = self.locals.get("actions")
            if rewards is not None:
                flat_rewards = np.asarray(rewards, dtype=float).reshape(-1)
                self._reward_sum += float(np.sum(flat_rewards))
                self._reward_steps += int(flat_rewards.size)
                self._since_eval_reward += float(np.sum(flat_rewards))
                self._since_eval_steps += int(flat_rewards.size)
            if actions is not None:
                self._actions.extend(float(v) for v in np.asarray(actions, dtype=float).reshape(-1))
            for info in infos:
                if not isinstance(info, dict):
                    continue
                parts = info.get("reward_parts")
                if isinstance(parts, dict):
                    self._parts.append(
                        {key: float(parts.get(key, 0.0)) for key in REWARD_PART_KEYS}
                    )
                if "position_direction" in info:
                    self._directions.append(int(info["position_direction"]))
                reason = info.get("close_reason")
                if reason:
                    self._closes.append(str(reason))
            return True

        def _snapshot(self) -> dict[str, Any]:
            train_mean = (
                self._since_eval_reward / self._since_eval_steps if self._since_eval_steps else 0.0
            )
            snapshot = {
                "timesteps": int(self.num_timesteps),
                "total_timesteps": self.total_timesteps,
                "train_reward_mean": train_mean,
                "reward_parts": sum_parts(self._parts),
                "position": position_mix(self._directions),
                "action": action_stats(self._actions),
                "closes": close_shares(self._closes),
                "n_closes": len(self._closes),
                "critic": self._critic_from_buffer(),
                "policy": policy_from_logger({}, None),
                "eval": None,
            }
            self._parts.clear()
            self._directions.clear()
            self._actions.clear()
            self._closes.clear()
            self._reward_sum = 0.0
            self._reward_steps = 0
            return snapshot

        def _critic_from_buffer(self) -> dict[str, float]:
            buffer = getattr(self.model, "rollout_buffer", None)
            if buffer is None or getattr(buffer, "advantages", None) is None:
                return critic_stats([], [], [])
            return critic_stats(buffer.advantages, buffer.returns, buffer.values)

        def _maybe_eval(self) -> None:
            if self._eval_fn is None or self._next_eval <= 0:
                return
            if int(self.num_timesteps) < self._next_eval:
                return
            try:
                self._last_eval = dict(self._eval_fn())
            except Exception as exc:
                log.warning("Dashboard eval skipped: %s", exc)
                return
            crossed = 1 + (int(self.num_timesteps) - self._next_eval) // self.eval_every
            self._next_eval += self.eval_every * crossed
            self._since_eval_reward = 0.0
            self._since_eval_steps = 0

        def _on_rollout_end(self) -> None:
            return None

else:

    class TrainingDashboardCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for TrainingDashboardCallback. "
                "Install it with: pip install stable-baselines3"
            )
