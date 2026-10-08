"""Terminal dashboard for a PPO training run.

Rollout behavior, reward parts, and critic stats print on ``log_every``.
A short deterministic slice adds return and risk metrics on the same cadence.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_rl.envs.reward import REWARD_PART_KEYS, empty_reward_parts
from quant_rl.eval.decision_diversity import (
    DiversityThresholds,
    active_prior_names,
    choice_rank_median,
    risk_mean_collapsed,
)
from quant_rl.utils.box_table import kv_lines, kv_widths, render_box_table

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


def ppo_log_interval(*, n_steps: int, n_envs: int, log_every: int) -> int:
    """How many PPO rollouts to skip between Stable-Baselines3 log dumps.

    ``log_every`` is in environment steps. One rollout is ``n_steps * n_envs``.
    """
    rollout = max(1, int(n_steps) * max(1, int(n_envs)))
    if log_every <= 0:
        return 1
    return max(1, int(log_every) // rollout)


def format_dashboard(snapshot: dict[str, Any]) -> str:
    """A few column rows, one per section, for the terminal log."""
    steps = int(snapshot["timesteps"])
    total = int(snapshot["total_timesteps"])
    parts = snapshot["reward_parts"]
    action = snapshot["action"]
    position = snapshot["position"]
    closes = snapshot["closes"]
    critic = snapshot["critic"]
    policy = snapshot["policy"]
    reward = "  ".join(f"{label} {parts.get(key, 0.0):+.4f}" for key, label in _PART_LABELS)
    per_1k = snapshot.get("reward_per_1k") or {}
    per_line = "  ".join(f"{label} {per_1k.get(key, 0.0):+.4f}" for key, label in _PART_LABELS)
    risk = "  ".join(f"{key} {100.0 * closes.get(key, 0.0):.1f}%" for key in _CLOSE_KEYS)
    lines = [
        f"PPO {steps:,} / {total:,}   train_reward {snapshot['train_reward_mean']:+.4f}",
        f"reward    {reward}",
        f"per1k     {per_line}  breach_events {int(snapshot.get('breach_events', 0))}",
        (
            "behavior  "
            f"mean {action['mean']:+.3f}  std {action['std']:.3f}  "
            f"long {100.0 * position['long']:.1f}%  "
            f"short {100.0 * position['short']:.1f}%  "
            f"flat {100.0 * position['flat']:.1f}%"
        ),
        f"risk      closes {int(snapshot['n_closes'])}  {risk}",
        (
            "policy    "
            f"expl {policy['explained_variance']:.4f}  "
            f"vloss {policy['value_loss']:.4f}  "
            f"ret {critic['ret_mean']:+.4f}  val {critic['val_mean']:+.4f}  "
            f"adv {critic['adv_mean']:+.4f}  "
            f"kl {policy['approx_kl']:.4f}  clip {policy['clip_fraction']:.4f}  "
            f"ent {policy['entropy']:.3f}  std {policy['std']:.3f}  "
            f"grad {policy['grad_norm']}"
        ),
    ]
    ev = snapshot.get("eval")
    if ev:
        if int(ev.get("n_trades", 0)) == 0:
            lines.append("eval      FAILED  trades 0  empty replay is not a flat return")
        else:
            lines.append(
                "eval      "
                f"return {ev['return_pct']:+.2f}%  sharpe {ev['sharpe']:.2f}  "
                f"sortino {ev['sortino']:.2f}  "
                f"maxdd {-100.0 * ev['max_drawdown']:.2f}%  "
                f"pf {ev['profit_factor']:.2f}  "
                f"win {100.0 * ev['win_rate']:.1f}%  "
                f"avg {ev['avg_trade']:+.2f}  trades {int(ev['n_trades'])}  "
                f"turn {ev['turnover']:.4f}  rew {ev['reward_mean']:+.4f}"
            )
    if ev:
        diversity = ev.get("diversity_text")
        if diversity:
            lines.append(str(diversity).rstrip())
    if "prior_names" in snapshot:
        names = snapshot.get("prior_names") or []
        text = ", ".join(str(name) for name in names) if names else "none"
        lines.append(f"prior     {text}")
    return "\n".join(lines)


def format_dashboard_table(snapshot: dict[str, Any]) -> str:
    """A table of the dashboard metrics with box borders for readability.

    Each section is a stacked box with --- borders. The diversity report
    is rendered separately by ``format_diversity_table``.
    """
    rows: list[tuple[str, list[tuple[str, str]]]] = []
    steps = int(snapshot["timesteps"])
    total = int(snapshot["total_timesteps"])
    parts = snapshot["reward_parts"]
    action = snapshot["action"]
    position = snapshot["position"]
    closes = snapshot["closes"]
    critic = snapshot["critic"]
    policy = snapshot["policy"]
    ev = snapshot.get("eval")

    training_rows = [
        ("timesteps", f"{steps:,}"),
        ("total_timesteps", f"{total:,}"),
        ("train_reward_mean", f"{snapshot['train_reward_mean']:+.4f}"),
    ]
    rows.append(("training", training_rows))

    # Reward parts (raw + per-1k)
    reward_rows: list[tuple[str, str]] = []
    per1k_rows: list[tuple[str, str]] = []
    per_1k = snapshot.get("reward_per_1k") or {}
    for key, label in _PART_LABELS:
        reward_rows.append((label, f"{parts.get(key, 0.0):+.4f}"))
        per1k_rows.append((label, f"{per_1k.get(key, 0.0):+.4f}"))
    rows.append(("reward", reward_rows))
    rows.append(("per1k", per1k_rows))

    behavior_rows = [
        ("mean", f"{action['mean']:+.3f}"),
        ("std", f"{action['std']:.3f}"),
        ("long", f"{100.0 * position['long']:.1f}%"),
        ("short", f"{100.0 * position['short']:.1f}%"),
        ("flat", f"{100.0 * position['flat']:.1f}%"),
    ]
    rows.append(("behavior", behavior_rows))

    risk_rows: list[tuple[str, str]] = [("closes", str(int(snapshot["n_closes"])))]
    for key in _CLOSE_KEYS:
        risk_rows.append((key, f"{100.0 * closes.get(key, 0.0):.1f}%"))
    rows.append(("risk", risk_rows))

    policy_rows = [
        ("explained_variance", f"{policy['explained_variance']:.4f}"),
        ("value_loss", f"{policy['value_loss']:.4f}"),
        ("ret", f"{critic['ret_mean']:+.4f}"),
        ("val", f"{critic['val_mean']:+.4f}"),
        ("adv", f"{critic['adv_mean']:+.4f}"),
        ("kl", f"{policy['approx_kl']:.4f}"),
        ("clip", f"{policy['clip_fraction']:.4f}"),
        ("entropy", f"{policy['entropy']:.3f}"),
        ("std", f"{policy['std']:.3f}"),
        ("grad_norm", str(policy.get("grad_norm", ""))),
    ]
    rows.append(("policy", policy_rows))

    if ev:
        eval_rows: list[tuple[str, str]] = [
            ("breach_events", str(int(ev.get("breach_events", 0)))),
        ]
        if int(ev.get("n_trades", 0)) == 0:
            eval_rows.append(("status", "FAILED  trades 0"))
        else:
            eval_rows.extend(
                [
                    ("return_pct", f"{ev['return_pct']:+.2f}%"),
                    ("sharpe", f"{ev['sharpe']:.2f}"),
                    ("sortino", f"{ev['sortino']:.2f}"),
                    ("max_drawdown", f"{-100.0 * ev['max_drawdown']:.2f}%"),
                    ("profit_factor", f"{ev['profit_factor']:.2f}"),
                    ("win_rate", f"{100.0 * ev['win_rate']:.1f}%"),
                    ("avg_trade", f"{ev['avg_trade']:+.2f}"),
                    ("n_trades", str(int(ev["n_trades"]))),
                    ("turnover", f"{ev['turnover']:.4f}"),
                    ("reward_mean", f"{ev['reward_mean']:+.4f}"),
                ]
            )
        rows.append(("eval", eval_rows))

    all_pairs = [pair for _, pairs in rows for pair in pairs]
    key_width, value_width = kv_widths(all_pairs)
    boxes = [(title, kv_lines(pairs, key_width, value_width)) for title, pairs in rows]
    return render_box_table(boxes)


# Eval drawdown is ``eval_max_drawdown`` so it does not overwrite the close-reason
# share stored in ``max_drawdown``.
_EVAL_ROW_KEYS: tuple[tuple[str, str], ...] = (
    ("return_pct", "return_pct"),
    ("sharpe", "sharpe"),
    ("sortino", "sortino"),
    ("max_drawdown", "eval_max_drawdown"),
    ("profit_factor", "profit_factor"),
    ("win_rate", "win_rate"),
    ("avg_trade", "avg_trade"),
    ("n_trades", "n_trades"),
    ("turnover", "turnover"),
    ("reward_mean", "eval_reward"),
)

DASHBOARD_COLUMNS: tuple[str, ...] = (
    "timestep",
    "train_reward_mean",
    *REWARD_PART_KEYS,
    *(f"{key}_per_1k" for key in REWARD_PART_KEYS),
    "breach_events",
    "action_mean",
    "action_std",
    "long",
    "short",
    "flat",
    "n_closes",
    *_CLOSE_KEYS,
    "grad_norm",
    "return_pct",
    "sharpe",
    "sortino",
    "eval_max_drawdown",
    "profit_factor",
    "win_rate",
    "avg_trade",
    "n_trades",
    "turnover",
    "eval_reward",
    "prior_active_stop",
    "prior_active_target",
    "prior_active_exit",
    "prior_active_direction",
    "prior_active_risk",
    "sl_rank_median",
    "tp_rank_median",
)

_PRIOR_NAMES: tuple[str, ...] = ("stop", "target", "exit", "direction", "risk")


def snapshot_row(snapshot: dict[str, Any], *, record_eval: bool = False) -> dict[str, Any]:
    """Flatten one dashboard snapshot. Eval keys appear only when ``record_eval``."""
    parts = snapshot["reward_parts"]
    action = snapshot["action"]
    position = snapshot["position"]
    closes = snapshot["closes"]
    policy = snapshot.get("policy") or {}
    row: dict[str, Any] = {
        "timestep": int(snapshot["timesteps"]),
        "train_reward_mean": float(snapshot["train_reward_mean"]),
    }
    for key in REWARD_PART_KEYS:
        row[key] = float(parts.get(key, 0.0))
    per_1k = snapshot.get("reward_per_1k") or {}
    for key in REWARD_PART_KEYS:
        row[f"{key}_per_1k"] = float(per_1k.get(key, 0.0))
    row["breach_events"] = int(snapshot.get("breach_events", 0))
    row["action_mean"] = float(action["mean"])
    row["action_std"] = float(action["std"])
    row["long"] = float(position["long"])
    row["short"] = float(position["short"])
    row["flat"] = float(position["flat"])
    row["n_closes"] = int(snapshot["n_closes"])
    for key in _CLOSE_KEYS:
        row[key] = float(closes.get(key, 0.0))
    row["grad_norm"] = _grad_norm_value(policy.get("grad_norm"))
    if record_eval and snapshot.get("eval"):
        ev = snapshot["eval"]
        for src, dest in _EVAL_ROW_KEYS:
            row[dest] = float(ev[src])
    active = set(snapshot.get("prior_names") or [])
    for name in _PRIOR_NAMES:
        row[f"prior_active_{name}"] = 1 if name in active else 0
    for key in ("sl_rank_median", "tp_rank_median"):
        value = snapshot.get(key)
        row[key] = "" if value is None or not np.isfinite(float(value)) else float(value)
    return row


def _action_width(model: Any) -> int | None:
    space = getattr(model, "action_space", None)
    shape = getattr(space, "shape", None)
    if not shape:
        return None
    return int(shape[0])


def _grad_norm_value(raw: Any) -> float | str:
    if raw is None or raw == "n/a" or raw == "":
        return ""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return ""


def append_dashboard_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one snapshot row and flush so a crash keeps the log."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not dest.exists() or dest.stat().st_size == 0
    with dest.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DASHBOARD_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in DASHBOARD_COLUMNS})
        handle.flush()


def _logger_float(values: dict[str, Any], key: str) -> float:
    try:
        value = float(values.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) and key.endswith("explained_variance"):
        return 0.0
    return value


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
        """Print and record the dashboard every ``log_every`` env steps."""

        def __init__(
            self,
            total_timesteps: int,
            eval_every: int = 100_000,
            eval_fn: Any | None = None,
            log_path: str | Path | None = None,
            log_every: int = 100_000,
            verbose: int = 0,
            diversity: DiversityThresholds | None = None,
            prior_every: int | None = None,
        ) -> None:
            super().__init__(verbose=verbose)
            self.total_timesteps = int(total_timesteps)
            self.eval_every = int(eval_every)
            self.log_every = int(log_every)
            self._eval_fn = eval_fn
            self._log_path = Path(log_path) if log_path is not None else None
            self._diversity = diversity
            self._prior_every = int(self.log_every if prior_every is None else prior_every)
            self._next_prior = self._prior_every if self._prior_every > 0 else 0
            self._prior_opens: list[dict[str, Any]] = []
            self._prior_names: list[str] = []
            self._risk_sum = 0.0
            self._risk_n = 0
            self._sl_rank_median = float("nan")
            self._tp_rank_median = float("nan")
            self._eval_just_ran = False
            self._next_log = 0
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
                callback._apply_prior_interval()
                callback._maybe_eval()
                if not callback._should_log():
                    orig()
                    return
                snapshot = callback._snapshot()
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
                if callback._eval_just_ran and callback._last_eval is not None:
                    snapshot["eval"] = callback._last_eval
                print(format_dashboard_table(snapshot))
                if callback._eval_just_ran and callback._last_eval is not None:
                    ev = snapshot.get("eval") or {}
                    if ev.get("diversity_table"):
                        print()
                        print(ev["diversity_table"])
                prior_names = snapshot.get("prior_names")
                if prior_names is not None:
                    text = ", ".join(str(name) for name in prior_names) or "none"
                    pair = ("active", text)
                    kw, vw = kv_widths([pair])
                    print()
                    print(render_box_table([("prior", kv_lines([pair], kw, vw))]))
                if callback._log_path is not None:
                    append_dashboard_row(
                        callback._log_path,
                        snapshot_row(snapshot, record_eval=callback._eval_just_ran),
                    )
                callback._advance_log()

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
                opened = info.get("prior_open")
                if isinstance(opened, dict):
                    self._prior_opens.append(opened)
                risk_u = info.get("risk_u")
                if isinstance(risk_u, (int, float)) and np.isfinite(risk_u):
                    self._risk_sum += float(risk_u)
                    self._risk_n += 1
            return True

        def _should_log(self) -> bool:
            if self.log_every <= 0:
                return True
            return int(self.num_timesteps) >= self._next_log or self._eval_just_ran

        def _advance_log(self) -> None:
            if self.log_every <= 0:
                return
            step = int(self.num_timesteps)
            self._next_log = self.log_every * (1 + step // self.log_every)

        def _snapshot(self) -> dict[str, Any]:
            train_mean = (
                self._since_eval_reward / self._since_eval_steps if self._since_eval_steps else 0.0
            )
            parts = sum_parts(self._parts)
            n_steps = max(1, int(self._reward_steps))
            per_1k = {key: float(value) * 1000.0 / n_steps for key, value in parts.items()}
            breach_events = sum(1 for part in self._parts if float(part.get("breach", 0.0)) != 0.0)
            snapshot = {
                "timesteps": int(self.num_timesteps),
                "total_timesteps": self.total_timesteps,
                "train_reward_mean": train_mean,
                "reward_parts": parts,
                "reward_per_1k": per_1k,
                "reward_steps": int(self._reward_steps),
                "breach_events": breach_events,
                "position": position_mix(self._directions),
                "action": action_stats(self._actions),
                "closes": close_shares(self._closes),
                "n_closes": len(self._closes),
                "critic": self._critic_from_buffer(),
                "policy": policy_from_logger({}, None),
                "eval": None,
                "prior_names": list(self._prior_names),
                "sl_rank_median": self._sl_rank_median,
                "tp_rank_median": self._tp_rank_median,
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

        def _apply_prior_interval(self) -> None:
            """Replace the active set from this interval of training opens.

            The dashboard replay is not consulted. Until the first interval
            finishes, the policy keeps the empty set it started with.
            """
            if self._prior_every <= 0 or int(self.num_timesteps) < self._next_prior:
                return
            frame = pd.DataFrame(self._prior_opens)
            names = active_prior_names(frame, self._diversity)
            risk_n = int(self._risk_n)
            risk_mean = self._risk_sum / risk_n if risk_n else float("nan")
            if risk_mean_collapsed(risk_mean, risk_n, self._diversity):
                names.add("risk")
            width = _action_width(self.model)
            if width is not None and width < 6:
                names.discard("direction")
            setter = getattr(self.model.policy, "set_active_prior", None)
            if callable(setter):
                setter(names)
            self._prior_names = sorted(names)
            self._sl_rank_median = choice_rank_median(frame, "sl")
            self._tp_rank_median = choice_rank_median(frame, "tp")
            text = ", ".join(self._prior_names) if self._prior_names else "none"
            log.info("prior active %s", text)
            self._prior_opens.clear()
            self._risk_sum = 0.0
            self._risk_n = 0
            step = int(self.num_timesteps)
            self._next_prior = self._prior_every * (1 + step // self._prior_every)

        def _maybe_eval(self) -> None:
            self._eval_just_ran = False
            if self._eval_fn is None or self._next_eval <= 0:
                return
            if int(self.num_timesteps) < self._next_eval:
                return
            try:
                self._last_eval = dict(self._eval_fn())
            except Exception as exc:
                log.warning("Dashboard eval skipped: %s", exc)
                return
            self._eval_just_ran = True
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
