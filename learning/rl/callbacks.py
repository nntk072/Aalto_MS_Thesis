"""SB3 training callbacks for progress logging and best-checkpoint eval."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import spaces

from learning.rl.learner import save_sb3_checkpoint
from learning.rl.ppo import clip_policy_log_std
from learning.rl.stopping import decide_early_abort

log = logging.getLogger(__name__)


def save_ppo_checkpoint(model: Any, path: str | Path) -> None:
    """Compatibility entry point for callback-owned checkpoint saves."""
    save_sb3_checkpoint(model, path)


def _action_coordinate_names(env: Any, width: int) -> list[str]:
    schema = getattr(env, "action_schema", None)
    to_dict = getattr(schema, "to_dict", None)
    if callable(to_dict):
        schema = to_dict()
    raw_names = schema.get("coordinates", ()) if isinstance(schema, dict) else ()
    names = [str(name) for name in raw_names] if isinstance(raw_names, (list, tuple)) else []
    return names if len(names) == width else [f"coordinate_{index}" for index in range(width)]


def _record_action_coordinates(
    totals: dict[str, dict[str, float]], action: Any, names: list[str]
) -> None:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != len(names):
        return
    for name, value in zip(names, values, strict=True):
        item = totals.setdefault(
            name,
            {
                "count": 0.0,
                "nonfinite_count": 0.0,
                "sum": 0.0,
                "sum_squares": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            },
        )
        if not np.isfinite(value):
            item["nonfinite_count"] += 1.0
            continue
        item["count"] += 1.0
        item["sum"] += float(value)
        item["sum_squares"] += float(value * value)
        item["min"] = min(item["min"], float(value))
        item["max"] = max(item["max"], float(value))


def _summarize_action_coordinates(
    totals: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for name, item in totals.items():
        count = int(item["count"])
        mean = item["sum"] / count if count else float("nan")
        variance = max(item["sum_squares"] / count - mean * mean, 0.0) if count else float("nan")
        summary[name] = {
            "n": count,
            "nonfinite_n": int(item["nonfinite_count"]),
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": item["min"] if count else float("nan"),
            "max": item["max"] if count else float("nan"),
        }
    return summary


try:
    from stable_baselines3.common.callbacks import BaseCallback as _Base

    _SB3_AVAILABLE = True
except ImportError:
    _Base = object  # type: ignore[assignment,misc]
    _SB3_AVAILABLE = False


if _SB3_AVAILABLE:

    class LearningDiagnosticsCallback(_Base):
        """Write per-PPO-rollout action, advantage, reward, and funnel diagnostics."""

        def __init__(self, log_path: str | Path, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._rows: list[dict[str, float]] = []
            self._funnel: dict[str, float] = {}
            self._reward_parts: dict[str, float] = {}
            self._constraints: dict[str, float] = {}
            self._window_steps = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            for info in infos:
                if not isinstance(info, dict):
                    continue
                self._window_steps += 1
                funnel = info.get("entry_funnel", {})
                if isinstance(funnel, dict):
                    for key, value in funnel.items():
                        self._funnel[str(key)] = self._funnel.get(str(key), 0.0) + float(value)
                parts = info.get("reward_parts", {})
                if isinstance(parts, dict):
                    for key, value in parts.items():
                        self._reward_parts[str(key)] = self._reward_parts.get(
                            str(key), 0.0
                        ) + float(value)
                constraints = info.get("constraints", {})
                if isinstance(constraints, dict):
                    for key, value in constraints.items():
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            metric_key = str(key)
                            self._constraints[metric_key] = self._constraints.get(
                                metric_key, 0.0
                            ) + float(value)
            return True

        def _on_rollout_end(self) -> None:
            model: Any = self.model
            buffer = model.rollout_buffer
            raw_actions = np.asarray(buffer.actions, dtype=np.float64)
            if isinstance(model.action_space, spaces.Box):
                actions = raw_actions.reshape(-1, raw_actions.shape[-1])
            else:
                actions = raw_actions.reshape(-1, 1)
            rewards = np.asarray(buffer.rewards, dtype=np.float64)
            advantages = np.asarray(buffer.advantages, dtype=np.float64)
            row: dict[str, float] = {
                "timestep": float(self.num_timesteps),
                "rollout/reward_mean": float(np.mean(rewards)),
                "rollout/reward_std": float(np.std(rewards)),
                "advantages/mean": float(np.mean(advantages)),
                "advantages/std": float(np.std(advantages)),
                "action/saturation_fraction": float(np.mean(np.abs(actions) >= 0.99)),
                "action/near_minus_one_fraction": float(np.mean(actions <= -0.95)),
                "action/near_plus_one_fraction": float(np.mean(actions >= 0.95)),
            }
            schema = getattr(model, "_action_schema", {})
            raw_names = (
                schema.get("coordinates", schema.get("names", []))
                if isinstance(schema, dict)
                else []
            )
            names = raw_names if isinstance(raw_names, list) else []
            if len(names) != actions.shape[1]:
                names = [f"coordinate_{index}" for index in range(actions.shape[1])]
            for index, name in enumerate(names):
                coordinate = actions[:, index]
                row[f"action/{name}_mean"] = float(np.mean(coordinate))
                row[f"action/{name}_std"] = float(np.std(coordinate))
                if "mode" in str(name) or "exit" in str(name):
                    bins = np.digitize(coordinate, [-1.0 / 3.0, 1.0 / 3.0])
                    for bin_index, label in enumerate(("low", "mid", "high")):
                        row[f"action/{name}_hist/{label}"] = float(np.mean(bins == bin_index))
            for key, value in self._funnel.items():
                row[f"entry_funnel/{key}"] = float(value)
                model.logger.record(f"entry_funnel/{key}", float(value))
            for key, value in self._reward_parts.items():
                row[f"reward_parts/cumulative_{key}"] = float(value)
                row[f"reward_parts/mean_{key}"] = float(value / max(1, self._window_steps))
                model.logger.record(f"reward_parts/cumulative_{key}", float(value))
                model.logger.record(
                    f"reward_parts/mean_{key}", float(value / max(1, self._window_steps))
                )
            for key, value in self._constraints.items():
                row[f"constraints/sum_{key}"] = float(value)
                row[f"constraints/mean_{key}"] = float(value / max(1, self._window_steps))
                model.logger.record(
                    f"constraints/mean_{key}", float(value / max(1, self._window_steps))
                )
            for key, value in row.items():
                if key.startswith(("action/", "rollout/", "advantages/")):
                    model.logger.record(f"quality/{key}", value)
            self._rows.append(row)
            self._funnel.clear()
            self._reward_parts.clear()
            self._constraints.clear()
            self._window_steps = 0

        def _on_training_end(self) -> None:
            if self._rows:
                pd.DataFrame(self._rows).to_csv(self._log_path, index=False)

    class LagrangianRiskCallback(_Base):
        """Primal-dual PPO shaping for annual account-survival risk.

        The environment's hard account guardrails remain unchanged. This
        callback adds the largest per-step increase among the normalized daily,
        max-loss, and trailing risk costs, then updates a nonnegative Lagrange
        multiplier from completed full-split episodes. It is a learning
        objective, not a risk guarantee.
        """

        _STATE_VERSION = 2

        def __init__(
            self,
            *,
            episode_cost_budget: float = 1.0,
            dual_learning_rate: float = 0.1,
            lambda_initial: float = 0.0,
            lambda_max: float = 100.0,
            gamma: float = 0.99,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose=verbose)
            configured_values = np.asarray(
                [episode_cost_budget, dual_learning_rate, lambda_initial, lambda_max, gamma],
                dtype=np.float64,
            )
            if not np.isfinite(configured_values).all():
                raise ValueError("Lagrangian PPO settings must all be finite")
            if episode_cost_budget < 0.0:
                raise ValueError("episode_cost_budget must be nonnegative")
            if dual_learning_rate <= 0.0:
                raise ValueError("dual_learning_rate must be positive")
            if not 0.0 < gamma <= 1.0:
                raise ValueError("gamma must be in (0, 1]")
            if lambda_initial < 0.0 or lambda_max <= 0.0 or lambda_initial > lambda_max:
                raise ValueError("Lagrange multiplier bounds are invalid")
            self.episode_cost_budget = float(episode_cost_budget)
            self.dual_learning_rate = float(dual_learning_rate)
            self.lambda_initial = float(lambda_initial)
            self.lambda_max = float(lambda_max)
            self.gamma = float(gamma)
            self.lagrange_multiplier = self.lambda_initial
            self._episode_costs: np.ndarray[Any, Any] = np.zeros(0, dtype=np.float64)
            self._episode_cost_discounts: np.ndarray[Any, Any] = np.ones(0, dtype=np.float64)
            self._completed_costs: list[float] = []
            self._rollout_environment_reward = 0.0
            self._rollout_penalty = 0.0
            self._rollout_transitions = 0
            self._dual_updates = 0

        def _config_state(self) -> dict[str, float | int]:
            return {
                "version": self._STATE_VERSION,
                "episode_cost_budget": self.episode_cost_budget,
                "dual_learning_rate": self.dual_learning_rate,
                "lambda_initial": self.lambda_initial,
                "lambda_max": self.lambda_max,
                "gamma": self.gamma,
            }

        def _persist_state(self) -> None:
            setattr(
                self.model,
                "_constrained_rl_state",
                {
                    **self._config_state(),
                    "lagrange_multiplier": self.lagrange_multiplier,
                    "dual_updates": self._dual_updates,
                },
            )

        def _on_training_start(self) -> None:
            state = getattr(self.model, "_constrained_rl_state", None)
            if state is not None:
                expected = self._config_state()
                for key, value in expected.items():
                    if state.get(key) != value:
                        raise ValueError(
                            f"checkpoint constrained-RL setting {key!r} differs from config"
                        )
                self.lagrange_multiplier = float(state["lagrange_multiplier"])
                self._dual_updates = int(state.get("dual_updates", 0))
                if not 0.0 <= self.lagrange_multiplier <= self.lambda_max:
                    raise ValueError("checkpoint Lagrange multiplier is outside configured bounds")
            self._episode_costs = np.zeros(int(self.model.n_envs), dtype=np.float64)
            self._episode_cost_discounts = np.ones(int(self.model.n_envs), dtype=np.float64)
            self._persist_state()

        @staticmethod
        def _episode_cost(info: Any) -> float:
            if not isinstance(info, dict):
                return 0.0
            constraints = info.get("constraints", {})
            if not isinstance(constraints, dict):
                return 0.0
            cost_fields = ("daily_cost", "max_loss_cost", "trailing_cost")
            costs = [
                max(0.0, float(constraints.get(key, 0.0)))
                for key in cost_fields
                if isinstance(constraints.get(key, 0.0), (int, float, np.integer, np.floating))
            ]
            breached = any(
                bool(constraints.get(key, False))
                for key in ("daily_breach", "max_loss_breach", "trailing_breach")
            )
            return max([*costs, float(breached)], default=0.0)

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")
            infos = self.locals.get("infos")
            if not isinstance(rewards, np.ndarray) or dones is None or infos is None:
                raise RuntimeError("Lagrangian PPO callback did not receive VecEnv step data")
            if len(rewards) != len(self._episode_costs) or len(infos) != len(rewards):
                raise RuntimeError("Lagrangian PPO callback environment batch size changed")

            self._rollout_environment_reward += float(np.sum(rewards))
            costs = np.asarray([self._episode_cost(info) for info in infos], dtype=np.float64)
            penalties = self.lagrange_multiplier * costs
            rewards -= penalties.astype(rewards.dtype, copy=False)
            self._rollout_penalty += float(np.sum(penalties))
            self._rollout_transitions += int(len(rewards))

            done_values = np.asarray(dones, dtype=bool)
            self._episode_costs += self._episode_cost_discounts * costs
            self._episode_cost_discounts *= self.gamma
            for index, done in enumerate(done_values):
                if done:
                    self._completed_costs.append(float(self._episode_costs[index]))
                    self._episode_costs[index] = 0.0
                    self._episode_cost_discounts[index] = 1.0
            return True

        def _on_rollout_end(self) -> None:
            model: Any = self.model
            steps = max(1, self._rollout_transitions)
            model.logger.record(
                "constrained_rl/environment_reward_mean",
                self._rollout_environment_reward / steps,
            )
            model.logger.record("constrained_rl/penalty_sum", self._rollout_penalty)
            model.logger.record("constrained_rl/lambda", self.lagrange_multiplier)
            model.logger.record("constrained_rl/completed_episodes", len(self._completed_costs))
            if self._completed_costs:
                observed_cost = float(np.mean(self._completed_costs))
                self.lagrange_multiplier = float(
                    np.clip(
                        self.lagrange_multiplier
                        + self.dual_learning_rate * (observed_cost - self.episode_cost_budget),
                        0.0,
                        self.lambda_max,
                    )
                )
                self._dual_updates += 1
                model.logger.record("constrained_rl/mean_completed_episode_cost", observed_cost)
                model.logger.record("constrained_rl/dual_updates", self._dual_updates)
                self._completed_costs.clear()
            self._persist_state()
            self._rollout_environment_reward = 0.0
            self._rollout_penalty = 0.0
            self._rollout_transitions = 0

    class ProgressLoggerCallback(_Base):
        """Record per-rollout training metrics and save them to a CSV.

        Captures ``model.logger.name_to_value`` after each rollout collection.
        Note: ``train/*`` metrics in a given row reflect the *previous* training
        step (they are computed after rollout collection, not before); this
        one-rollout lag is negligible for visualisation purposes.

        The CSV is written to *log_path* when training ends.
        """

        def __init__(self, log_path: str | Path, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._rows: list[dict[str, Any]] = []

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            row: dict[str, Any] = {"timestep": self.num_timesteps}
            for k, v in self.model.logger.name_to_value.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            self._rows.append(row)

        def _on_training_end(self) -> None:
            if self._rows:
                pd.DataFrame(self._rows).to_csv(self._log_path, index=False)

    class EpisodeEquityCallback(_Base):
        """Write one row per finished training episode to ``episode_equity.csv``.

        Warns when the shaped reward sum is positive while end equity is
        below the episode start. That is the reward/equity mismatch.
        """

        def __init__(self, log_path: str | Path, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._columns = [
                "timestep",
                "environment_index",
                "end_equity",
                "start_equity",
                "equity_delta",
                "reward_sum",
                "n_trades",
                "episode_steps",
                "completed_data_split",
                "episode_end_reason",
                "breach_reason",
            ]

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            infos = self.locals.get("infos") or []
            if dones is None:
                return True
            rows: list[dict[str, Any]] = []
            for i, done in enumerate(np.asarray(dones).reshape(-1)):
                if not done or i >= len(infos):
                    continue
                ep = infos[i].get("episode_equity") if isinstance(infos[i], dict) else None
                if not ep:
                    continue
                end_equity: float = float(ep["end_equity"])
                start_equity: float = float(ep["start_equity"])
                reward_sum: float = float(ep["reward_sum"])
                equity_delta = end_equity - start_equity
                row = {
                    "timestep": int(self.num_timesteps),
                    "environment_index": int(i),
                    "end_equity": end_equity,
                    "start_equity": start_equity,
                    "equity_delta": equity_delta,
                    "reward_sum": reward_sum,
                    "n_trades": int(ep["n_trades"]),
                    "episode_steps": int(ep.get("episode_steps", 0)),
                    "completed_data_split": bool(ep.get("completed_data_split", False)),
                    "episode_end_reason": str(ep.get("episode_end_reason") or ""),
                    "breach_reason": str(ep.get("breach_reason") or ""),
                }
                rows.append(row)
                if reward_sum > 0.0 and end_equity < start_equity:
                    log.warning(
                        "episode reward %.4f is positive but equity fell %.2f (breach=%s)",
                        reward_sum,
                        -equity_delta,
                        row["breach_reason"] or "none",
                    )
            if rows:
                pd.DataFrame(rows, columns=self._columns).to_csv(
                    self._log_path,
                    mode="a",
                    header=not self._log_path.exists(),
                    index=False,
                )
            return True

        def _on_training_end(self) -> None:
            if not self._log_path.exists():
                pd.DataFrame(columns=self._columns).to_csv(self._log_path, index=False)

    class BestCheckpointEvalCallback(_Base):
        """Save the policy with the highest end equity on a train-calendar rollout.

        The score is the account equity at the end of the rollout, not the
        shaped episode reward. A higher reward with a lower balance does not
        replace the saved checkpoint. The rollout uses deterministic actions
        so a repeated evaluation of a fixed policy and data slice is stable.

        Parameters
        ----------
        eval_env_factory : callable
            ``() -> TradingEnv`` returning a *fresh* evaluation environment.
            Called every ``eval_freq`` rollouts; reusing the same env would
            leak rollout state between evals.
        eval_freq : int
            Run an evaluation every ``eval_freq`` rollout steps.
        best_model_path : str | Path
            File path to write the best model to (SB3 ``.save()`` format).
        n_eval_episodes : int
            Number of independent rollouts to average over for the best-model
            decision. Defaults to 1, which is noisy but cheap; training runs
            should use 3–5 once ``eval_freq`` is tuned.
        """

        def __init__(
            self,
            eval_env_factory: Any,
            eval_freq: int = 5,
            best_model_path: str | Path = "best_model",
            n_eval_episodes: int = 1,
            verbose: int = 0,
            viability_eval_env_factory: Any | None = None,
            viability_eval_freq: int = 5_000_000,
        ) -> None:
            super().__init__(verbose=verbose)
            self._eval_env_factory = eval_env_factory
            self._eval_freq = int(eval_freq)
            self._best_model_path = Path(best_model_path)
            self._best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self._n_eval_episodes = int(n_eval_episodes)
            self._next_eval_at = self._eval_freq
            self._viability_eval_env_factory = viability_eval_env_factory or eval_env_factory
            self._viability_eval_freq = int(viability_eval_freq)
            self._next_viability_eval_at = self._viability_eval_freq
            self.viability_eval_history: list[dict[str, Any]] = []
            self.best_end_equity: float = -np.inf
            # Kept in sync with ``best_end_equity`` so older callers still read a score.
            self.best_mean_reward: float = -np.inf
            self.best_n_trades = 0
            self.best_checkpoint_status = "NO_ACTIVE_CHECKPOINT"
            self.consecutive_inactive_viability_evals = 0
            self.viability_failure_reason: str | None = None

        def _on_step(self) -> bool:
            # Use threshold crossings because vectorized PPO increments by
            # n_envs and may never land exactly on a requested interval.
            if self._eval_freq > 0 and self.num_timesteps >= self._next_eval_at:
                self._run_eval()
                while self._next_eval_at <= self.num_timesteps:
                    self._next_eval_at += self._eval_freq
            if self._viability_eval_freq > 0 and self.num_timesteps >= self._next_viability_eval_at:
                self._run_viability_eval()
                while self._next_viability_eval_at <= self.num_timesteps:
                    self._next_viability_eval_at += self._viability_eval_freq
            if self.viability_failure_reason is not None:
                return False
            return True

        def _run_eval(self) -> None:
            end_equities: list[float] = []
            completed_trades = 0
            survival_valid = True
            for _ in range(self._n_eval_episodes):
                env = self._eval_env_factory()
                obs, _ = env.reset()
                done = truncated = False
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _reward, done, truncated, _ = env.step(action)
                equity = float(env.account.equity)
                end_equities.append(equity)
                trade_log = getattr(env, "trade_log", ())
                closing_types = {
                    "close",
                    "eod_close",
                    "forced_close",
                    "stop_close",
                    "tp_close",
                }
                completed_trades += sum(
                    1 for trade in trade_log if trade.get("type") in closing_types
                )
                survival_valid = survival_valid and not bool(getattr(env, "breach_events", ()))
            end_equity = float(np.mean(end_equities))
            if completed_trades < 1 or not survival_valid or not np.isfinite(end_equity):
                self.best_checkpoint_status = "NO_ACTIVE_CHECKPOINT"
                if self.verbose:
                    print(
                        "[BestCheckpointEval] ineligible "
                        f"trades={completed_trades} survival_valid={survival_valid} "
                        f"end_equity_finite={np.isfinite(end_equity)}"
                    )
                return
            self.best_checkpoint_status = "ACTIVE_CHECKPOINT_AVAILABLE"
            if end_equity > self.best_end_equity:
                self.best_end_equity = end_equity
                self.best_mean_reward = end_equity
                self.best_n_trades = completed_trades
                save_ppo_checkpoint(self.model, self._best_model_path)
                if self.verbose:
                    print(
                        f"[BestCheckpointEval] new best end_equity={end_equity:.2f} at "
                        f"timestep={self.num_timesteps} → saved {self._best_model_path}"
                    )

        def _run_viability_eval(self) -> None:
            """Replay the complete training calendar for deterministic viability."""
            end_equities: list[float] = []
            completed_trades = 0
            survival_valid = True
            entry_state: dict[str, Any] = {}
            entry_funnel_diagnostics: dict[str, int] = {}
            action_coordinate_totals: dict[str, dict[str, float]] = {}
            action_coordinate_schema: list[str] | None = None
            action_coordinate_schema_mismatch = False
            for _ in range(self._n_eval_episodes):
                env = self._viability_eval_env_factory()
                try:
                    obs, _ = env.reset()
                    done = truncated = False
                    while not (done or truncated):
                        action, _ = self.model.predict(obs, deterministic=True)
                        action_values = np.asarray(action).reshape(-1)
                        names = _action_coordinate_names(env, int(action_values.size))
                        if action_coordinate_schema is None:
                            action_coordinate_schema = names
                        elif names != action_coordinate_schema:
                            action_coordinate_schema_mismatch = True
                        _record_action_coordinates(action_coordinate_totals, action, names)
                        obs, _reward, done, truncated, _ = env.step(action)
                    end_equities.append(float(env.account.equity))
                    trade_log = getattr(env, "trade_log", ())
                    closing_types = {
                        "close",
                        "eod_close",
                        "forced_close",
                        "stop_close",
                        "tp_close",
                    }
                    completed_trades += sum(
                        1 for trade in trade_log if trade.get("type") in closing_types
                    )
                    survival_valid = survival_valid and not bool(getattr(env, "breach_events", ()))
                    entry_diag = getattr(env, "_entry_diag", {})
                    if isinstance(entry_diag, dict):
                        for key, value in entry_diag.items():
                            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                                entry_funnel_diagnostics[str(key)] = entry_funnel_diagnostics.get(
                                    str(key), 0
                                ) + int(value)
                    state_fn = getattr(env, "entry_state_diagnostics", None)
                    if callable(state_fn):
                        state = state_fn()
                        if isinstance(state, dict):
                            for key, value in state.items():
                                if isinstance(value, (int, float, np.integer, np.floating)):
                                    entry_state[str(key)] = entry_state.get(str(key), 0) + value
                                else:
                                    entry_state[str(key)] = value
                finally:
                    close = getattr(env, "close", None)
                    if callable(close):
                        close()
            end_equity = float(np.mean(end_equities))
            finite_equity = bool(np.isfinite(end_equity))
            row: dict[str, Any] = {
                "timestep": int(self.num_timesteps),
                "deterministic_n_trades": int(completed_trades),
                "candidate_created": int(entry_state.get("candidate_created", 0)),
                "armed": int(entry_state.get("armed_bars", 0)),
                "trigger_requested": int(entry_state.get("trigger_requested", 0)),
                "trigger_refused": int(entry_state.get("trigger_refused", 0)),
                "opened": int(entry_state.get("entered", 0)),
                "closed": int(entry_state.get("closed", 0)),
                "entry_funnel_diagnostics": entry_funnel_diagnostics,
                "action_coordinate_schema": action_coordinate_schema or [],
                "action_coordinate_schema_mismatch": action_coordinate_schema_mismatch,
                "action_coordinate_diagnostics": _summarize_action_coordinates(
                    action_coordinate_totals
                ),
                "survival_valid": bool(survival_valid),
                "end_equity_finite": finite_equity,
                "entry_state_diagnostics": entry_state,
            }
            self.viability_eval_history.append(row)

            inactive = completed_trades == 0 and survival_valid and finite_equity
            self.consecutive_inactive_viability_evals = (
                self.consecutive_inactive_viability_evals + 1 if inactive else 0
            )
            if (
                inactive
                and self.num_timesteps >= 5_000_000
                and bool(entry_state.get("entry_state_machine", False))
                and int(entry_state.get("candidate_created", 0)) == 0
            ):
                self.viability_failure_reason = (
                    "FAIL_ENTRY_PATH: full training replay created no entry candidates"
                )
            elif (
                self.num_timesteps >= 10_000_000 and self.consecutive_inactive_viability_evals >= 2
            ):
                self.viability_failure_reason = (
                    "FAIL_POLICY_COLLAPSE: full training replay has zero trades "
                    "at consecutive 5M checkpoints"
                )
            if self.verbose:
                print(
                    "[TrainingViabilityEval] "
                    f"step={self.num_timesteps} trades={completed_trades} "
                    f"survival_valid={survival_valid} "
                    f"end_equity_finite={finite_equity} "
                    f"entry_state={entry_state}"
                )

    class ClipLogStdCallback(_Base):
        """Clamp PPO Gaussian ``log_std`` after updates (no-op on Discrete)."""

        def __init__(
            self,
            log_std_min: float = -0.7,
            log_std_max: float = 0.0,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose=verbose)
            self.log_std_min = float(log_std_min)
            self.log_std_max = float(log_std_max)

        def _clip(self) -> None:
            clip_policy_log_std(self.model.policy, self.log_std_min, self.log_std_max)

        def _on_training_start(self) -> None:
            self._clip()

        def _on_rollout_start(self) -> None:
            self._clip()

        def _on_step(self) -> bool:
            return True

        def _on_training_end(self) -> None:
            self._clip()

    class PeriodicCheckpointCallback(_Base):
        """Save the policy every ``save_freq`` environment timesteps.

        SB3 ``CheckpointCallback`` counts VecEnv calls, so ``n_envs=64`` made
        bar-length intervals skip most of a 20M run. This uses
        ``model.num_timesteps`` and also refreshes ``ppo_latest.zip``.
        """

        def __init__(
            self,
            save_freq: int,
            save_path: str | Path,
            name_prefix: str = "ppo_ckpt",
            latest_name: str = "ppo_latest",
            verbose: int = 1,
        ) -> None:
            super().__init__(verbose=verbose)
            self.save_freq = int(save_freq)
            self.save_path = Path(save_path)
            self.name_prefix = name_prefix
            self.latest_name = latest_name
            self._next_save = self.save_freq if self.save_freq > 0 else None

        def _init_callback(self) -> None:
            self.save_path.mkdir(parents=True, exist_ok=True)
            current = int(getattr(self.model, "num_timesteps", 0))
            if self.save_freq > 0:
                self._next_save = (current // self.save_freq + 1) * self.save_freq

        def _save(self) -> None:
            ckpt = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            save_ppo_checkpoint(self.model, ckpt)
            save_ppo_checkpoint(self.model, self.save_path / self.latest_name)
            if self.verbose:
                print(f"[PeriodicCheckpoint] timestep={self.num_timesteps} → {ckpt}.zip")

        def _on_step(self) -> bool:
            if self._next_save is None or self.save_freq <= 0:
                return True
            if self.num_timesteps >= self._next_save:
                self._save()
                crossed = 1 + (self.num_timesteps - self._next_save) // self.save_freq
                self._next_save += self.save_freq * crossed
            return True

    class EarlyAbortCallback(_Base):
        """Stop ``learn()`` when training is invalid or recent episodes fail.

        Returning ``False`` from ``_on_step`` is the Stable-Baselines3 stop.
        Equity and no-trade checks wait until ``min_timesteps``. A non-finite
        logged metric or episode equity stops immediately.
        """

        def __init__(
            self,
            min_timesteps: int = 5_000_000,
            window: int = 4,
            stop_on_nonfinite: bool = True,
            stop_on_no_trades: bool = False,
            stop_on_equity_gate: bool = False,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose=verbose)
            self.min_timesteps = int(min_timesteps)
            self.window = int(window)
            self.stop_on_nonfinite = bool(stop_on_nonfinite)
            self.stop_on_no_trades = bool(stop_on_no_trades)
            self.stop_on_equity_gate = bool(stop_on_equity_gate)
            self._episodes: list[dict[str, Any]] = []
            self._nonfinite = False
            self.reason: str | None = None

        def _note_episode(self, ep: dict[str, Any]) -> None:
            for key in (
                "end_equity",
                "start_equity",
                "slope",
                "max_peak_trailing_dd",
                "reward_sum",
            ):
                if key not in ep:
                    continue
                try:
                    if not np.isfinite(float(ep[key])):
                        self._nonfinite = True
                except (TypeError, ValueError):
                    self._nonfinite = True
            self._episodes.append(ep)

        def _on_rollout_end(self) -> None:
            if not self.stop_on_nonfinite:
                return
            logger = getattr(self.model, "logger", None)
            values = getattr(logger, "name_to_value", {}) if logger is not None else {}
            for key, value in values.items():
                # SB3 sets this to NaN when every return in the rollout is 0.
                if str(key).endswith("explained_variance"):
                    continue
                try:
                    if not np.isfinite(float(value)):
                        self._nonfinite = True
                        return
                except (TypeError, ValueError):
                    continue

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            infos = self.locals.get("infos") or []
            if dones is not None:
                for i, done in enumerate(np.asarray(dones).reshape(-1)):
                    if not done or i >= len(infos):
                        continue
                    info = infos[i]
                    ep = info.get("episode_equity") if isinstance(info, dict) else None
                    if ep:
                        self._note_episode(ep)
            reason = decide_early_abort(
                self._episodes,
                num_timesteps=int(self.num_timesteps),
                min_timesteps=self.min_timesteps,
                window=self.window,
                nonfinite=self._nonfinite,
                stop_on_nonfinite=self.stop_on_nonfinite,
                stop_on_no_trades=self.stop_on_no_trades,
                stop_on_equity_gate=self.stop_on_equity_gate,
            )
            if reason is None:
                return True
            self.reason = reason
            log.error("EARLY_ABORT %s at timestep=%s", reason, self.num_timesteps)
            return False

else:

    class LearningDiagnosticsCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("stable-baselines3 is required for LearningDiagnosticsCallback")

    class ProgressLoggerCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for ProgressLoggerCallback. "
                "Install it with: pip install stable-baselines3"
            )

    class EpisodeEquityCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for EpisodeEquityCallback. "
                "Install it with: pip install stable-baselines3"
            )

    class BestCheckpointEvalCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for BestCheckpointEvalCallback. "
                "Install it with: pip install stable-baselines3"
            )

    class ClipLogStdCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for ClipLogStdCallback. "
                "Install it with: pip install stable-baselines3"
            )

    class PeriodicCheckpointCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for PeriodicCheckpointCallback. "
                "Install it with: pip install stable-baselines3"
            )

    class LagrangianRiskCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("stable-baselines3 is required for LagrangianRiskCallback")

    class EarlyAbortCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for EarlyAbortCallback. "
                "Install it with: pip install stable-baselines3"
            )
