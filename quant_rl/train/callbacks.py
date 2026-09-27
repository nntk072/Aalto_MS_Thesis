"""SB3 training callbacks for progress logging and best-checkpoint eval."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_rl.models.ppo_policy import clip_policy_log_std
from quant_rl.train.equity_gate import decide_early_abort

log = logging.getLogger(__name__)


def save_ppo_checkpoint(model: Any, path: str | Path) -> None:
    """Save PPO weights; omit the dashboard ``train`` hook (unpickleable on Py 3.12+)."""
    model.save(path, exclude=["train"])


try:
    from stable_baselines3.common.callbacks import BaseCallback as _Base

    _SB3_AVAILABLE = True
except ImportError:
    _Base = object  # type: ignore[assignment,misc]
    _SB3_AVAILABLE = False


if _SB3_AVAILABLE:

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
            self._rows: list[dict[str, Any]] = []

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            infos = self.locals.get("infos") or []
            if dones is None:
                return True
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
                    "end_equity": end_equity,
                    "start_equity": start_equity,
                    "equity_delta": equity_delta,
                    "reward_sum": reward_sum,
                    "n_trades": int(ep["n_trades"]),
                    "breach_reason": str(ep.get("breach_reason") or ""),
                }
                self._rows.append(row)
                if reward_sum > 0.0 and end_equity < start_equity:
                    log.warning(
                        "episode reward %.4f is positive but equity fell %.2f (breach=%s)",
                        reward_sum,
                        -equity_delta,
                        row["breach_reason"] or "none",
                    )
            return True

        def _on_training_end(self) -> None:
            if self._rows:
                pd.DataFrame(self._rows).to_csv(self._log_path, index=False)

    class BestCheckpointEvalCallback(_Base):
        """Save the policy with the highest end equity on a train-calendar rollout.

        The score is the account equity at the end of the rollout, not the
        shaped episode reward. A higher reward with a lower balance does not
        replace the saved checkpoint.

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
        ) -> None:
            super().__init__(verbose=verbose)
            self._eval_env_factory = eval_env_factory
            self._eval_freq = int(eval_freq)
            self._best_model_path = Path(best_model_path)
            self._best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self._n_eval_episodes = int(n_eval_episodes)
            self.best_end_equity: float = -np.inf
            # Kept in sync with ``best_end_equity`` so older callers still read a score.
            self.best_mean_reward: float = -np.inf

        def _on_step(self) -> bool:
            # ``num_timesteps`` is the SB3-standard global step counter.
            if self._eval_freq > 0 and self.num_timesteps % self._eval_freq == 0:
                self._run_eval()
            return True

        def _run_eval(self) -> None:
            end_equities: list[float] = []
            for _ in range(self._n_eval_episodes):
                env = self._eval_env_factory()
                obs, _ = env.reset()
                done = truncated = False
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _reward, done, truncated, _ = env.step(action)
                end_equities.append(float(env.account.equity))
            end_equity = float(np.mean(end_equities))
            if end_equity > self.best_end_equity:
                self.best_end_equity = end_equity
                self.best_mean_reward = end_equity
                save_ppo_checkpoint(self.model, self._best_model_path)
                if self.verbose:
                    print(
                        f"[BestCheckpointEval] new best end_equity={end_equity:.2f} at "
                        f"timestep={self.num_timesteps} → saved {self._best_model_path}"
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
            min_timesteps: int = 2_000_000,
            window: int = 4,
            stop_on_nonfinite: bool = True,
            stop_on_no_trades: bool = True,
            stop_on_equity_gate: bool = True,
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
            for value in values.values():
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

    class EarlyAbortCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for EarlyAbortCallback. "
                "Install it with: pip install stable-baselines3"
            )
