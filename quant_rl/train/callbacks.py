"""SB3 training callbacks for progress logging and best-checkpoint eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

    class BestCheckpointEvalCallback(_Base):
        """Periodically evaluate on a held-out validation split and save the
        best model by mean episode reward.

        PPO's default behavior saves a final model at the end of training,
        but the final model is rarely the best one — a noisy single rollout
        can pick a local minimum. This callback wraps a per-N-rollouts
        evaluation on a fresh env (with ``episodic=False`` so a single
        guardrail breach does not kill the whole evaluation) and copies
        the policy weights to ``best_model_path`` whenever the new
        evaluation improves on the running best.

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
            self.best_mean_reward: float = -np.inf

        def _on_step(self) -> bool:
            # ``num_timesteps`` is the SB3-standard global step counter.
            if self._eval_freq > 0 and self.num_timesteps % self._eval_freq == 0:
                self._run_eval()
            return True

        def _run_eval(self) -> None:
            episode_rewards: list[float] = []
            for _ in range(self._n_eval_episodes):
                env = self._eval_env_factory()
                obs, _ = env.reset()
                ep_reward = 0.0
                done = truncated = False
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = env.step(action)
                    ep_reward += float(reward)
                episode_rewards.append(ep_reward)
            mean_reward = float(np.mean(episode_rewards))
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self._best_model_path)
                if self.verbose:
                    print(
                        f"[BestCheckpointEval] new best={mean_reward:.3f} at "
                        f"timestep={self.num_timesteps} → saved {self._best_model_path}"
                    )

else:

    class ProgressLoggerCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for ProgressLoggerCallback. "
                "Install it with: pip install stable-baselines3"
            )

    class BestCheckpointEvalCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for BestCheckpointEvalCallback. "
                "Install it with: pip install stable-baselines3"
            )
