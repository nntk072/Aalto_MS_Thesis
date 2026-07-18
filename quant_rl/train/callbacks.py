"""SB3 training callbacks for progress logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

else:

    class ProgressLoggerCallback:  # type: ignore[no-redef]
        """Stub — stable-baselines3 is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "stable-baselines3 is required for ProgressLoggerCallback. "
                "Install it with: pip install stable-baselines3"
            )
