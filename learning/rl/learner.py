"""Stable-Baselines3 implementation of the workflow-facing learner port."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from artifacts.contracts import ArtifactRef
from learning.contracts import FitRequest, FitResult, Learner


def save_sb3_checkpoint(model: Any, path: str | Path) -> None:
    """Save SB3 weights without serializing unpickleable ``train``/``learn`` hooks."""
    model.save(path, exclude=["train", "learn"])


class SB3Learner:
    """Fit one prepared SB3 model/environment pair and persist its checkpoint."""

    def __init__(
        self,
        model: Any,
        *,
        callbacks: Any,
        checkpoint_path: str | Path,
        log_interval: int,
        reset_num_timesteps: bool,
    ) -> None:
        self.model = model
        self.callbacks = callbacks
        self.checkpoint_path = Path(checkpoint_path)
        self.log_interval = log_interval
        self.reset_num_timesteps = reset_num_timesteps

    def fit(self, request: FitRequest) -> FitResult:
        """Run the requested remaining steps and return a content-addressed checkpoint."""
        if request.requested_steps:
            self.model.learn(
                total_timesteps=request.requested_steps,
                callback=self.callbacks,
                log_interval=self.log_interval,
                reset_num_timesteps=self.reset_num_timesteps,
            )
        save_sb3_checkpoint(self.model, self.checkpoint_path)
        saved_path = Path(f"{self.checkpoint_path}.zip")
        if not saved_path.is_file():
            saved_path = self.checkpoint_path
        digest_builder = hashlib.sha256()
        with saved_path.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        return FitResult(
            artifact=ArtifactRef(
                key=f"checkpoint:{saved_path.name}",
                media_type="application/zip",
                sha256=digest,
            ),
            dataset=request.dataset,
            config_identity=request.config_identity,
            input_schema_version=request.input_schema_version,
            requested_steps=request.requested_steps,
            training_steps=int(self.model.num_timesteps),
        )


def build_sb3_learner(
    model: Any,
    *,
    callbacks: Any,
    checkpoint_path: str | Path,
    log_interval: int,
    reset_num_timesteps: bool,
) -> Learner:
    """Construct the current learner backend behind the stable training port."""
    return SB3Learner(
        model,
        callbacks=callbacks,
        checkpoint_path=checkpoint_path,
        log_interval=log_interval,
        reset_num_timesteps=reset_num_timesteps,
    )


__all__ = ["SB3Learner", "build_sb3_learner", "save_sb3_checkpoint"]
