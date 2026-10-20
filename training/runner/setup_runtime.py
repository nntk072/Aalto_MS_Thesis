"""Prepare data, configuration, and environment assets for an RL training run."""

from __future__ import annotations

import logging

from .configuration_stage import _prepare_training_configuration
from .data_stage import _prepare_training_data
from .run_assets_stage import _prepare_run_assets
from .stages import TrainingPreparation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

__all__ = ["TrainingPreparation", "prepare_training_run"]


def prepare_training_run() -> TrainingPreparation:
    """Resolve config, data/features, and run assets before any policy learning starts."""
    config_stage = _prepare_training_configuration()
    data_stage = _prepare_training_data(config_stage)
    return _prepare_run_assets(config_stage, data_stage)
