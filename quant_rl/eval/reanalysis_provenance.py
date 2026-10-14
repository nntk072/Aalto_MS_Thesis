"""Lightweight, fail-closed provenance checks for historical run reanalysis.

Keep these guards importable without the evaluation/training runtime so they
can be tested before an expensive checkpoint or data pipeline is loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from quant_rl.data.split import make_train_mask
from quant_rl.features.build import feature_cache_path

log = logging.getLogger(__name__)


def _load_run_config(run_dir: Path) -> DictConfig:
    """Require the original run snapshot for checkpoint evaluation parity."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Run config snapshot missing: {cfg_path}; refusing to evaluate "
            "a checkpoint against unrelated default settings"
        )
    log.info("Loading config snapshot: %s", cfg_path)
    cfg = OmegaConf.load(cfg_path)
    if not isinstance(cfg, DictConfig) or not cfg:
        raise ValueError(
            f"Run config snapshot must be a nonempty configuration mapping: {cfg_path}"
        )
    # get_split_config() deliberately falls back to hard-coded defaults for
    # legacy training callers. Historical checkpoint evaluation must instead
    # use the date boundaries saved with the checkpoint.
    for field in ("train_end", "test_start"):
        value = OmegaConf.select(cfg, f"data.split.{field}", default=None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Run config snapshot is missing explicit data.split.{field}: {cfg_path}"
            )
    return cfg


def _validate_reanalysis_request(
    source: Path, output: Path, *, allow_historical: bool = False
) -> None:
    """Require holdout acknowledgement and an entirely new output location."""
    if not allow_historical:
        raise ValueError("historical holdout reanalysis requires --allow-historical")
    source_path, output_path = source.resolve(), output.resolve()
    if source_path == output_path or source_path in output_path.parents:
        raise ValueError("reanalysis requires a separate output outside the source run")
    if output_path.exists():
        raise FileExistsError(f"reanalysis output already exists: {output_path}")


def _feature_build_scope(
    bars: Any,
    cfg: DictConfig,
    cache_dir: Path,
    symbol: str,
    train_end: str,
    secondary: Any | None = None,
) -> tuple[Any, Path]:
    """Reproduce the mask and content-hashed cache identity of training."""
    mask = make_train_mask(cast(Any, bars.index), train_end)
    return mask, feature_cache_path(
        cache_dir, symbol, cfg, bars, train_mask=mask, secondary=secondary
    )
