"""Configuration loading via OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

_DEFAULT_CFG = Path(__file__).parent / "default.yaml"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load default config, then apply string overrides like key=value."""
    cfg = cast(DictConfig, OmegaConf.load(_DEFAULT_CFG))
    if overrides:
        for ov in overrides:
            key, _, val = ov.partition("=")
            # Try to coerce to int/float/bool before storing
            coerced: object = val
            try:
                coerced = int(val)
            except ValueError:
                try:
                    coerced = float(val)
                except ValueError:
                    if val.lower() in ("true", "false"):
                        coerced = val.lower() == "true"
            OmegaConf.update(cfg, key.strip(), coerced, merge=True)
    return cfg
