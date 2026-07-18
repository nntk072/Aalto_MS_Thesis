"""Configuration loading via OmegaConf."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


_DEFAULT_CFG = Path(__file__).parent / "default.yaml"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load default config, then apply string overrides like key=value."""
    cfg: DictConfig = OmegaConf.load(_DEFAULT_CFG)
    if overrides:
        for ov in overrides:
            key, _, val = ov.partition("=")
            OmegaConf.update(cfg, key.strip(), val.strip(), merge=True)
    return cfg
