"""Configuration loading via OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

_DEFAULT_CFG = Path(__file__).parent / "default.yaml"


def load_config(
    overrides: list[str] | None = None,
    config_path: str | Path | None = None,
) -> DictConfig:
    """Load config from ``config_path`` (default.yaml when None), then apply overrides.

    Parameters
    ----------
    overrides : list[str], optional
        ``key=value`` string overrides applied on top of the loaded YAML.
    config_path : str | Path, optional
        YAML config file to load instead of ``quant_rl/config/default.yaml``.
        Lets ``quant_rl/train/train_rl.py`` load the Chain B variant configs
        (``config/features_*_mtf.yaml``) just like ``scripts/train_rl.py``.
    """
    cfg_path = Path(config_path) if config_path else _DEFAULT_CFG
    base = cast(DictConfig, OmegaConf.load(_DEFAULT_CFG))
    extra = cast(DictConfig, OmegaConf.load(cfg_path)) if config_path else OmegaConf.create()
    # Merge over default.yaml: works both for Chain B/F variant *fragments*
    # (config/features_*_mtf.yaml only override the features.* keys they set,
    # everything else comes from defaults) and for full standalone configs.
    cfg = cast(DictConfig, OmegaConf.merge(base, extra))
    if overrides:
        # Snapshot the *pre-override* merged config so variant-config merges
        # (which happen above, before overrides) are not flagged as typos.
        known_paths = _leaf_paths(cfg)
        for ov in overrides:
            key = ov.partition("=")[0].strip()
            if key and key not in known_paths:
                raise ValueError(
                    f"override key {key!r} does not exist in default.yaml — check for a typo"
                )
            _, _, val = ov.partition("=")
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


def _leaf_paths(cfg: DictConfig, prefix: str = "") -> set[str]:
    """Return all dot-paths to *leaf* keys (including explicit ``null`` leaves)."""
    paths: set[str] = set()
    for k in cfg.keys():
        key = str(k)
        path = f"{prefix}.{key}" if prefix else key
        child = cfg[k]
        if isinstance(child, DictConfig):
            paths |= _leaf_paths(child, path)
        else:
            paths.add(path)
    return paths
