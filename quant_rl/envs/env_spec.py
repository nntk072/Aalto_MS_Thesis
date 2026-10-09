"""Small, pickleable description of one bundle-backed environment worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class EnvSpec:
    """Configuration needed to recreate an env without shipping training data."""

    bundle_dir: str
    algo: str
    reward: str
    arch: str
    config: Mapping[str, Any]
    episodic: bool = True
    use_vae: bool = False

    @classmethod
    def from_cfg(
        cls,
        cfg: Mapping[str, Any] | DictConfig,
        *,
        algo: str,
        reward: str,
        arch: str,
        bundle_dir: str,
        episodic: bool = True,
        use_vae: bool = False,
    ) -> EnvSpec:
        """Copy resolved config primitives into a compact immutable worker spec."""
        raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
        if not isinstance(raw, Mapping):
            raise TypeError("environment config must be a mapping")
        return cls(
            bundle_dir=str(bundle_dir),
            algo=str(algo),
            reward=str(reward),
            arch=str(arch),
            config={str(key): value for key, value in raw.items()},
            episodic=bool(episodic),
            use_vae=bool(use_vae),
        )
