"""Shared ablation utilities for train_rl.py and ablation_runner.py.

Extracts the variant config merge logic and Tier 1 entry-state funnel
metrics so both entrypoints stay in sync without circular imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STRATEGY_CONFIGS = {
    "po3_ifvg": _PROJECT_ROOT / "config" / "2_po3_ifvg.yaml",
    "distribution": _PROJECT_ROOT / "config" / "3_distribution.yaml",
}
_PD_CONTEXT_CONFIG = _PROJECT_ROOT / "config" / "features" / "pd_context.yaml"
_EXPERIMENTS_CONFIG = _PROJECT_ROOT / "config" / "experiments.yaml"

_LADDER_FLAGS: tuple[str, ...] = (
    "entry_state_machine",
    "entry_state_observation",
    "allow_agent_sl_mode",
    "allow_agent_tp_mode",
    "allow_multi_tp",
    "agent_direction_control",
    "strategy_actions",
    "arm_requires_retest",
    "candidate_max_age_bars",
)


def load_variant_config(
    variant_name: str,
    experiments_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a single variant dict from experiments.yaml.

    Args:
        variant_name: Name of the variant to look up.
        experiments_path: Path to experiments YAML (default: config/experiments.yaml).

    Returns:
        The variant dict.

    Raises:
        SystemExit: If the variant is not found.
    """
    path = Path(experiments_path) if experiments_path else _EXPERIMENTS_CONFIG
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping in {path}")
    for variant in raw.get("variants", []):
        if str(variant.get("name")) == variant_name:
            return dict(variant)
    raise SystemExit(f"unknown variant: {variant_name}")


def merge_variant_cfg(
    base: DictConfig,
    *,
    strategy: str,
    include_pd_context: bool,
    variant: dict[str, Any] | None = None,
) -> DictConfig:
    """Merge strategy / PD-context overlays onto a base config.

    This is extracted from ablation_runner.merge_variant_cfg so train_rl.py
    can apply the same variant semantics without importing from scripts/.
    """
    cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(base, resolve=True)))
    if include_pd_context:
        pd_path = _PD_CONTEXT_CONFIG
        if pd_path.is_file():
            cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.load(pd_path)))
    if strategy in _STRATEGY_CONFIGS:
        strat_path = _STRATEGY_CONFIGS[strategy]
        if strat_path.is_file():
            cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.load(strat_path)))
    for key in _LADDER_FLAGS:
        if variant is not None and key in variant:
            cfg.env[key] = variant[key]
    if variant is not None and bool(variant.get("strategy_actions", False)):
        cfg.features.include_strategy_state = True
    return cfg


def compute_tier1_funnel(
    entry_diag: dict[str, Any] | None,
    entry_state_diag: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute Tier 1 entry-state funnel metrics from diagnostic dicts.

    Combines the lower-granularity ``entry_diag`` (from evaluate_model's
    ``_entry_diag``) with the FSM lifecycle counters from
    ``entry_state_diagnostics()``.

    Returns a flat dict suitable for inclusion in training_log.json.
    """
    entry_diag = entry_diag or {}
    entry_state_diag = entry_state_diag or {}

    opened = int(entry_diag.get("opened", 0))
    attempts = opened + sum(v for k, v in entry_diag.items() if k.startswith("rejected_"))

    funnel: dict[str, Any] = {
        "entry_opened": opened,
        "entry_attempts": attempts,
        "entry_rejection_rate": round(1.0 - opened / attempts, 6) if attempts else 0.0,
    }

    for key, value in entry_diag.items():
        if key.startswith("rejected_"):
            funnel[key] = int(value)

    for key in (
        "candidate_created",
        "candidate_expired",
        "candidate_invalidated",
        "trigger_requested",
        "trigger_refused",
        "entered",
        "closed",
        "armed_bars",
        "arm_to_trigger_rate",
        "trigger_refusal_rate",
    ):
        if key in entry_state_diag:
            funnel[key] = entry_state_diag[key]

    return funnel
