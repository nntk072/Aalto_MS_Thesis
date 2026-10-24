"""Per-variant FTMO policy YAML loading and validation."""

from __future__ import annotations

import math
from typing import Any

import yaml

from configuration.paths import repository_root

PROJECT_ROOT = repository_root(__file__)
VARIANT_POLICY_DIR = PROJECT_ROOT / "config" / "variants"

_NUMERIC_FIELDS = {
    "daily_loss_limit",
    "soft_daily_loss_limit",
    "max_loss_limit",
    "soft_max_loss_limit",
    "risk_per_trade_limit",
    "risk_sizing_limit_usd",
    "trailing_dd_limit",
    "soft_trailing_dd_limit",
    "risk_per_trade_limit_pct",
    "daily_loss_limit_pct",
    "soft_daily_loss_limit_pct",
    "max_loss_limit_initial_pct",
    "soft_max_loss_limit_initial_pct",
}
_REQUIRED_FIELDS = _NUMERIC_FIELDS | {"limit_value_mode"}
_PERCENT_FIELDS = {
    "risk_per_trade_limit_pct",
    "daily_loss_limit_pct",
    "soft_daily_loss_limit_pct",
    "max_loss_limit_initial_pct",
    "soft_max_loss_limit_initial_pct",
}


def load_variant_ftmo_config(name: str, *, required: bool = True) -> dict[str, Any]:
    """Load one variant's explicit risk settings from ``config/variants``.

    A separate file makes the active limits and both selectable value modes
    inspectable for each variant. Missing files are allowed only for callers
    resolving temporary experiment registries.
    """
    path = VARIANT_POLICY_DIR / f"{name}.yaml"
    if not path.is_file():
        if required:
            raise ValueError(f"missing per-variant policy YAML: {path}")
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load per-variant policy YAML {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("variant") != name:
        raise ValueError(f"{path} must declare variant: {name}")
    ftmo = raw.get("ftmo")
    if not isinstance(ftmo, dict):
        raise ValueError(f"{path} must contain an ftmo mapping")
    missing = _REQUIRED_FIELDS - set(ftmo)
    extra = set(ftmo) - _REQUIRED_FIELDS
    if missing or extra:
        raise ValueError(
            f"{path} ftmo fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if ftmo["limit_value_mode"] not in {"fixed_usd", "percent_equity"}:
        raise ValueError(f"{path} has invalid ftmo.limit_value_mode")
    normalized: dict[str, Any] = {"limit_value_mode": str(ftmo["limit_value_mode"])}
    for key in sorted(_NUMERIC_FIELDS):
        value = ftmo[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} ftmo.{key} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{path} ftmo.{key} must be finite and non-negative")
        if key in _PERCENT_FIELDS and number > 1.0:
            raise ValueError(f"{path} ftmo.{key} must be a fraction in [0, 1]")
        normalized[key] = number
    return normalized


def variant_ftmo_overrides(name: str, *, required: bool = True) -> list[str]:
    """Return flat CLI overrides for one validated per-variant policy file."""
    values = load_variant_ftmo_config(name, required=required)
    return [f"ftmo.{key}={value}" for key, value in values.items()]


__all__ = ["VARIANT_POLICY_DIR", "load_variant_ftmo_config", "variant_ftmo_overrides"]
