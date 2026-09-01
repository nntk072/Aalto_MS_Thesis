"""Configuration loading typo-safety tests (Master Roadmap Stage 4)."""

from __future__ import annotations

import pytest

from quant_rl.config import load_config


def test_typoed_override_raises() -> None:
    """A typo like ``ppo.leraning_rate`` must raise instead of silently
    creating a stray config key that `build_agent` ignores."""
    with pytest.raises(ValueError, match="does not exist in default.yaml"):
        load_config(["ppo.leraning_rate=1e-4"])


def test_missing_section_raises() -> None:
    """A key under a non-existent section must also be rejected."""
    with pytest.raises(ValueError, match="does not exist"):
        load_config(["ppo2.learning_rate=1e-4"])


def test_correct_override_applies() -> None:
    """The real key still type-coerces and lands in the merged config."""
    cfg = load_config(["ppo.learning_rate=1e-4", "training.max_days=10"])
    assert cfg.ppo.learning_rate == 1e-4
    assert cfg.training.max_days == 10


def test_variant_config_merge_is_not_flagged() -> None:
    """Chain B/F variant fragments (features.* keys) load via config_path,
    *before* overrides, so overriding their keys must not raise."""
    overrides = ["features.include_po3=false", "ppo.batch_size=128"]
    cfg = load_config(overrides)  # no variant config needed: features.* is in default.yaml
    assert cfg.features.include_po3 is False
    assert cfg.ppo.batch_size == 128


def test_null_valued_leaf_override_is_allowed() -> None:
    """Explicit ``null`` leaves in default.yaml must remain overridable
    (they are present-day leaves, not typos)."""
    cfg = load_config(
        ["training.max_days=14", "backtest.validation.take_profit_per_trade_usd=50.0"]
    )
    assert cfg.training.max_days == 14
    assert cfg.backtest.validation.take_profit_per_trade_usd == 50.0
