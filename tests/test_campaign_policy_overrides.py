from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.campaigns.execution import (
    campaign_config_override_lines,
    normalize_campaign_config_overrides,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_SPEC = (
    ROOT / "config/campaigns/rq1_entry_fsm_pair_seed50_train2025_20261024_policy2m_v1.yaml"
)


def test_campaign_policy_overrides_are_sorted_and_shell_safe() -> None:
    values = normalize_campaign_config_overrides(
        {
            "ftmo.trailing_dd_limit": 0.0,
            "ftmo.daily_loss_limit": 5000.0,
            "env.peak_trailing_dd_limit": 0.0,
            "enabled": False,
        }
    )

    assert list(values) == [
        "enabled",
        "env.peak_trailing_dd_limit",
        "ftmo.daily_loss_limit",
        "ftmo.trailing_dd_limit",
    ]
    assert campaign_config_override_lines(values).splitlines() == [
        "enabled=false",
        "env.peak_trailing_dd_limit=0.0",
        "ftmo.daily_loss_limit=5000.0",
        "ftmo.trailing_dd_limit=0.0",
    ]


@pytest.mark.parametrize(
    "value",
    [
        ["ftmo.trailing_dd_limit=0"],
        {"bad key": 0},
        {"ftmo.trailing_dd_limit": float("nan")},
        {"ftmo.trailing_dd_limit": "0"},
    ],
)
def test_campaign_policy_override_validation_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ValueError):
        normalize_campaign_config_overrides(value)


def test_approved_2m_policy_spec_uses_each_variants_policy_yaml() -> None:
    spec = yaml.safe_load(POLICY_SPEC.read_text(encoding="utf-8"))

    assert spec["timesteps"] == 2_000_000
    assert spec["seeds"] == [50]
    assert spec["evaluation"]["deterministic"] is False
    assert "config_overrides" not in spec
    for name in spec["variants"]:
        policy = yaml.safe_load((ROOT / "config/variants" / f"{name}.yaml").read_text())
        assert policy["variant"] == name
        assert policy["ftmo"]["limit_value_mode"] == "fixed_usd"
        assert policy["ftmo"]["daily_loss_limit"] == 5_000.0
        assert policy["ftmo"]["soft_daily_loss_limit"] == 5_000.0
        assert policy["ftmo"]["max_loss_limit"] == 10_000.0
        assert policy["ftmo"]["soft_max_loss_limit"] == 10_000.0
        assert policy["ftmo"]["risk_per_trade_limit"] == 1_000.0
        assert policy["ftmo"]["risk_sizing_limit_usd"] == 900.0
        assert policy["ftmo"]["max_loss_limit_initial_pct"] == 0.10
