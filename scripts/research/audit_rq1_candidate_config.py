#!/usr/bin/env python3
"""Audit the provisional RQ1 gate pair through the production config merger.

This is a configuration preflight only. It does not freeze an RQ1 pair, train a
policy, or evaluate a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from quant_rl.action_schema import build_action_schema
from quant_rl.train.train_rl import _load_merged_config
from quant_rl.train.variant_resolver import resolve_variant

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "ladder_a0_flat_po3"
GATE_PATH = "strategy.entry.enforce_gate"


def _config_for_gate(gate_enabled: bool) -> dict[str, Any]:
    resolved = resolve_variant(VARIANT)
    overrides = [
        item for item in resolved["overrides"] if not item.startswith("ppo.total_timesteps=")
    ]
    overrides.extend(
        [
            "env.agent_direction_control=true",
            "env.entry_state_machine=false",
            f"strategy.entry.enforce_gate={'true' if gate_enabled else 'false'}",
            "execution.randomized_fill_latency_ms=30.0",
        ]
    )
    args = argparse.Namespace(
        config=resolved["base_config"],
        strategy=resolved["strategy"],
        variant=VARIANT,
        quality_variant=None,
        overrides=overrides,
    )
    cfg = _load_merged_config(args)
    value = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError("resolved configuration must be a mapping")
    return cast(dict[str, Any], value)


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            paths.extend(_diff_paths(left_item, right_item, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    control = _config_for_gate(False)
    treatment = _config_for_gate(True)
    changed_paths = _diff_paths(control, treatment)
    expected_changed_paths = [GATE_PATH]
    if changed_paths != expected_changed_paths:
        raise RuntimeError(f"unexpected control/treatment config diff: {changed_paths}")

    env = control["env"]
    data = control["data"]
    strategy = control["strategy"]
    action_schema = build_action_schema(
        has_direction=bool(env.get("agent_direction_control", False)),
        has_sl_mode=bool(env.get("allow_agent_sl_mode", False)),
        has_tp_mode=bool(env.get("allow_agent_tp_mode", False)),
        has_multi_tp=bool(env.get("allow_multi_tp", False)),
        has_simplex=bool(env.get("allow_simplex", False))
        and bool(env.get("allow_multi_tp", False)),
    ).to_dict()

    required = {
        "strategy": strategy.get("name") == "po3_ifvg",
        "confirmed_first_eligible": strategy.get("entry", {}).get("mode")
        == "confirmed_first_eligible",
        "strategy_actions": bool(env.get("strategy_actions", False)),
        "shared_direction_control": bool(env.get("agent_direction_control", False)),
        "entry_fsm_disabled": not bool(env.get("entry_state_machine", False)),
        "six_coordinate_schema": action_schema.get("width") == 6
        and action_schema.get("key") == "6_dir",
        "matching_feature_cache": data.get("prebuilt_feature_cache")
        == "data/prebuilt_feature_cache/idea1_po3_ifvg.parquet",
        "new_full_period_us100_m1": data["m1_files"]["US100.cash"].endswith("202610162349.csv"),
        "new_full_period_us500_m1": data["m1_files"]["US500.cash"].endswith("202610162349.csv"),
        "new_full_period_us100_ticks": data["tick_files"]["US100.cash"].endswith(
            "202610162349.csv"
        ),
        "new_full_period_us500_ticks": data["tick_files"]["US500.cash"].endswith(
            "202610162349.csv"
        ),
        "train_through_2025": data["split"]["train_end"] == "2025-12-31",
        "test_from_2026": data["split"]["test_start"] == "2026-01-01",
        "seeded_latency_below_30ms": control["execution"]["randomized_fill_latency_ms"] == 30.0,
        "tick_execution_enabled": bool(control["costs"].get("use_tick_execution", False)),
        "quote_wait_60s": control["execution"]["max_tick_quote_wait_ms"] == 60_000,
        "control_gate_off": control["strategy"]["entry"]["enforce_gate"] is False,
        "treatment_gate_on": treatment["strategy"]["entry"]["enforce_gate"] is True,
    }
    failed = sorted(key for key, passed in required.items() if not passed)
    if failed:
        raise RuntimeError(f"RQ1 candidate preflight requirements failed: {failed}")

    shared = dict(control)
    shared_strategy = dict(shared["strategy"])
    shared_entry = dict(shared_strategy["entry"])
    shared_entry.pop("enforce_gate")
    shared_strategy["entry"] = shared_entry
    shared["strategy"] = shared_strategy
    report = {
        "schema_version": 1,
        "status": "candidate_config_preflight_passed",
        "role": "provisional_only_not_frozen_not_trained_not_evaluated",
        "variant_template": VARIANT,
        "candidate_intervention": GATE_PATH,
        "changed_config_paths": changed_paths,
        "control_gate": False,
        "treatment_gate": True,
        "shared_settings_sha256": _sha256(shared),
        "control_config_sha256": _sha256(control),
        "treatment_config_sha256": _sha256(treatment),
        "action_schema": action_schema,
        "cache": data["prebuilt_feature_cache"],
        "market_inputs": {
            "m1_files": data["m1_files"],
            "tick_files": data["tick_files"],
            "train_end": data["split"]["train_end"],
            "test_start": data["split"]["test_start"],
        },
        "execution": control["execution"],
        "checks": required,
        "excluded_from_preflight": [
            "seed and training budget freeze",
            "P4 validity and independent review acceptance",
            "real-data model action trace",
            "training, checkpoint selection, and performance claims",
        ],
        "source_files": [
            "config/experiments.yaml",
            "config/2_po3_ifvg.yaml",
            "config/features/full_po3_mtf.yaml",
            "quant_rl/config/default.yaml",
            "quant_rl/train/train_rl.py",
            "quant_rl/train/variant_resolver.py",
            "quant_rl/action_schema.py",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={report['status']} changed_paths={changed_paths} schema={action_schema['key']}")
    print(f"report={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
