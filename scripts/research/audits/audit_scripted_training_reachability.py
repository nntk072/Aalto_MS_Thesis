#!/usr/bin/env python3
"""Run a deterministic, full-2025 scripted entry/exit trace on resolved configs.

This is a mechanical reachability preflight, not a policy or performance test.
It loads data through the production training stages and evaluates only the
training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from configuration.paths import repository_root
from market_data.loading.ticks import ticks_covering
from training.configuration.variant_resolution import resolve_variant, train_args
from training.runner.configuration_stage import _prepare_training_configuration
from training.runner.data_stage import _prepare_training_data
from training.runner.seed import _setup_rngs
from training.setup.environments import make_env

ROOT = repository_root(__file__)
DEFAULT_VARIANTS = ("ladder_a4b_entry_fsm", "ladder_a4a_entry_fsm_hidden")
CLOSING_TYPES = {"close", "eod_close", "forced_close", "stop_close", "tp_close"}
log = logging.getLogger("scripted_training_reachability")


def _config_diff(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        changes: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                changes.append(path)
            else:
                changes.extend(_config_diff(left[key], right[key], path))
        return changes
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        return [
            path
            for index, (a, b) in enumerate(zip(left, right, strict=True))
            for path in _config_diff(a, b, f"{prefix}[{index}]")
        ]
    return [] if left == right else [prefix]


def _resolved_stage(name: str, seed: int) -> Any:
    resolved = resolve_variant(name, steps=100_000, seeds=[seed])
    previous_argv = sys.argv
    try:
        sys.argv = ["train_rl.py", *train_args(resolved, seed)]
        stage = _prepare_training_configuration()
    finally:
        sys.argv = previous_argv
    return stage


def _scripted_action(env: Any) -> np.ndarray[Any, Any]:
    schema = env.action_schema
    coordinates = list(schema.to_dict()["coordinates"])
    width = int(schema.width)
    if tuple(env.action_space.shape) != (width,):
        raise ValueError("resolved action schema and environment action space disagree")
    action = np.zeros(width, dtype=np.float32)
    for index, name in enumerate(coordinates):
        if name == "intensity":
            action[index] = 1.0
        elif name in {"sl_mode", "tp_mode"}:
            action[index] = -1.0  # explicit fixed mode; no risk-limit override
        elif name == "exit":
            action[index] = 1.0  # allow the configured EMA exit path
        # Neutral direction defers to the resolved strategy context. Risk, stop,
        # and target controls remain at their schema-neutral midpoint.
    return action


def _run_variant(
    name: str,
    stage: Any,
    data: Any,
    *,
    seed: int,
    progress_every: int,
) -> dict[str, Any]:
    args = stage.args
    cfg = stage.cfg
    _setup_rngs(seed)
    env = make_env(
        data.train_bars,
        data.train_feat,
        cfg,
        algo=args.algo,
        reward=args.reward,
        episodic=False,
        use_vae=args.use_vae,
        vae=data.env_vae,
        pre_ny_by_date=data.pre_ny_by_date,
        tickbook=ticks_covering(data.primary_ticks, data.train_bars),
        arch=args.arch,
    )
    base = getattr(env, "unwrapped", env)
    action = _scripted_action(base)
    obs, _ = env.reset(seed=seed)
    steps = 0
    done = truncated = False
    try:
        while not (done or truncated):
            obs, _reward, done, truncated, _info = env.step(action)
            steps += 1
            if progress_every > 0 and steps % progress_every == 0:
                log.info("%s scripted replay: %d / %d bars", name, steps, len(data.train_bars))
        base = getattr(env, "unwrapped", env)
        state_fn = getattr(base, "entry_state_diagnostics", None)
        entry_state = state_fn() if callable(state_fn) else {}
        if not isinstance(entry_state, dict):
            entry_state = {}
        trade_log = list(getattr(base, "trade_log", ()))
        closed_trades = sum(1 for trade in trade_log if trade.get("type") in CLOSING_TYPES)
        opened = int(entry_state.get("entered", 0))
        closed = int(entry_state.get("closed", closed_trades))
        breaches = len(getattr(base, "breach_events", ()))
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    cfg_payload = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg_sha = hashlib.sha256(
        json.dumps(cfg_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    schema = base.action_schema.to_dict()
    raw_coordinates = schema.get("coordinates")
    if not isinstance(raw_coordinates, list):
        raise ValueError("environment action schema is missing its coordinate list")
    coordinates = [str(name) for name in raw_coordinates]
    return {
        "variant": name,
        "status": "PASS" if opened > 0 and closed_trades > 0 else "FAIL_UNREACHABLE",
        "effective_config_sha256": cfg_sha,
        "action_schema": schema,
        "scripted_action_by_coordinate": dict(zip(coordinates, action.tolist(), strict=True)),
        "train_rows": int(len(data.train_bars)),
        "train_start": str(data.train_bars.index[0]),
        "train_end": str(data.train_bars.index[-1]),
        "steps": steps,
        "candidate_created": int(entry_state.get("candidate_created", 0)),
        "armed": int(entry_state.get("armed_bars", 0)),
        "trigger_requested": int(entry_state.get("trigger_requested", 0)),
        "trigger_refused": int(entry_state.get("trigger_refused", 0)),
        "opened": opened,
        "closed": closed,
        "completed_trade_log_records": int(closed_trades),
        "breach_count": breaches,
        "entry_state_diagnostics": entry_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/thesis/evidence/scripted_training_reachability_20261023.json",
    )
    args = parser.parse_args()
    variants = args.variants or list(DEFAULT_VARIANTS)
    if args.seed < 0 or args.progress_every < 0:
        parser.error("seed and progress interval must be non-negative")

    stages: dict[str, Any] = {}
    for name in variants:
        stages[name] = _resolved_stage(name, args.seed)
    control = OmegaConf.to_container(stages[variants[0]].cfg, resolve=True, enum_to_str=True)
    for name in variants[1:]:
        treatment = OmegaConf.to_container(stages[name].cfg, resolve=True, enum_to_str=True)
        differences = _config_diff(control, treatment)
        if differences != ["env.entry_state_observation"]:
            raise RuntimeError(
                f"selected pair is not matched except for the frozen observation flag: {differences}"
            )

    data = _prepare_training_data(stages[variants[0]])
    if (
        data.train_bars.index[0].date().isoformat() != "2024-12-30"
        or data.train_bars.index[-1].date().isoformat() != "2025-12-31"
    ):
        raise RuntimeError(
            "scripted reachability requires the configured full training interval "
            "2024-12-30 through 2025-12-31"
        )
    report: dict[str, Any] = {
        "schema_version": "scripted_training_reachability_v1",
        "status": "RUNNING",
        "seed": args.seed,
        "variants": [],
        "training_split_only": True,
        "test_split_evaluated": False,
        "mechanical_gate": "at least one scripted opened and closed trade per variant",
    }
    for name in variants:
        result = _run_variant(
            name,
            stages[name],
            data,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        report["variants"].append(result)
        passed = all(item["status"] == "PASS" for item in report["variants"])
        report["status"] = (
            "PASS" if passed and len(report["variants"]) == len(variants) else "RUNNING"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp_path = args.out.with_suffix(args.out.suffix + ".tmp")
        temp_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(args.out)
        log.info(
            "%s reachability: status=%s candidates=%d opened=%d closed=%d",
            name,
            result["status"],
            result["candidate_created"],
            result["opened"],
            result["completed_trade_log_records"],
        )
    if report["status"] != "PASS":
        report["status"] = "NO_GO"
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
