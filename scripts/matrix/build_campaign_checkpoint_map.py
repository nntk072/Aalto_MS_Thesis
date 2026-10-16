#!/usr/bin/env python3
"""Create an immutable checkpoint-lineage map for a successor campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any, cast

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "scripts" / "matrix" / "campaign_controller.py"
CONTROLLER_SPEC = importlib.util.spec_from_file_location(
    "campaign_controller_for_checkpoint_map", CONTROLLER_PATH
)
if CONTROLLER_SPEC is None or CONTROLLER_SPEC.loader is None:
    raise ImportError(f"cannot load campaign controller from {CONTROLLER_PATH}")
CONTROLLER_MODULE = importlib.util.module_from_spec(CONTROLLER_SPEC)
sys.modules[CONTROLLER_SPEC.name] = CONTROLLER_MODULE
CONTROLLER_SPEC.loader.exec_module(CONTROLLER_MODULE)
build_initial_checkpoint_map = cast(Any, CONTROLLER_MODULE).build_initial_checkpoint_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-campaign",
        action="append",
        type=pathlib.Path,
        required=True,
        help="source campaign directory; repeat for multiple parent campaigns",
    )
    parser.add_argument("--variant-file", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", required=True, help="space/comma-separated seed list")
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    variants = [
        line.strip()
        for line in args.variant_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    seeds = [int(seed) for seed in args.seeds.replace(",", " ").split()]
    if not variants or not seeds or len(set(variants)) != len(variants):
        raise ValueError("variant file or seed list is empty or duplicated")
    required_keys = {f"{variant}__seed{seed}" for variant in variants for seed in seeds}
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError(
            f"checkpoint map output must be stored inside this repository "
            f"(output={output}, repository={ROOT.resolve()})"
        ) from None
    if output.exists():
        raise FileExistsError(f"refusing to replace existing checkpoint map: {output}")
    data = build_initial_checkpoint_map(args.source_campaign, required_keys, args.timesteps)
    data["target_run_keys"] = sorted(required_keys)
    temporary = output.with_name(output.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "run_keys": len(required_keys)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
