#!/usr/bin/env python3
"""Patch QUANT_RL_MAX_N_ENVS clamp into scale_training_cfg."""

from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "quant_rl/utils/device.py")
text = path.read_text()
if "QUANT_RL_MAX_N_ENVS" in text:
    print(f"{path}: already patched")
    raise SystemExit(0)

candidates = [
    """    n_envs = suggest_n_envs(
        requested=requested,
        total_ram_bytes=total_ram,
        available_ram_bytes=avail_ram,
        cpu_count=cpu_count,
        vram_bytes=vram,
    )
    cfg.env.n_envs = n_envs
""",
    """    n_envs = suggest_n_envs(
        requested=requested,
        total_ram_bytes=total_ram,
        available_ram_bytes=avail_ram,
        cpu_count=os.cpu_count() or 1,
        vram_bytes=vram,
    )
    cfg.env.n_envs = n_envs
""",
]
insert_tail = """    max_envs_env = os.environ.get("QUANT_RL_MAX_N_ENVS")
    if max_envs_env is not None:
        n_envs = min(n_envs, max(1, int(max_envs_env)))
        n_envs = _floor_power_of_two(max(1, n_envs))
    cfg.env.n_envs = n_envs
"""
for needle in candidates:
    if needle in text:
        head = needle.rsplit("cfg.env.n_envs = n_envs", 1)[0]
        path.write_text(text.replace(needle, head + insert_tail, 1))
        print(f"{path}: patched")
        raise SystemExit(0)
raise SystemExit(f"{path}: needle not found")
