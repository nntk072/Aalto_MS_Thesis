#!/usr/bin/env python3
from pathlib import Path

path = Path("/scratch/work/nguyenl37/Aalto_MS_Thesis/quant_rl/utils/device.py")
text = path.read_text()
bad = """    n_envs = suggest_n_envs(
        requested=requested,
        total_ram_bytes=total_ram,
        available_ram_bytes=avail_ram,
        cpu_count=cpu_count,
        vram_bytes=vram,
    )
        max_envs_env = os.environ.get("QUANT_RL_MAX_N_ENVS")
    if max_envs_env is not None:
        n_envs = min(n_envs, max(1, int(max_envs_env)))
        n_envs = _floor_power_of_two(max(1, n_envs))
    cfg.env.n_envs = n_envs
"""
good = """    n_envs = suggest_n_envs(
        requested=requested,
        total_ram_bytes=total_ram,
        available_ram_bytes=avail_ram,
        cpu_count=cpu_count,
        vram_bytes=vram,
    )
    max_envs_env = os.environ.get("QUANT_RL_MAX_N_ENVS")
    if max_envs_env is not None:
        n_envs = min(n_envs, max(1, int(max_envs_env)))
        n_envs = _floor_power_of_two(max(1, n_envs))
    cfg.env.n_envs = n_envs
"""
if bad not in text:
    raise SystemExit("bad block not found")
path.write_text(text.replace(bad, good, 1))
compile(path.read_text(), str(path), "exec")
print("fixed and syntax-ok")
