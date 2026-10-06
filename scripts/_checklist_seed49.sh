#!/bin/bash
set -euo pipefail
ssh -o BatchMode=yes nguyenl37@triton.aalto.fi bash -s <<'REMOTE'
set -euo pipefail
echo ==== SQUEUE ====
squeue -u nguyenl37
echo ==== CHECKLIST ====
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
printf '%-45s %-14s %s\n' FILE ACTION MD5
print_row() { printf '%-45s %-14s %s\n' "$1" "$2" "$(md5sum "$1" | awk '{print $1}')"; }
print_row scripts/launch_po3_seed49.sh COPIED_WSL
print_row .agents/rules/triton-slurm.md COPIED_WSL
print_row quant_rl/utils/device.py SURGICAL_PATCH
print_row AGENTS.md LEFT_TRITON
print_row quant_rl/train/train_rl.py LEFT_TRITON
print_row quant_rl/eval/rollout.py LEFT_TRITON
print_row quant_rl/envs/trading_env.py LEFT_TRITON
print_row quant_rl/config/default.yaml LEFT_TRITON
echo ==== confirms ====
grep -c strategy_actions quant_rl/train/train_rl.py quant_rl/envs/trading_env.py quant_rl/eval/rollout.py
grep -c compact_actions quant_rl/train/train_rl.py quant_rl/envs/trading_env.py quant_rl/config/default.yaml
grep -n QUANT_RL_MAX_N_ENVS quant_rl/utils/device.py | head -5
grep -nE 'mem=|PO3/MTF' .agents/rules/triton-slurm.md
REMOTE
