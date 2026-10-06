#!/bin/bash
set -euo pipefail
REPO_WSL=/home/nguyenl37/Aalto_MS_Thesis
REPO_TRITON=/scratch/work/nguyenl37/Aalto_MS_Thesis
HOST=nguyenl37@triton.aalto.fi

scp \
  "$REPO_WSL/scripts/launch_po3_seed49.sh" \
  "$REPO_WSL/.agents/rules/triton-slurm.md" \
  "$REPO_WSL/scripts/_patch_max_n_envs.py" \
  "$HOST:/tmp/"

ssh -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -euo pipefail
REPO=/scratch/work/nguyenl37/Aalto_MS_Thesis
cp /tmp/launch_po3_seed49.sh "$REPO/scripts/launch_po3_seed49.sh"
cp /tmp/triton-slurm.md "$REPO/.agents/rules/triton-slurm.md"
chmod +x "$REPO/scripts/launch_po3_seed49.sh"
python3 /tmp/_patch_max_n_envs.py "$REPO/quant_rl/utils/device.py"
echo ==== CHECKLIST ====
cd "$REPO"
for f in scripts/launch_po3_seed49.sh .agents/rules/triton-slurm.md quant_rl/utils/device.py AGENTS.md quant_rl/train/train_rl.py quant_rl/eval/rollout.py quant_rl/envs/trading_env.py quant_rl/config/default.yaml; do
  printf '%-45s ' "$f"
  md5sum "$f" | awk '{print $1}'
done
echo --- launch ---
cat scripts/launch_po3_seed49.sh
echo --- mem/PO3 ---
grep -nE 'mem=|PO3/MTF' .agents/rules/triton-slurm.md
echo --- maxenvs ---
grep -n QUANT_RL_MAX_N_ENVS quant_rl/utils/device.py
echo --- strategy_actions on Triton ---
grep -c strategy_actions quant_rl/train/train_rl.py quant_rl/envs/trading_env.py quant_rl/eval/rollout.py
echo --- compact_actions on Triton ---
grep -c compact_actions quant_rl/train/train_rl.py quant_rl/envs/trading_env.py quant_rl/config/default.yaml
REMOTE
