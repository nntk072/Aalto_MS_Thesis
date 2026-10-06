#!/bin/bash
set -euo pipefail
ssh -o BatchMode=yes nguyenl37@triton.aalto.fi bash -s <<'REMOTE'
set -euo pipefail
if squeue -u nguyenl37 -h | grep -q .; then
  echo "JOB ALREADY RUNNING — not launching"
  squeue -u nguyenl37
  exit 0
fi
tmux has-session -t quant-rl-train 2>/dev/null || tmux new-session -d -s quant-rl-train
tmux kill-window -t quant-rl-train:po3-seed49 2>/dev/null || true
tmux new-window -t quant-rl-train -n po3-seed49
tmux send-keys -t quant-rl-train:po3-seed49 'cd /scratch/work/nguyenl37/Aalto_MS_Thesis && srun --partition=gpu-grace-h200-141g --gpus=gh200:1 --time=6:00:00 --mem=128G --ntasks=1 --cpus-per-task=8 bash scripts/launch_po3_seed49.sh 2>&1 | tee outputs/po3_seed49_launch.log' Enter
sleep 8
echo ==== SQUEUE ====
squeue -u nguyenl37
echo ==== TMUX ====
tmux capture-pane -pt quant-rl-train:po3-seed49 -S -40
REMOTE
