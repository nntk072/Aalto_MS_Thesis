#!/bin/bash
set -euo pipefail
chmod +x /scratch/work/nguyenl37/Aalto_MS_Thesis/scripts/launch_po3_seed49.sh
mkdir -p /scratch/work/nguyenl37/Aalto_MS_Thesis/outputs
squeue -u nguyenl37 || true
tmux has-session -t quant-rl-train 2>/dev/null || tmux new-session -d -s quant-rl-train
tmux kill-window -t quant-rl-train:po3-seed49 2>/dev/null || true
CMD="cd /scratch/work/nguyenl37/Aalto_MS_Thesis && srun --partition=gpu-grace-h200-141g --gpus=gh200:1 --time=6:00:00 --mem=256G --ntasks=1 --cpus-per-task=8 bash /scratch/work/nguyenl37/Aalto_MS_Thesis/scripts/launch_po3_seed49.sh 2>&1 | tee /scratch/work/nguyenl37/Aalto_MS_Thesis/outputs/po3_seed49_launch.log; echo EXIT:\$?; exec bash"
tmux new-window -t quant-rl-train -n po3-seed49 -d "$CMD"
echo LAUNCHED
sleep 25
squeue -u nguyenl37 -o "%.18i %.12P %.20j %.8u %.2t %.10M %.6D %R %.10m"
echo ---LOG---
tail -n 50 /scratch/work/nguyenl37/Aalto_MS_Thesis/outputs/po3_seed49_launch.log || true
echo ---PANE---
tmux capture-pane -t quant-rl-train:po3-seed49 -p -S -50 || true
