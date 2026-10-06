#!/bin/bash
set -euo pipefail
ssh -o BatchMode=yes nguyenl37@triton.aalto.fi bash -s <<'REMOTE'
set -euo pipefail
echo ==== SQUEUE ====
squeue -u nguyenl37
echo ==== TMUX ====
tmux capture-pane -pt quant-rl-train:po3-seed49 -S -60
echo ==== LOG TAIL ====
tail -n 40 /scratch/work/nguyenl37/Aalto_MS_Thesis/outputs/po3_seed49_launch.log 2>/dev/null || echo NO_LOG_YET
REMOTE
