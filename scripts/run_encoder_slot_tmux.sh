#!/bin/bash
# Login-node launcher: one detached tmux session and one 12h GH200 allocation.
# Trains the merged overlay, then the unconstrained baseline, for ARCH.
# Does not scancel or kill an existing session.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-encoder-slot}"
PARTITION="${PARTITION:-gpu-grace-h200-141g}"
GPUS="${GPUS:-gh200:1}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-512G}"
CPUS="${CPUS:-48}"
TRAIN="$REPO/scripts/train_overlay_baseline_20m.sh"
LOG="${LOG:-$REPO/outputs/encoder_slot_tmux.log}"

cd "$REPO"
mkdir -p outputs

if [[ ! -x "$TRAIN" ]]; then
  echo "ERROR: missing $TRAIN" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "REUSE: tmux session $SESSION already exists — not starting another srun."
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

echo "ALLOC: tmux $SESSION + srun partition=$PARTITION gpus=$GPUS time=$TIME mem=$MEM cpus=$CPUS arch=${ARCH:-tcn}"
echo "Log: $LOG"

tmux new-session -d -s "$SESSION" -c "$REPO" \
  "export SKIP_IDEA3=${SKIP_IDEA3:-0}; export SKIP_BASELINE=${SKIP_BASELINE:-${SKIP_IDEA3:-0}}; export ARCH=${ARCH:-tcn}; srun --partition=$PARTITION --gpus=$GPUS --time=$TIME --mem=$MEM --ntasks=1 --cpus-per-task=$CPUS --chdir=$REPO bash $TRAIN 2>&1 | tee $LOG; echo DONE exit=\$?; exec bash"

echo "Started tmux session $SESSION (detached)."
echo "Attach: tmux attach -t $SESSION"
