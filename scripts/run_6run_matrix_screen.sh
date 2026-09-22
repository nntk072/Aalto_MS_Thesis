#!/bin/bash
# Login-node launcher: detached screen + one 6h 2×GH200 allocation.
# Runs the 6-run encoder matrix (3 overlay + 3 baseline, 20M each).
# Does not scancel or kill an existing session.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${SCREEN_SESSION:-matrix-6run}"
PARTITION="${PARTITION:-gpu-grace-h200-141g}"
GPUS="${GPUS:-gh200:2}"
TIME="${TIME:-6:00:00}"
MEM="${MEM:-900G}"
CPUS="${CPUS:-128}"
MATRIX="$REPO/scripts/run_dual_gpu_6run_matrix.sh"
LOG="${LOG:-$REPO/outputs/matrix_6run_screen.log}"

cd "$REPO"
mkdir -p outputs

if [[ ! -x "$MATRIX" ]]; then
  echo "ERROR: missing $MATRIX" >&2
  exit 1
fi

if screen -ls | grep -q "[[:space:]]*[0-9]*\.${SESSION}[[:space:]]"; then
  echo "REUSE: screen session $SESSION already exists — not starting another srun."
  echo "Attach: screen -r $SESSION"
  exit 0
fi

echo "ALLOC: screen $SESSION + srun partition=$PARTITION gpus=$GPUS time=$TIME mem=$MEM cpus=$CPUS"
echo "Log: $LOG"

screen -dmS "$SESSION" bash -lc "
  cd '$REPO' || exit 1
  export WAVE_PARALLEL='${WAVE_PARALLEL:-3}'
  srun --partition='$PARTITION' --gpus='$GPUS' --time='$TIME' \
    --mem='$MEM' --ntasks=1 --cpus-per-task='$CPUS' --chdir='$REPO' \
    bash '$MATRIX' 2>&1 | tee '$LOG'
  echo DONE exit=\$?
  exec bash
"

echo "Started screen session $SESSION (detached)."
echo "Attach: screen -r $SESSION"
echo "Detach: Ctrl-a d"
