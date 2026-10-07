#!/bin/bash
# Login-node launcher: detached tmux + ONE GH200 srun for the full test bar.
# Does not scancel or tmux kill-session. Tear down leftover jobs by hand first.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-full-tests}"
PARTITION="${PARTITION:-gpu-grace-h200-141g}"
GPUS="${GPUS:-gh200:1}"
TIME="${TIME:-06:00:00}"
MEM="${MEM:-128G}"
CPUS="${CPUS:-8}"
RUN="$REPO/scripts/run_full_tests.sh"
LOG="$REPO/outputs/full_tests_tmux.log"

cd "$REPO"
mkdir -p outputs

if [[ ! -x "$RUN" ]]; then
  echo "ERROR: missing $RUN" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "REUSE: tmux session $SESSION already exists — not starting another srun."
  echo "Attach: tmux attach -t $SESSION"
  echo "You may now disconnect."
  exit 0
fi

echo "ALLOC: tmux $SESSION + srun partition=$PARTITION gpus=$GPUS time=$TIME mem=$MEM cpus=$CPUS"
echo "Log: $LOG"

tmux new-session -d -s "$SESSION" -c "$REPO" \
  "srun --partition=$PARTITION --gpus=$GPUS --time=$TIME --mem=$MEM --ntasks=1 --cpus-per-task=$CPUS --chdir=$REPO bash $RUN 2>&1 | tee $LOG; echo DONE exit=\$?; exec bash"

echo "Started tmux session $SESSION (detached)."
echo "Attach: tmux attach -t $SESSION"
echo "You may now disconnect. Laptop sleep will not kill this session."
