#!/bin/bash
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-gpu-matrix-18}"
PARTITION="${PARTITION:-gpu-grace-h200-141g}"
GPUS="${GPUS:-gh200:1}"
TIME="${TIME:-6:00:00}"
MEM="${MEM:-900G}"
CPUS="${CPUS:-64}"
MATRIX="$REPO/scripts/matrix/run_18variant_matrix.sh"
LOG="${LOG:-$REPO/outputs/gpu_matrix_18/tmux_orch.log}"

cd "$REPO"
OUT_DIR="${OUT_DIR:-$REPO/outputs/gpu_matrix_18}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$MATRIX" ]]; then
  echo "ERROR: missing $MATRIX" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "REUSE: tmux session $SESSION already exists — not starting another allocation."
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

echo "ALLOC: tmux $SESSION + srun partition=$PARTITION gpus=$GPUS time=$TIME mem=$MEM cpus=$CPUS"
echo "Log: $LOG"

tmux new-session -d -s "$SESSION" -n orch -c "$REPO" \
  "bash $MATRIX 2>&1 | tee $LOG; echo DONE exit=\$?; exec bash"

follow_variant() {
  local variant="$1"
  local log="$REPO/outputs/gpu_matrix_18/${variant}.log"
  tmux new-window -t "$SESSION" -n "$variant" -c "$REPO" \
    "if [[ ! -f '$log' ]]; then echo 'waiting for $variant log to appear'; while [[ ! -f '$log' ]]; do sleep 10; done; fi; exec tail -n +1 -F '$log'"
}

while IFS= read -r variant; do
  [[ -z "$variant" ]] && continue
  follow_variant "$variant"
done < "$REPO/scripts/variant_list.txt"

tmux select-window -t "$SESSION:orch"

echo "Started tmux session $SESSION (detached)."
echo "Attach: tmux attach -t $SESSION"
echo "Windows: orch + 18 variant follow windows"
