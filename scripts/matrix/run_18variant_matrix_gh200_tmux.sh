#!/bin/bash
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-gpu-matrix-18}"
MATRIX="$REPO/scripts/matrix/run_18variant_matrix.sh"
SELECTION="${1:-all}"

cd "$REPO"
if [[ -z "${OUT_DIR:-}" ]]; then
  RESUME_DIR="$(
    find "$REPO/outputs" -mindepth 1 -maxdepth 1 -type d -name 'gpu_matrix_*' -print0 2>/dev/null |
      while IFS= read -r -d '' candidate; do
        if find "$candidate" -type f -name 'ppo_latest.zip' -print -quit 2>/dev/null | grep -q .; then
          printf '%s\0' "$candidate"
        fi
      done |
      xargs -0 -r -n1 printf '%s\n' |
      while read -r candidate; do
        find "$candidate" -type f -name 'ppo_latest.zip' -printf '%T@ %p\n' 2>/dev/null
      done | sort -n | tail -1 | cut -d' ' -f2- |
      sed 's#/ladder_[^/]*/.*##'
  )"
  if [[ -n "$RESUME_DIR" ]]; then
    OUT_DIR="$RESUME_DIR"
    RUN_STAMP="${RUN_STAMP:-${OUT_DIR##*_}}"
    echo "RESUME: using existing matrix output $OUT_DIR"
  else
    RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
    OUT_DIR="$REPO/outputs/gpu_matrix_${RUN_STAMP}"
  fi
else
  RUN_STAMP="${RUN_STAMP:-$(basename "$OUT_DIR" | sed 's/^gpu_matrix_//')}"
fi
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
LOG="${LOG:-$LOG_DIR/orchestrator.log}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ ! -x "$MATRIX" ]]; then
  echo "ERROR: missing $MATRIX" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "REUSE: tmux session $SESSION already exists — not starting another allocation."
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

echo "ALLOC: tmux $SESSION run=$RUN_STAMP + mode=${MODE:-auto} gpu=${GPU_KIND:-auto}"
echo "Log: $LOG"

tmux new-session -d -s "$SESSION" -n orch -c "$REPO" \
  "env TRITON_REPO='$REPO' RUN_STAMP='$RUN_STAMP' OUT_DIR='$OUT_DIR' LOG_DIR='$LOG_DIR' SEED='${SEED:-42}' STEPS='${STEPS:-20000000}' bash '$MATRIX' '$SELECTION' 2>&1 | tee -a '$LOG'; status=\${PIPESTATUS[0]}; echo DONE exit=\$status; exec bash"

follow_variant() {
  local variant="$1"
  local log="${LOG_DIR}/${variant}.log"
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
