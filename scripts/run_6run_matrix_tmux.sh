#!/bin/bash
# Login-node launcher: detached tmux + one 6h 2×GH200 allocation.
# Window orch runs the matrix. Six windows follow one run each.
# Switch: Ctrl-b n / Ctrl-b p / Ctrl-b w. Detach: Ctrl-b d.
# Does not scancel or kill an existing session.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-matrix-6run}"
PARTITION="${PARTITION:-gpu-grace-h200-141g}"
GPUS="${GPUS:-gh200:2}"
TIME="${TIME:-6:00:00}"
MEM="${MEM:-900G}"
CPUS="${CPUS:-128}"
MATRIX="$REPO/scripts/run_dual_gpu_6run_matrix.sh"
LOG="${LOG:-$REPO/outputs/matrix_6run_tmux.log}"

cd "$REPO"
mkdir -p outputs

if [[ ! -x "$MATRIX" ]]; then
  echo "ERROR: missing $MATRIX" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "REUSE: tmux session $SESSION already exists — not starting another srun."
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

echo "ALLOC: tmux $SESSION + srun partition=$PARTITION gpus=$GPUS time=$TIME mem=$MEM cpus=$CPUS"
echo "Log: $LOG"

tmux new-session -d -s "$SESSION" -n orch -c "$REPO" \
  "export WAVE_PARALLEL='${WAVE_PARALLEL:-3}'; srun --partition=$PARTITION --gpus=$GPUS --time=$TIME --mem=$MEM --ntasks=1 --cpus-per-task=$CPUS --chdir=$REPO bash $MATRIX 2>&1 | tee $LOG; echo DONE exit=\$?; exec bash"

follow_run() {
  local mode=$1 arch=$2
  local mode_short arch_short
  case "$mode" in
    overlay) mode_short=ov ;;
    baseline) mode_short=bl ;;
    *) mode_short=$mode ;;
  esac
  case "$arch" in
    transformer) arch_short=tr ;;
    *) arch_short=$arch ;;
  esac
  local log="outputs/${mode}_${arch}_20m.log"
  tmux new-window -t "$SESSION" -n "${mode_short}-${arch_short}" -c "$REPO" \
    "if [[ ! -f '$log' ]]; then echo 'waiting for $mode $arch — this run starts after the overlay wave'; fi; while [[ ! -f '$log' ]]; do sleep 20; done; exec tail -n +1 -F '$log'"
}

for arch in tcn gru transformer; do
  follow_run overlay "$arch"
  follow_run baseline "$arch"
done
tmux select-window -t "$SESSION:orch"

echo "Started tmux session $SESSION (detached)."
echo "Attach: tmux attach -t $SESSION"
echo "Windows: orch + ov-tcn ov-gru ov-tr bl-tcn bl-gru bl-tr"
echo "Switch: Ctrl-b n, Ctrl-b p, or Ctrl-b w. Detach: Ctrl-b d"
