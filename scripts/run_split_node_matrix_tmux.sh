#!/bin/bash
# Two 1-GPU jobs at once: overlay on one node, baseline on the other.
# 32 CPUs and 900G each. Does not scancel an existing session.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-matrix-6run}"
PARTITION="${PARTITION:-gpu-grace-h200-141g}"
TIME="${TIME:-6:00:00}"
MEM="${MEM:-900G}"
CPUS="${CPUS:-32}"
WAVE="$REPO/scripts/run_strategy_wave.sh"

cd "$REPO"
mkdir -p outputs

if [[ ! -x "$WAVE" ]]; then
  echo "ERROR: missing $WAVE" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "REUSE: tmux session $SESSION already exists — not starting another srun."
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

start_job() {
  local mode=$1
  local log="$REPO/outputs/${mode}_job.log"
  local name="bl-job"
  if [[ "$mode" == "overlay" ]]; then
    name="ov-job"
  fi
  tmux new-window -d -t "$SESSION" -n "$name" -c "$REPO" \
    "srun --partition=$PARTITION --gpus=gh200:1 --time=$TIME --mem=$MEM --ntasks=1 --cpus-per-task=$CPUS --chdir=$REPO env MODE=$mode bash $WAVE 2>&1 | tee $log; echo DONE exit=\$?; exec bash"
}

follow_run() {
  local mode=$1 arch=$2
  local mode_short arch_short
  case "$mode" in
    overlay) mode_short=ov ;;
    baseline) mode_short=bl ;;
  esac
  case "$arch" in
    transformer) arch_short=tr ;;
    *) arch_short=$arch ;;
  esac
  local log="outputs/${mode}_${arch}_20m.log"
  tmux new-window -d -t "$SESSION" -n "${mode_short}-${arch_short}" -c "$REPO" \
    "if [[ ! -f '$log' ]]; then echo 'waiting for $mode $arch'; fi; while [[ ! -f '$log' ]]; do sleep 20; done; exec tail -n +1 -F '$log'"
}

echo "ALLOC: two jobs gpus=gh200:1 time=$TIME mem=$MEM cpus=$CPUS"
tmux new-session -d -s "$SESSION" -n ov-job -c "$REPO" \
  "srun --partition=$PARTITION --gpus=gh200:1 --time=$TIME --mem=$MEM --ntasks=1 --cpus-per-task=$CPUS --chdir=$REPO env MODE=overlay bash $WAVE 2>&1 | tee $REPO/outputs/overlay_job.log; echo DONE exit=\$?; exec bash"

# Second job after the first is accepted, so the 720 GH200-minute cap sees one reservation leave first.
for _ in $(seq 1 60); do
  if squeue -u "$USER" -h -t R,PD | grep -q .; then
    break
  fi
  sleep 1
done
start_job baseline

for arch in tcn gru transformer; do
  follow_run overlay "$arch"
  follow_run baseline "$arch"
done
tmux select-window -t "$SESSION:ov-job"

echo "Started tmux session $SESSION."
echo "Attach: tmux attach -t $SESSION"
echo "Windows: ov-job bl-job ov-tcn bl-tcn ov-gru bl-gru ov-tr bl-tr"
echo "Switch: Ctrl-b w. Detach: Ctrl-b d"
