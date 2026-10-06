#!/bin/bash
# Login-node launcher: ONE tmux session, TWO windows, each with its OWN
# gh200:1 allocation pinned to a separate node (gpuarm1 / gpuarm2).
#
# Why one GPU per window instead of --gpus=gh200:2 in a single srun:
#   scripts/triton/README.md and .agents/rules/triton-slurm.md specify ONE
#   allocation per model: gh200:1, --time=6:00:00, --mem=512G, --cpus-per-task=48.
#   A 2-GPU/12h request asks the group quota (AssocGrpGRESRunMinutes) for double
#   the GPU-minutes in one shot and stayed PENDING while both nodes sat idle.
#   Two 6h single-GPU allocations are the documented shape and start right away.
#
#   tmux attach -t rr-two-gpu              # Ctrl-b w switches window
#   tmux capture-pane -t rr-two-gpu:idea1 -p | tail -40
#
# Agent rule: srun needs explicit user permission in chat before running this.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-rr-two-gpu}"
PARTITION="${PARTITION:-gpu-grace-h200-141g}"
TIME="${TIME:-6:00:00}"      # documented default for one model
MEM="${MEM:-512G}"           # one n_envs=32 train needs ~512G
CPUS="${CPUS:-48}"           # node has 144; 48 for a single trainer
LOG_DIR="${LOG_DIR:-$REPO/outputs}"

cd "$REPO"
mkdir -p "$LOG_DIR"

# One window per trainer: separate pane, separate log, independent visibility.
tmux has-session -t "$SESSION" 2>/dev/null \
  || tmux new-session -d -s "$SESSION" -n bootstrap -c "$REPO"
for w in idea1 idea2; do
  tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx "$w" \
    || tmux new-window -d -t "$SESSION" -n "$w" -c "$REPO"
done

# Different node per window so the two allocations do not stack on one node.
alloc () {
  local window="$1" script="$2" log="$3"
  local node
  case "$window" in
    idea1) node="${NODE1:-gpuarm1}" ;;
    idea2) node="${NODE2:-gpuarm2}" ;;
  esac
  echo "ALLOC $window -> $node  $PARTITION gh200:1 time=$TIME mem=$MEM cpus=$CPUS"
  # No -c flag on send-keys: the working directory comes from the window's own
  # -c "$REPO" at creation time above.
  tmux send-keys -t "$SESSION:$window" \
    "cd $REPO && source scripts/triton/activate_venv.sh && srun --partition=$PARTITION \
     --nodelist=$node --gpus=gh200:1 --time=$TIME --mem=$MEM --ntasks=1 \
     --cpus-per-task=$CPUS --chdir=$REPO bash $script 2>&1 | tee $log; \
     echo DONE rc=\$?; exec bash" C-m

  local job="" state=""
  for _ in $(seq 1 20); do
    sleep 3
    job=$(tmux capture-pane -t "$SESSION:$window" -p -S -40 \
          | grep -oE 'job [0-9]+' | tail -1 | awk '{print $2}')
    [[ -n "$job" ]] || continue
    state=$(squeue -j "$job" -h -o '%T' 2>/dev/null || true)
    echo "  $window job=$job state=${state:-gone}"
    [[ "$state" == "RUNNING" ]] && break
  done
  if [[ "$state" != "RUNNING" ]]; then
    echo "  $window did NOT start. Scheduler said:"
    scontrol show job "$job" 2>/dev/null | tr ' ' '\n' \
      | awk -F= '/^(JobState|Reason)=/' || true
    return 1
  fi
}

alloc idea1 scripts/train_rr_idea1.sh "$LOG_DIR/rr_idea1_tcn.log"
alloc idea2 scripts/train_rr_idea2.sh "$LOG_DIR/rr_idea2_tcn.log"

echo
echo "tmux '$SESSION' up: idea1 on gpuarm1, idea2 on gpuarm2"
echo "  attach : tmux attach -t $SESSION        (Ctrl-b w switches window)"
echo "  watch1 : tmux capture-pane -t $SESSION:idea1 -p | tail -40"
echo "  watch2 : tmux capture-pane -t $SESSION:idea2 -p | tail -40"
echo "  1R = \$500 fixed | reward_mode=rr | arch=tcn"
echo
# Two logs per run, deliberately:
#   outputs/rr_idea*_tcn.log          startup only (pre-run-dir lines, kept by tee)
#   outputs/<run_dir>/train.log       full transcript, beside model + eval artifacts
echo "Full per-run transcript lands in the run folder once it is created:"
echo "  ls -t outputs/*_rl_train_*/train.log | head -2"
echo "  tail -f \$(ls -t outputs/*_rl_train_*/train.log | head -1)"
