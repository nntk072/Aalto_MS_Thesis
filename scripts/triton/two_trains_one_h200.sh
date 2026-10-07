#!/usr/bin/env bash
# Two n_envs=64 trains, one GH200 and 6h each.
#
# Each train is its own allocation: gh200:1, 512G, 48 CPUs, 6h.
# A gh200:2 request only fits when one node has both GPUs free.
# 12h and >=1024G stay PENDING. srun --overlap is not used.
#
# Agents: run only after the user agrees to a new allocation.
#   bash scripts/triton/two_trains_one_h200.sh
#   REPLACE=1 bash scripts/triton/two_trains_one_h200.sh   # scancel other GPU jobs
#       only after this one is RUNNING
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-quant-rl-train}"
PARTITION="${PARTITION:-gpu-grace-h200-141g}"
GPUS="${GPUS:-gh200:1}"
MEM="${MEM:-512G}"
CPUS="${CPUS:-48}"
TIME="${TIME:-6:00:00}"
SEED="${SEED:-50}"
REPLACE="${REPLACE:-0}"

cd "$REPO"

# Permanent Idea 1 / Idea 2 launch: strategy-owned side (5-D), no 10-bar open block.
train_line() {
  local strategy="$1" log="$2" device="$3"
  printf '%s' "source $REPO/scripts/triton/activate_venv.sh && cd $REPO && CUDA_VISIBLE_DEVICES=${device} QUANT_RL_MAX_N_ENVS=64 python -u -m quant_rl.train.train_rl --config config/features/full_po3_mtf.yaml --strategy $strategy --seed $SEED --arch tcn --out outputs features.include_session_ohlc=true features.liquidity.enabled=true features.po3_state_mtf.enabled=true features.ifvg_mtf.enabled=true env.n_envs=64 env.open_manipulation_bars=0 env.peak_trailing_dd_limit=0 ftmo.trailing_dd_limit=0.07 2>&1 | tee outputs/$log"
}

if [[ "$REPLACE" != "1" ]]; then
  running=$(squeue -u "$USER" -h -t R -o '%i' | wc -l | tr -d ' ')
  if [[ "$running" != "0" ]]; then
    echo "A GPU job is already running. Reuse it, or re-run with REPLACE=1."
    squeue -u "$USER" -o '%.18i %.8T %.10M %N %b %m %C'
    exit 0
  fi
fi

tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n bootstrap
tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx idea1 || tmux new-window -d -t "$SESSION" -n idea1
tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx idea2 || tmux new-window -d -t "$SESSION" -n idea2

# One free GH200 on each Grace node. Pin so the two jobs do not stack.
wait_alloc() {
  local window="$1" node="$2"
  local job="" state=""
  echo "ALLOC $window on $node: partition=$PARTITION gpus=$GPUS mem=$MEM cpus=$CPUS time=$TIME"
  tmux send-keys -t "$SESSION:$window" \
    "cd $REPO && srun --partition=$PARTITION --nodelist=$node --gpus=$GPUS --time=$TIME --mem=$MEM --ntasks=1 --cpus-per-task=$CPUS --pty bash" C-m
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    sleep 3
    job=$(tmux capture-pane -t "$SESSION:$window" -p -S -30 | grep -oE 'job [0-9]+' | awk 'END{print $2}')
    if [[ -n "$job" ]]; then
      state=$(squeue -j "$job" -h -o '%T' || true)
      echo "job $job state=${state:-gone}"
      [[ "$state" == "RUNNING" ]] && break
    fi
  done
  if [[ "$state" != "RUNNING" ]]; then
    echo "$window did not start. Cancel this request and stop. Do not raise mem, CPUs, GPUs, or time."
    [[ -n "$job" ]] && scancel "$job" || true
    scontrol show job "$job" 2>/dev/null | tr ' ' '\n' | awk -F= '/^(JobState|Reason|StartTime)=/' || true
    return 1
  fi
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if tmux capture-pane -t "$SESSION:$window" -p -S -15 | grep -q "@$node"; then
      return 0
    fi
    sleep 2
  done
  echo "$window is RUNNING but the node prompt never appeared"
  return 1
}

wait_alloc idea1 gpuarm1
tmux send-keys -t "$SESSION:idea1" "$(train_line po3_ifvg tcn_idea1_64.log 0)" C-m
wait_alloc idea2 gpuarm2
tmux send-keys -t "$SESSION:idea2" "$(train_line distribution tcn_idea2_64.log 0)" C-m

echo "idea1: tmux window $SESSION:idea1 on gpuarm1  log outputs/tcn_idea1_64.log"
echo "idea2: tmux window $SESSION:idea2 on gpuarm2  log outputs/tcn_idea2_64.log"
echo "Attach: tmux attach -t $SESSION"
