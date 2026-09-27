#!/usr/bin/env bash
# Two n_envs=64 trains on ONE H200. One allocation, no size search.
#
# Shape (the one that actually starts): h200:1, 900G, 64 CPUs, 6h.
# 12h and >=1024G stay PENDING. srun --test-only is not used.
# Second window ssh's the node. srun --overlap fails with "Insane message length".
#
# Agents: run only after the user agrees to a new allocation.
#   bash scripts/triton/two_trains_one_h200.sh
#   REPLACE=1 bash scripts/triton/two_trains_one_h200.sh   # scancel other GPU jobs
#       only after this one is RUNNING
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-quant-rl-train}"
PARTITION="${PARTITION:-gpu-h200-141g-short}"
GPUS="${GPUS:-h200:1}"
MEM="${MEM:-900G}"
CPUS="${CPUS:-64}"
TIME="${TIME:-6:00:00}"
SEED="${SEED:-50}"
REPLACE="${REPLACE:-0}"

cd "$REPO"

train_line() {
  local strategy="$1" log="$2"
  printf '%s' "source $REPO/scripts/triton/activate_venv.sh && cd $REPO && CUDA_VISIBLE_DEVICES=0 QUANT_RL_MAX_N_ENVS=64 python -u -m quant_rl.train.train_rl --config config/features_full_po3_mtf.yaml --strategy $strategy --seed $SEED --arch tcn --out outputs features.include_session_ohlc=true features.liquidity.enabled=true features.po3_state_mtf.enabled=true features.ifvg_mtf.enabled=true env.n_envs=64 env.peak_trailing_dd_limit=0 ftmo.trailing_dd_limit=0.07 2>&1 | tee outputs/$log"
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

echo "ALLOC once: partition=$PARTITION gpus=$GPUS mem=$MEM cpus=$CPUS time=$TIME"
tmux send-keys -t "$SESSION:idea1" \
  "cd $REPO && srun --partition=$PARTITION --gpus=$GPUS --time=$TIME --mem=$MEM --ntasks=1 --cpus-per-task=$CPUS --pty bash" C-m

job=""
state=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  sleep 2
  job=$(tmux capture-pane -t "$SESSION:idea1" -p -S -30 | grep -oE 'job [0-9]+' | awk 'END{print $2}')
  if [[ -n "$job" ]]; then
    state=$(squeue -j "$job" -h -o '%T' || true)
    echo "job $job state=${state:-gone}"
    [[ "$state" == "RUNNING" ]] && break
  fi
done

if [[ "$state" != "RUNNING" ]]; then
  echo "Did not start. Cancel this request and stop. Do not raise mem, CPUs, GPUs, or time."
  [[ -n "$job" ]] && scancel "$job" || true
  scontrol show job "$job" 2>/dev/null | tr ' ' '\n' | awk -F= '/^(JobState|Reason|StartTime)=/' || true
  exit 1
fi

node=$(squeue -j "$job" -h -o '%N')
echo "Running on $node ($job)"

if [[ "$REPLACE" == "1" ]]; then
  squeue -u "$USER" -h -t R -o '%i' | while read -r other; do
    [[ "$other" == "$job" ]] && continue
    echo "scancel $other"
    scancel "$other"
  done
fi

# Wait until the compute-node prompt exists before typing the train command.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if tmux capture-pane -t "$SESSION:idea1" -p -S -15 | grep -q "@$node"; then
    break
  fi
  sleep 2
done

tmux send-keys -t "$SESSION:idea1" "$(train_line po3_ifvg tcn_idea1_64.log)" C-m

tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx idea2 || tmux new-window -d -t "$SESSION" -n idea2
tmux send-keys -t "$SESSION:idea2" \
  "ssh -t $node \"$(train_line distribution tcn_idea2_64.log)\"" C-m

echo "idea1: tmux window $SESSION:idea1  (srun on $node)  log outputs/tcn_idea1_64.log"
echo "idea2: tmux window $SESSION:idea2  (ssh $node, same job $job)  log outputs/tcn_idea2_64.log"
echo "Attach: tmux attach -t $SESSION"
