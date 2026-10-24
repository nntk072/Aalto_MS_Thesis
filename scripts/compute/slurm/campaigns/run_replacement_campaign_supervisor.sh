#!/usr/bin/env bash
# Sequentially reacquire the same bounded allocation until this campaign is terminal.
# Run only in a named, persistent login-node tmux session after user approval.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-replacement18_seed50_train2025_test2026_current_mtf_20261019}"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
LOG="$OUT_DIR/logs/allocation_supervisor.log"
NODES=(gpuarm1 gpuarm2)
MIN_CPUS=48
MIN_MEM_MB=$((512 * 1024))

if [[ -z "${TMUX:-}" ]]; then
  echo "Run this supervisor in a named persistent tmux session on the login node." >&2
  exit 2
fi

cd "$REPO"
mkdir -p "$OUT_DIR/logs"
exec >>"$LOG" 2>&1
exec 9>"$OUT_DIR/.allocation_supervisor.lock"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another supervisor owns this campaign; exiting"
  exit 2
fi
echo "$(date --iso-8601=seconds) supervisor start host=$(hostname -s) campaign=$CAMPAIGN_ID"

campaign_state() {
  local manifest="$OUT_DIR/campaign.json"
  if [[ ! -s "$manifest" ]]; then
    echo "missing"
    return
  fi
  python3 - "$manifest" <<'PY'
import json, sys
from collections import Counter
campaign = json.load(open(sys.argv[1], encoding="utf-8"))
counts = Counter(job["status"] for job in campaign["jobs"])
active = sum(counts.get(name, 0) for name in ("pending", "running", "interrupted"))
print(active, counts.get("failed", 0), counts.get("succeeded", 0))
PY
}

node_is_ready() {
  local node="$1"
  local record
  record="$(scontrol show node -o "$node")"
  echo "$(date --iso-8601=seconds) inventory $record"
  python3 "$REPO/compute/slurm/capacity.py" \
    --partition gpu-grace-h200-141g \
    --node "$node" \
    --nodes 1 \
    --gpu-type gh200 \
    --gpus-per-node 1 \
    --cpus-per-node "$MIN_CPUS" \
    --memory-mb "$MIN_MEM_MB" >/dev/null
}

campaign_finished() {
  local state active failed succeeded
  state="$(campaign_state)"
  [[ "$state" != "missing" ]] || return 1
  read -r active failed succeeded <<<"$state"
  echo "$(date --iso-8601=seconds) campaign active=$active failed=$failed succeeded=$succeeded"
  [[ "$active" == "0" ]]
}

while true; do
  if campaign_finished; then
    state="$(campaign_state)"
    read -r _active failed succeeded <<<"$state"
    echo "$(date --iso-8601=seconds) campaign terminal failed=$failed succeeded=$succeeded"
    exit 0
  fi

  # Keep one acquisition at a time. While the fixed pair of nodes is occupied,
  # poll scheduler state and wait; never queue overlapping GH200 successors.
  nodes_ready=1
  for node in "${NODES[@]}"; do
    if ! node_is_ready "$node"; then
      nodes_ready=0
    fi
  done
  own_gpu_jobs="$(squeue -h -u "$(id -un)" -p gpu-grace-h200-141g -o '%i %t %N')"
  if [[ -n "$own_gpu_jobs" ]]; then
    echo "$(date --iso-8601=seconds) existing user GH200 job(s); waiting without a new request"
    printf '%s\n' "$own_gpu_jobs"
    sleep 60
    continue
  fi
  if [[ "$nodes_ready" -ne 1 ]]; then
    sleep 60
    continue
  fi

  echo "$(date --iso-8601=seconds) requesting 2 GH200 total (1 per node), 48 CPUs and 512G per node, 6h"
  set +e
  srun \
    --immediate=60 \
    --partition=gpu-grace-h200-141g \
    --nodes=2 --nodelist=gpuarm1,gpuarm2 \
    --ntasks=2 --ntasks-per-node=1 \
    --cpus-per-task="$MIN_CPUS" \
    --gpus-per-node=gh200:1 \
    --mem=512G \
    --time=06:00:00 \
    --job-name=thesis18new \
    --export=ALL,TRITON_REPO="$REPO",CAMPAIGN_SPEC="$SPEC",CAMPAIGN_ID="$CAMPAIGN_ID" \
    --chdir="$REPO" \
    /bin/bash -c '
      set -euo pipefail
      allocation_script="$1"
      printf "allocation shell host=%s bash=%s script=%s\\n" "$(hostname -s)" "$BASH_VERSION" "$allocation_script"
      /bin/bash -n "$allocation_script"
      exec /bin/bash "$allocation_script"
    ' thesis-allocation "$REPO/scripts/compute/slurm/campaigns/run_replacement_campaign_allocation.sh"
  request_status=$?
  set -e
  echo "$(date --iso-8601=seconds) allocation returned exit=$request_status"
  if [[ "$request_status" -ne 0 ]]; then
    echo "$(date --iso-8601=seconds) allocation failed; refusing automatic retry until diagnosis"
    exit "$request_status"
  fi

  state="$(campaign_state)"
  if [[ "$state" == "missing" ]]; then
    echo "$(date --iso-8601=seconds) campaign manifest absent after request; stop for diagnosis"
    exit 2
  fi
  read -r active failed succeeded <<<"$state"
  echo "$(date --iso-8601=seconds) post-allocation active=$active failed=$failed succeeded=$succeeded"
  if [[ "$active" == "0" ]]; then
    echo "$(date --iso-8601=seconds) no recoverable campaign work remains"
    exit 0
  fi
  # The next loop performs fresh per-node CPU/RAM/GPU and queue checks before
  # making one sequential successor request.
done
