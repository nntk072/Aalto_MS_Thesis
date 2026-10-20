#!/usr/bin/env bash
# Sequentially reacquire the approved two-node GH200 shape until the pair campaign is terminal.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261019.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261019}"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
LOG="$OUT_DIR/logs/allocation_supervisor.log"
NODES=(gpuarm1 gpuarm2)
# Any failed allocation after diagnosis is an early-exit gate: stop and inspect.
NO_PROGRESS_LIMIT=1

if [[ -z "${TMUX:-}" ]]; then
  echo "Run this supervisor in a named persistent login tmux session." >&2
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
echo "$(date --iso-8601=seconds) supervisor start campaign=$CAMPAIGN_ID host=$(hostname -s)"

campaign_state() {
  local manifest="$OUT_DIR/campaign.json"
  [[ -s "$manifest" ]] || { echo missing; return; }
  python3 - "$manifest" <<'PY'
import json
import sys
from collections import Counter

counts = Counter(job["status"] for job in json.load(open(sys.argv[1], encoding="utf-8"))["jobs"])
active = sum(counts.get(name, 0) for name in ("pending", "running", "interrupted"))
print(active, counts.get("failed", 0), counts.get("succeeded", 0))
PY
}

checkpoint_signature() {
  local root="$OUT_DIR/runs/attempts"
  [[ -d "$root" ]] || { echo empty; return; }
  find "$root" -type f \( -name 'ppo_ckpt_*_steps.zip' -o -name 'ppo_final.zip' \) \
    -printf '%P %s %T@\n' | sort | sha256sum | awk '{print $1}'
}

node_ready() {
  local node="$1" record
  record="$(scontrol show node -o "$node")"
  echo "$(date --iso-8601=seconds) inventory $record"
  printf '%s\n' "$record" | python3 -c '
import sys

fields = dict(token.split("=", 1) for token in sys.stdin.read().split() if "=" in token)
state = fields.get("State", "").split("+", 1)[0]
free_cpus = int(fields.get("CPUEfctv", fields.get("CPUTot", "0"))) - int(fields.get("CPUAlloc", "0"))
free_mem = int(fields.get("RealMemory", "0")) - int(fields.get("AllocMem", "0"))
gres = fields.get("Gres", "")
ready = state == "IDLE" and free_cpus >= 48 and free_mem >= 512 * 1024 and "gpu:gh200:2" in gres
raise SystemExit(0 if ready else 1)
'
}

failures_without_progress=0
while true; do
  state="$(campaign_state)"
  if [[ "$state" == missing ]]; then
    active=1
    failed=0
    succeeded=0
    echo "$(date --iso-8601=seconds) initial manifest not present; allocation may establish it"
  else
    read -r active failed succeeded <<<"$state"
  fi
  echo "$(date --iso-8601=seconds) campaign active=$active failed=$failed succeeded=$succeeded"
  if [[ "$active" == 0 ]]; then
    echo "$(date --iso-8601=seconds) campaign terminal failed=$failed succeeded=$succeeded"
    exit 0
  fi

  jobs="$(squeue -u "${USER:?}" -p gpu-grace-h200-141g -h -o '%i %j %T %N')"
  if [[ -n "$jobs" ]]; then
    echo "$(date --iso-8601=seconds) GH200 request/allocation already present; waiting without submitting another"
    printf '%s\n' "$jobs"
    sleep 60
    continue
  fi

  nodes_ready=1
  for node in "${NODES[@]}"; do
    if ! node_ready "$node"; then nodes_ready=0; fi
  done
  if [[ "$nodes_ready" != 1 ]]; then
    echo "$(date --iso-8601=seconds) required two-node inventory unavailable; recheck in 60s"
    sleep 60
    continue
  fi

  before="$(checkpoint_signature)"
  echo "$(date --iso-8601=seconds) requesting one GH200/node, 48 CPUs and 512G/node, 6h"
  set +e
  srun \
    --immediate=60 \
    --partition=gpu-grace-h200-141g \
    --nodes=2 --nodelist=gpuarm1,gpuarm2 \
    --ntasks=2 --ntasks-per-node=1 \
    --cpus-per-task=48 --gpus-per-node=gh200:1 --mem=512G \
    --time=06:00:00 --job-name=thesis-rq1 \
    --export=ALL,TRITON_REPO="$REPO",CAMPAIGN_SPEC="$SPEC",CAMPAIGN_ID="$CAMPAIGN_ID",RQ1_WORKERS=2 \
    --chdir="$REPO" \
    bash "$REPO/scripts/compute/slurm/campaigns/run_rq1_pair_allocation.sh"
  rc=$?
  set -e
  echo "$(date --iso-8601=seconds) allocation returned exit=$rc"

  state="$(campaign_state)"
  if [[ "$state" == missing ]]; then
    active=1
    failed=0
    succeeded=0
  else
    read -r active failed succeeded <<<"$state"
  fi
  echo "$(date --iso-8601=seconds) post-allocation active=$active failed=$failed succeeded=$succeeded"
  if [[ "$active" == 0 ]]; then
    echo "$(date --iso-8601=seconds) campaign terminal failed=$failed succeeded=$succeeded"
    exit 0
  fi
  after="$(checkpoint_signature)"
  if [[ "$after" != "$before" ]]; then
    failures_without_progress=0
    echo "$(date --iso-8601=seconds) checkpoint progress observed"
  elif ((rc != 0)); then
    ((failures_without_progress+=1))
    echo "$(date --iso-8601=seconds) no checkpoint progress; failures=$failures_without_progress/$NO_PROGRESS_LIMIT"
    if ((failures_without_progress >= NO_PROGRESS_LIMIT)); then
      echo "$(date --iso-8601=seconds) stopping reacquisition for diagnosis; campaign and artifacts preserved"
      exit 2
    fi
  else
    failures_without_progress=0
  fi
done
