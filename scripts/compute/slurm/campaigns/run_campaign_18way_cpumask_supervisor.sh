#!/usr/bin/env bash
# Keep one corrected two-node GH200 campaign moving across wall-time limits.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/18_variant_seed50_20m_full_split_n64_cpumask_v3_20261016.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016}"
SOURCE_CAMPAIGN="${SOURCE_CAMPAIGN:-$REPO/outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_mainplan_20261016}"
ORIGINAL_CAMPAIGN_ID="$CAMPAIGN_ID"
HANDOFF="$REPO/outputs/gpu_campaigns/$ORIGINAL_CAMPAIGN_ID/reports/continuation_handoff.json"
if [[ -s "$HANDOFF" ]]; then
  mapfile -t handoff_values < <(
    python3 "$REPO/compute/slurm/continuation.py" \
      --repo "$REPO" \
      --source-campaign "$REPO/outputs/gpu_campaigns/$ORIGINAL_CAMPAIGN_ID" \
      --handoff "$HANDOFF"
  )
  if [[ "${#handoff_values[@]}" -ne 3 ]]; then
    echo "invalid continuation handoff; refusing to submit another allocation" >&2
    exit 1
  fi
  CAMPAIGN_ID="${handoff_values[0]}"
  SPEC="${handoff_values[1]}"
  SOURCE_CAMPAIGN="${handoff_values[2]}"
  export CAMPAIGN_ID
  export CAMPAIGN_SPEC="$SPEC"
  export SOURCE_CAMPAIGN
  echo "continuation handoff: $ORIGINAL_CAMPAIGN_ID -> $CAMPAIGN_ID"
fi
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
LOG="$OUT_DIR/logs/allocation_supervisor.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date --iso-8601=seconds) corrected CPU-mask supervisor start host=$(hostname -s)"
source "$REPO/scripts/compute/slurm/campaign_srun_gh200_guard.sh"

exec 9>"$OUT_DIR/.allocation_supervisor.lock"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another supervisor already owns this campaign; exiting"
  exit 0
fi

read_state() {
  [[ -s "$OUT_DIR/campaign.json" ]] || { echo missing; return; }
  PATH=/usr/bin:/bin python3 - "$OUT_DIR/campaign.json" <<'PY'
import json, sys
from collections import Counter
counts = Counter(job["status"] for job in json.load(open(sys.argv[1], encoding="utf-8"))["jobs"])
active = sum(counts.get(name, 0) for name in ("pending", "running", "interrupted"))
print(active, counts.get("failed", 0), counts.get("succeeded", 0))
PY
}

checkpoint_signature() {
  local root="$OUT_DIR/runs/attempts"
  [[ -d "$root" ]] || { echo missing; return; }
  find "$root" -type f -name 'ppo_ckpt_*_steps.zip' -printf '%P %s %T@\n' \
    | sort | sha256sum | awk '{print $1}'
}

allocation_state() {
  local queued
  if ! queued="$(squeue -u "${USER:?}" -h -o '%j %T')"; then
    echo unknown
    return
  fi
  printf '%s\n' "$queued" | awk '
    $1 == "thesis18maskv3" && $2 == "RUNNING" { state="running" }
    $1 == "thesis18maskv3" && ($2 == "PENDING" || $2 == "CONFIGURING") { state="pending" }
    END { print state == "" ? "none" : state }
  '
}

while true; do
  state="$(read_state)"
  if [[ "$state" != missing ]]; then
    read -r active failed succeeded <<<"$state"
    echo "$(date --iso-8601=seconds) campaign active=$active failed=$failed succeeded=$succeeded"
    if [[ "$active" == 0 ]]; then
      [[ "$failed" == 0 && "$succeeded" == 18 ]] && exit 0
      exit 1
    fi
  fi

  allocation_status="$(allocation_state)"
  case "$allocation_status" in
    running)
      echo "$(date --iso-8601=seconds) waiting for running thesis18cpumask allocation"
      sleep 60
      continue
      ;;
    pending|unknown)
      echo "$(date --iso-8601=seconds) existing allocation is $allocation_status; stop without repeated polling or a duplicate request"
      exit 75
      ;;
  esac

  echo "$(date --iso-8601=seconds) requesting two GH200 nodes, 48 CPUs and 512G/node, 6h"
  checkpoints_before="$(checkpoint_signature)"
  set +e
  srun \
    --immediate=60 \
    --partition=gpu-grace-h200-141g \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --cpus-per-task=48 --gpus-per-node=gh200:1 --mem=512G \
    --time=06:00:00 --job-name=thesis18maskv3 \
    --export=ALL,TRITON_REPO="$REPO",CAMPAIGN_SPEC="$SPEC",CAMPAIGN_ID="$CAMPAIGN_ID",SOURCE_CAMPAIGN="$SOURCE_CAMPAIGN" \
    --chdir="$REPO" \
    bash "$REPO/scripts/compute/slurm/campaigns/run_campaign_18way_cpumask_allocation.sh"
  rc=$?
  set -e
  echo "$(date --iso-8601=seconds) allocation returned exit=$rc"
  if [[ -e "$OUT_DIR/gpu_ci_gate.failed" && ! -s "$OUT_DIR/gpu_ci_gate.passed" ]]; then
    echo "GPU CI gate failed; stopping the supervisor without another allocation"
    exit 1
  fi

  state="$(read_state)"
  if [[ "$state" == missing ]]; then
    echo "$(date --iso-8601=seconds) allocation produced no campaign manifest; preserve logs and stop without an automatic retry"
    exit 2
  fi
  read -r active failed succeeded <<<"$state"
  echo "$(date --iso-8601=seconds) campaign after allocation active=$active failed=$failed succeeded=$succeeded"
  if [[ "$active" == 0 ]]; then
    [[ "$failed" == 0 && "$succeeded" == 18 ]] && exit 0
    exit 1
  fi
  checkpoints_after="$(checkpoint_signature)"
  if [[ "$checkpoints_after" != "$checkpoints_before" ]]; then
    echo "$(date --iso-8601=seconds) allocation made checkpoint progress"
  else
    echo "$(date --iso-8601=seconds) allocation made no checkpoint progress; preserve logs and stop without an automatic retry (exit=$rc)"
    exit 2
  fi
  sleep 30
done
