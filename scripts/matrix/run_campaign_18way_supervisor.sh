#!/usr/bin/env bash
# Keep the exploratory 18-variant campaign moving across legal six-hour
# one-node/two-GH200 allocations. Only interrupted/pending runs are continued; terminal
# failures are preserved by campaign_controller.py and never retried.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml}"
CAMPAIGN_ID="ladder18_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
LOG="$OUT_DIR/logs/allocation_supervisor.log"
DONE_MARKER="$OUT_DIR/allocation_controller.exit"
MAX_ALLOCATION_ERRORS="${MAX_ALLOCATION_ERRORS:-3}"
SLEEP_BETWEEN_ALLOCATIONS="${SLEEP_BETWEEN_ALLOCATIONS:-30}"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date --iso-8601=seconds) supervisor start host=$(hostname -s)"

read_statuses() {
  if [[ ! -s "$OUT_DIR/campaign.json" ]]; then
    echo "missing"
    return
  fi
  python3 - "$OUT_DIR/campaign.json" <<'PY'
import json, sys
from collections import Counter
data = json.load(open(sys.argv[1], encoding="utf-8"))
counts = Counter(job["status"] for job in data["jobs"])
print(" ".join(f"{key}={counts.get(key, 0)}" for key in ("pending", "running", "interrupted", "succeeded", "failed")))
print("active=" + str(sum(counts.get(key, 0) for key in ("pending", "running", "interrupted"))))
print("failed=" + str(counts.get("failed", 0)))
print("succeeded=" + str(counts.get("succeeded", 0)))
PY
}

read_signature() {
  if [[ ! -s "$OUT_DIR/campaign.json" ]]; then
    echo "missing"
    return
  fi
  python3 - "$OUT_DIR/campaign.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
jobs = sorted(
    ({key: job.get(key) for key in ("key", "status", "attempts")} for job in data["jobs"]),
    key=lambda job: job["key"],
)
print(json.dumps(jobs, sort_keys=True, separators=(",", ":")))
PY
}

allocation_errors=0
while true; do
  state="$(read_statuses)"
  if [[ "$state" != "missing" ]]; then
    active="$(awk -F= '/^active=/{print $2}' <<<"$state")"
    failed="$(awk -F= '/^failed=/{print $2}' <<<"$state")"
    succeeded="$(awk -F= '/^succeeded=/{print $2}' <<<"$state")"
    echo "$(date --iso-8601=seconds) campaign $state"
    if [[ "$active" == "0" ]]; then
      if [[ "$failed" == "0" && "$succeeded" == "18" ]]; then
        echo "$(date --iso-8601=seconds) campaign complete: 18 succeeded"
        exit 0
      fi
      echo "$(date --iso-8601=seconds) campaign terminal with failures=$failed succeeded=$succeeded"
      exit 1
    fi
  fi

  before_signature="$(read_signature)"
  rm -f "$DONE_MARKER"
  echo "$(date --iso-8601=seconds) requesting gpuarm1, 2 GH200, 64 CPUs, 1000G, max 6h"
  set +e
  srun \
    --partition=gpu-grace-h200-141g \
    --nodes=1 \
    --nodelist=gpuarm1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=64 \
    --gpus-per-node=gh200:2 \
    --mem=1000G \
    --time=06:00:00 \
    --job-name=thesis18x \
    --export=ALL,TRITON_REPO="$REPO",CAMPAIGN_SPEC="$SPEC" \
    --chdir="$REPO" \
    bash scripts/matrix/run_campaign_allocation.sh
  rc=$?
  set -e
  echo "$(date --iso-8601=seconds) allocation returned exit=$rc"

  state="$(read_statuses)"
  if [[ "$state" == "missing" ]]; then
    ((allocation_errors+=1))
    echo "$(date --iso-8601=seconds) no campaign manifest after allocation attempt errors=$allocation_errors/$MAX_ALLOCATION_ERRORS"
    if (( allocation_errors >= MAX_ALLOCATION_ERRORS )); then
      echo "allocation retry limit reached; inspect Slurm and this log before restarting"
      exit 2
    fi
    sleep "$((120 * allocation_errors))"
  else
    echo "$(date --iso-8601=seconds) campaign after allocation: $state"
    after_signature="$(read_signature)"
    if [[ "$before_signature" == "$after_signature" && "$rc" != "0" ]]; then
      ((allocation_errors+=1))
      echo "$(date --iso-8601=seconds) allocation made no campaign progress errors=$allocation_errors/$MAX_ALLOCATION_ERRORS"
      if (( allocation_errors >= MAX_ALLOCATION_ERRORS )); then
        echo "allocation retry limit reached without campaign progress; inspect Slurm and this log before restarting"
        exit 2
      fi
      sleep "$((120 * allocation_errors))"
      continue
    fi
    allocation_errors=0
    sleep "$SLEEP_BETWEEN_ALLOCATIONS"
  fi
done
