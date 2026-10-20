#!/usr/bin/env bash
# Reacquire the selected two-node shape and resume interrupted workers safely.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/18_variant_seed42_20m_purelegacy_shared_bundle_n64_20261014.yaml}"
CAMPAIGN_ID="ladder18_seed42_20m_purelegacy_shared_bundle_n64_20261014"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
LOG="$OUT_DIR/logs/allocation_supervisor.log"
MAX_ERRORS="${MAX_ALLOCATION_ERRORS:-3}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date --iso-8601=seconds) supervisor start host=$(hostname -s)"

read_state() {
  [[ -s "$OUT_DIR/campaign.json" ]] || { echo missing; return; }
  python3 - "$OUT_DIR/campaign.json" <<'PY'
import json, sys
from collections import Counter
counts = Counter(job["status"] for job in json.load(open(sys.argv[1], encoding="utf-8"))["jobs"])
active = sum(counts.get(name, 0) for name in ("pending", "running", "interrupted"))
print(active, counts.get("failed", 0), counts.get("succeeded", 0))
PY
}

errors=0
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

  echo "$(date --iso-8601=seconds) requesting two nodes, one GH200/node, 48 CPUs and 512G/node, 6h"
  set +e
  srun \
    --partition=gpu-grace-h200-141g \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --cpus-per-task=48 --gpus-per-node=gh200:1 --mem=512G \
    --time=06:00:00 --job-name=thesis18x2 \
    --export=ALL,TRITON_REPO="$REPO",CAMPAIGN_SPEC="$SPEC" \
    --chdir="$REPO" \
    bash scripts/compute/slurm/campaigns/run_campaign_18way_two_node_allocation.sh
  rc=$?
  set -e
  echo "$(date --iso-8601=seconds) allocation returned exit=$rc"

  state="$(read_state)"
  if [[ "$state" == missing ]]; then
    ((errors+=1))
    if (( errors >= MAX_ERRORS )); then echo "retry limit reached before campaign manifest"; exit 2; fi
    sleep "$((120 * errors))"
    continue
  fi
  read -r active failed succeeded <<<"$state"
  echo "$(date --iso-8601=seconds) campaign after allocation active=$active failed=$failed succeeded=$succeeded"
  if [[ "$active" == 0 ]]; then
    [[ "$failed" == 0 && "$succeeded" == 18 ]] && exit 0
    exit 1
  fi
  if (( rc != 0 )); then ((errors+=1)); else errors=0; fi
  if (( errors >= MAX_ERRORS )); then echo "allocation retry limit reached; inspect this log"; exit 2; fi
  sleep 30
done
