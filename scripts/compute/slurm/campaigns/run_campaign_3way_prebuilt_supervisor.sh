#!/usr/bin/env bash
# Reacquire two GH200s until all three prebuilt-cache runs are terminal.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/3_strategy_prebuilt_cache_seed42_20m_20261014.yaml}"
CAMPAIGN_ID="ladder3_prebuilt_cache_seed42_20m_20261014"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
LOG="$OUT_DIR/logs/allocation_supervisor.log"
MAX_ERRORS="${MAX_ALLOCATION_ERRORS:-3}"
allocation_errors=0

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date --iso-8601=seconds) supervisor start host=$(hostname -s)"

while true; do
  if [[ -s "$OUT_DIR/campaign.json" ]]; then
    state="$(python3 - "$OUT_DIR/campaign.json" <<'PY'
import json, sys
from collections import Counter
counts = Counter(item["status"] for item in json.load(open(sys.argv[1], encoding="utf-8"))["jobs"])
print(sum(counts.get(key, 0) for key in ("pending", "running", "interrupted")), counts.get("failed", 0), counts.get("succeeded", 0))
PY
)"
    read -r active failed succeeded <<<"$state"
    echo "$(date --iso-8601=seconds) campaign active=$active failed=$failed succeeded=$succeeded"
    if [[ "$active" == "0" ]]; then
      [[ "$failed" == "0" && "$succeeded" == "3" ]] && exit 0
      exit 1
    fi
  fi

  echo "$(date --iso-8601=seconds) requesting gpuarm1, 2 GH200, 64 CPUs, 1000G, max 6h"
  set +e
  srun \
    --partition=gpu-grace-h200-141g \
    --nodes=1 --nodelist=gpuarm1 \
    --ntasks=1 --ntasks-per-node=1 \
    --cpus-per-task=64 --gpus-per-node=gh200:2 --mem=1000G \
    --time=06:00:00 --job-name=thesis3pre \
    --export=ALL,TRITON_REPO="$REPO",CAMPAIGN_SPEC="$SPEC" \
    --chdir="$REPO" \
    bash scripts/compute/slurm/campaigns/run_campaign_3way_prebuilt_allocation.sh
  rc=$?
  set -e
  echo "$(date --iso-8601=seconds) allocation returned exit=$rc"

  if [[ ! -s "$OUT_DIR/campaign.json" ]]; then
    ((allocation_errors+=1))
    if (( allocation_errors >= MAX_ERRORS )); then
      echo "allocation retry limit reached before campaign manifest creation"
      exit 2
    fi
    sleep "$((120 * allocation_errors))"
  else
    allocation_errors=0
    sleep 30
  fi
done
