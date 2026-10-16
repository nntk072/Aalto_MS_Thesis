#!/usr/bin/env bash
# Bootstrap one guarded allocation for cache recovery and the GPU CI gate.
# Once that allocation is released, hand continuity to the campaign supervisor.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/18_variant_seed50_20m_full_split_n64_cpumask_v3_20261016.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016}"
SOURCE_CAMPAIGN="${SOURCE_CAMPAIGN:-$REPO/outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_mainplan_20261016}"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
LOG="$OUT_DIR/logs/allocation_supervisor.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date --iso-8601=seconds) guarded recovery bootstrap start host=$(hostname -s)"

exec 9>"$OUT_DIR/.allocation_supervisor.lock"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another supervisor owns this campaign; exiting"
  exit 0
fi

if [[ -s "$OUT_DIR/campaign.json" ]]; then
  state="$(python3 - "$OUT_DIR/campaign.json" <<'PY'
import json, sys
from collections import Counter
counts = Counter(job["status"] for job in json.load(open(sys.argv[1], encoding="utf-8"))["jobs"])
print(sum(counts.get(name, 0) for name in ("pending", "running", "interrupted")),
      counts.get("failed", 0), counts.get("succeeded", 0))
PY
)"
  read -r active failed succeeded <<<"$state"
  echo "$(date --iso-8601=seconds) preflight campaign active=$active failed=$failed succeeded=$succeeded"
  if [[ "$active" == 0 && "$failed" == 18 && "$succeeded" == 0 ]]; then
    echo "$(date --iso-8601=seconds) requesting one guarded 2-node GH200 recovery allocation"
    srun \
      --partition=gpu-grace-h200-141g \
      --nodes=2 --ntasks=2 --ntasks-per-node=1 \
      --cpus-per-task=48 --gpus-per-node=gh200:1 --mem=512G \
      --time=06:00:00 --job-name=thesis18maskv3 \
      --export=ALL,TRITON_REPO="$REPO",CAMPAIGN_SPEC="$SPEC",CAMPAIGN_ID="$CAMPAIGN_ID",SOURCE_CAMPAIGN="$SOURCE_CAMPAIGN" \
      --chdir="$REPO" \
      bash "$REPO/scripts/matrix/run_campaign_18way_cpumask_allocation.sh"
    echo "$(date --iso-8601=seconds) bootstrap allocation returned; handing off to campaign supervisor"
  fi
fi

exec bash "$REPO/scripts/matrix/run_campaign_18way_cpumask_supervisor.sh"
