#!/usr/bin/env bash
# Re-export and evaluate 18 already-trained 20M checkpoints on an H200 node.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/18_variant_seed42_20m_finalization_20261015.yaml}"
CAMPAIGN_ID="ladder18_seed42_20m_finalization_20261015"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
CHECKPOINT_MAP="$OUT_DIR/checkpoint_migration_map.json"

cd "$REPO"
mkdir -p "$OUT_DIR/logs"
source "$REPO/scripts/triton/activate_venv.sh"

set +e
python "$REPO/scripts/matrix/campaign_controller.py" \
  --campaign-spec "$REPO/$SPEC" \
  --initial-checkpoints "$CHECKPOINT_MAP" \
  --workers 4 \
  --n-envs 64 \
  --worker-nodes "${SLURM_JOB_NODELIST:?}" \
  --worker-cpus 10 \
  --worker-gpu-type h200 \
  --worker-gpus-per-node 1 \
  --fast-reports \
  2>&1 | tee -a "$OUT_DIR/logs/controller.log"
status=${PIPESTATUS[0]}
set -e

echo "Finalization controller exit=$status; keeping this allocation shell open for inspection."
exec bash -i
