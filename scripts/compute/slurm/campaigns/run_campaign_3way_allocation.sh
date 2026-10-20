#!/usr/bin/env bash
# One allocation for the three-strategy exploratory campaign.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/3_strategy_seed42_20m_exploratory_selection_acknowledged_20261014.yaml}"
CAMPAIGN_ID="ladder3_strategy_seed42_20m_exploratory_selection_acknowledged_20261014"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
DONE_MARKER="$OUT_DIR/allocation_controller.exit"
cd "$REPO"

if [[ "${SLURM_PROCID:-}" == "0" ]]; then
  trap 'mkdir -p "$OUT_DIR"; touch "$DONE_MARKER"' EXIT
  mkdir -p "$OUT_DIR/logs"
  # On GH200 nodes this selects the aarch64 project environment.
  # shellcheck disable=SC1091
  source "$REPO/scripts/compute/slurm/activate_venv.sh"
  timeout --signal=TERM --kill-after=2m 20700s \
    python "$REPO/experiments/campaigns/execution.py" \
    --campaign-spec "$REPO/$SPEC" \
    --resume \
    --workers 3 \
    --n-envs 4 \
    --worker-nodes "${SLURM_JOB_NODELIST:?}" \
    --worker-cpus 3 \
    --worker-gpu-type gh200 \
    --worker-gpus-per-node 2 \
    --fast-reports 2>&1 | tee -a "$OUT_DIR/logs/controller.log"
else
  while [[ ! -e "$DONE_MARKER" ]]; do
    sleep 10
  done
fi
