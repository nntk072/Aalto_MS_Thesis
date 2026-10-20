#!/usr/bin/env bash
# Entry point for a one-node, two-GH200 exploratory 18-worker run. The
# controller is rank 0; additional ranks, if any, keep the allocation alive.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml}"
CAMPAIGN_ID="ladder18_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014"
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
    --workers 18 \
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
