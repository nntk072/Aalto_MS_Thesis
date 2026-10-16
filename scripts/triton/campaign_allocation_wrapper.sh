#!/usr/bin/env bash
# Build shared bars and restore only the proven cache-failed identities within
# the already-approved two-node allocation, then run its unchanged controller.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
ALLOCATION_SCRIPT="$REPO/scripts/matrix/run_campaign_18way_cpumask_allocation.sh"
OUT_DIR="$REPO/outputs/gpu_campaigns/${CAMPAIGN_ID:?}"
PREPARED="$OUT_DIR/.cache_recovery_prepared_${SLURM_JOB_ID:?}"
FAILED="$OUT_DIR/.cache_recovery_failed_${SLURM_JOB_ID:?}"
DONE_MARKER="$OUT_DIR/allocation_controller.exit"

if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
  while [[ ! -e "$PREPARED" && ! -e "$FAILED" ]]; do sleep 2; done
  [[ -e "$PREPARED" ]] || exit 1
  while [[ -e "$DONE_MARKER" ]]; do sleep 2; done
  exec bash "$ALLOCATION_SCRIPT"
fi

mkdir -p "$OUT_DIR"
rm -f "$PREPARED" "$FAILED"
if ! source "$REPO/scripts/triton/activate_venv.sh"; then
  touch "$FAILED"
  exit 1
fi
if ! python "$REPO/scripts/triton/prebuild_campaign_bar_cache.py"; then
  touch "$FAILED"
  exit 1
fi
if ! python "$REPO/scripts/triton/prepare_campaign_resume.py" --campaign-dir "$OUT_DIR"; then
  touch "$FAILED"
  exit 1
fi
rm -f "$DONE_MARKER"
touch "$PREPARED"
exec bash "$ALLOCATION_SCRIPT"
