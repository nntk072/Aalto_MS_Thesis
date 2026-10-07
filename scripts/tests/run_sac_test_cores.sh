#!/bin/bash
# SAC smoke test on a GPU-partition node.
#
# WHY THIS IS CPU-ONLY (no --gpus):
#   test_sac_smoke builds small MLPs and steps SAC a few times -- it is CPU
#   bound, not GPU bound. It times out on the login node purely for lack of
#   free cores while the two 20M trainers are running. The GPU nodes show
#   "48/96/0/144": 96 idle cores each. Borrowing cores is what actually helps,
#   and omitting --gpus keeps this at zero GPU quota so it cannot compete with
#   the training jobs for AssocGrpGRESRunMinutes.
#
#   srun needs explicit user permission in chat before running this.
set -uo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
source "$REPO/scripts/triton/activate_venv.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/outputs/pytest_sac_gpu_$STAMP.log"
mkdir -p "$REPO/outputs"

echo "node=$(hostname)  $(date '+%F %T')  cores=$(nproc)"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY
echo "log: $LOG"

# Generous per-test timeout: on a busy login node SAC can need >100s, which
# looked like a failure but was only contention.
python -m pytest "${@:-tests/test_train/test_sac_smoke.py}" \
  -q -p no:cacheprovider --timeout=600 -rf --tb=line \
  2>&1 | tee "$LOG"
echo "rc=${PIPESTATUS[0]}"
echo "finished $(date '+%F %T')"