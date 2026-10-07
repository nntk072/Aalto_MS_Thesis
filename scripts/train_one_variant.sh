#!/usr/bin/env bash
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"

VARIANT="${VARIANT:-}"
SEED="${SEED:-42}"
STEPS="${STEPS:-20000000}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/gpu_matrix_18}"

if [[ -z "$VARIANT" ]]; then
  echo "ERROR: VARIANT is required" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

source "$REPO/scripts/triton/activate_venv.sh"

LOG="$OUT_DIR/${VARIANT}.log"
DONE="$OUT_DIR/${VARIANT}.done"
FAILED="$OUT_DIR/${VARIANT}.failed"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export QUANT_RL_MAX_N_ENVS="${QUANT_RL_MAX_N_ENVS:-64}"
export PYTHONUNBUFFERED=1

echo "=== train $VARIANT seed=$SEED gpu=$GPU_ID log=$LOG ==="

set +e
python -u -m quant_rl.train.train_rl \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --seeds "$SEED" \
  --arch tcn \
  --out "$OUT_DIR" \
  ppo.total_timesteps="$STEPS" \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "$STATUS" -eq 0 ]]; then
  touch "$DONE"
else
  touch "$FAILED"
fi

echo "=== train $VARIANT exit=$STATUS ==="
exit "$STATUS"
