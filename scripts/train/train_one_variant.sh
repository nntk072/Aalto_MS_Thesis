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
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$LOG_DIR"

source "$REPO/scripts/triton/activate_venv.sh"

LOG="$LOG_DIR/${VARIANT}.log"
DONE="$OUT_DIR/${VARIANT}.done"
FAILED="$OUT_DIR/${VARIANT}.failed"
RESUME="${RESUME:-}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export QUANT_RL_MAX_N_ENVS="${QUANT_RL_MAX_N_ENVS:-64}"
export PYTHONUNBUFFERED=1

STRATEGY=$(python -c "
from quant_rl.train.ablation_utils import load_variant_config
import sys
variant = load_variant_config(sys.argv[1])
print(variant.get('strategy', 'baseline'))
" "$VARIANT")

STRATEGY_ACTIONS=$(python -c "
from quant_rl.train.ablation_utils import load_variant_config
import sys
v = load_variant_config(sys.argv[1])
print(v.get('strategy_actions', False))
" "$VARIANT")

if [[ "$STRATEGY" == "po3_ifvg" || "$STRATEGY" == "distribution" || "$STRATEGY_ACTIONS" == "True" ]]; then
  BASE_CONFIG="config/features/full_po3_mtf.yaml"
  EXTRA_OVERRIDES=(
    "features.include_session_ohlc=true"
    "features.liquidity.enabled=true"
    "features.po3_state_mtf.enabled=true"
    "features.ifvg_mtf.enabled=true"
    "features.include_strategy_state=true"
    "env.open_manipulation_bars=0"
    "env.peak_trailing_dd_limit=0"
    "ftmo.trailing_dd_limit=0.07"
  )
else
  BASE_CONFIG="quant_rl/config/default.yaml"
  EXTRA_OVERRIDES=()
fi

echo "=== train $VARIANT strategy=$STRATEGY seed=$SEED gpu=$GPU_ID log=$LOG ==="
if [[ -n "$RESUME" ]]; then
  echo "=== resuming checkpoint $RESUME ==="
fi

TRAIN_ARGS=(
  --variant "$VARIANT"
  --config "$BASE_CONFIG"
  --seed "$SEED"
  --seeds "$SEED"
  --arch tcn
  --out "$OUT_DIR/$VARIANT"
)
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi

set +e
rm -f "$FAILED"
python -u -m quant_rl.train.train_rl \
  "${TRAIN_ARGS[@]}" \
  "${EXTRA_OVERRIDES[@]}" \
  ppo.total_timesteps="$STEPS" \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "$STATUS" -eq 0 ]]; then
  rm -f "$FAILED"
  touch "$DONE"
else
  touch "$FAILED"
fi

echo "=== train $VARIANT exit=$STATUS ==="
exit "$STATUS"
