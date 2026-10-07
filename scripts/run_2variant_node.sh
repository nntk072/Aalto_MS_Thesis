#!/usr/bin/env bash
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"

NODE_IDX="${1:-0}"
GPU_ID="${2:-0}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/gpu_matrix_18}"

VARIANT_LIST="$REPO/scripts/variant_list.txt"
if [[ ! -f "$VARIANT_LIST" ]]; then
  echo "ERROR: missing $VARIANT_LIST" >&2
  exit 1
fi

LINE1=$((NODE_IDX * 2))
LINE2=$((LINE1 + 1))

VARIANT1=$(sed -n "$((LINE1 + 1))p" "$VARIANT_LIST")
VARIANT2=$(sed -n "$((LINE2 + 1))p" "$VARIANT_LIST")

if [[ -z "$VARIANT1" || -z "$VARIANT2" ]]; then
  echo "ERROR: not enough variants in list for NODE_IDX=$NODE_IDX" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

SEED="${SEED:-42}"
STEPS="${STEPS:-20000000}"

echo "=== node_$NODE_IDX gpu=$GPU_ID v1=$VARIANT1 v2=$VARIANT2 ==="

set +e
VARIANT="$VARIANT1" SEED="$SEED" STEPS="$STEPS" GPU_ID="$GPU_ID" OUT_DIR="$OUT_DIR" \
  bash "$REPO/scripts/train_one_variant.sh" &
PID1=$!

VARIANT="$VARIANT2" SEED="$SEED" STEPS="$STEPS" GPU_ID="$GPU_ID" OUT_DIR="$OUT_DIR" \
  bash "$REPO/scripts/train_one_variant.sh" &
PID2=$!

wait "$PID1"
STATUS1=$?
wait "$PID2"
STATUS2=$?
set -e

if [[ "$STATUS1" -eq 0 && "$STATUS2" -eq 0 ]]; then
  touch "$OUT_DIR/node_${NODE_IDX}.done"
  echo "=== node_$NODE_IDX done ==="
  exit 0
else
  touch "$OUT_DIR/node_${NODE_IDX}.failed"
  echo "=== node_$NODE_IDX FAILED v1=$STATUS1 v2=$STATUS2 ==="
  exit 1
fi
