#!/bin/bash
# Inside one 1-GPU srun: three 20M trains for MODE (overlay or baseline).
# All three share CUDA device 0. RAM is the limit, not a second GPU.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
mkdir -p outputs

MODE="${MODE:-}"
if [[ "$MODE" != "overlay" && "$MODE" != "baseline" ]]; then
  echo "ERROR: MODE must be overlay or baseline" >&2
  exit 1
fi

TRAIN="$REPO/scripts/train_encoder_20m.sh"
if [[ ! -x "$TRAIN" ]]; then
  echo "ERROR: missing $TRAIN" >&2
  exit 1
fi

declare -A PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

echo "=== $MODE wave start $(date -Is) ==="
free -g || true
for arch in tcn gru transformer; do
  echo "LAUNCH mode=$MODE arch=$arch gpu=0"
  CUDA_VISIBLE_DEVICES=0 MODE=$MODE ARCH=$arch bash "$TRAIN" &
  PIDS[$arch]=$!
done

status=0
for arch in tcn gru transformer; do
  if ! wait "${PIDS[$arch]}"; then
    echo "FAILED mode=$MODE arch=$arch"
    status=1
  fi
done
echo "=== $MODE wave done $(date -Is) status=$status ==="
exit "$status"
