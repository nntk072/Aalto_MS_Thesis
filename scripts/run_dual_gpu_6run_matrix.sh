#!/bin/bash
# Inside one 2×GH200 srun: wave 1 = 3 overlays in parallel, wave 2 = 3 baselines.
# GPU pinning: TCN+Transformer on GPU0, GRU on GPU1.
# OOM fallback: WAVE_PARALLEL=2 runs 2 parallel + 1 sequential per wave.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
mkdir -p outputs

TRAIN="$REPO/scripts/train_encoder_20m.sh"
if [[ ! -x "$TRAIN" ]]; then
  echo "ERROR: missing $TRAIN" >&2
  exit 1
fi

WAVE_PARALLEL="${WAVE_PARALLEL:-3}"
if [[ "$WAVE_PARALLEL" != "2" && "$WAVE_PARALLEL" != "3" ]]; then
  echo "ERROR: WAVE_PARALLEL must be 2 or 3, got: $WAVE_PARALLEL" >&2
  exit 1
fi

FAILED=()
declare -A PIDS=()
declare -a ARCHS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

log_mem() {
  echo "--- $(date -Is) memory ---"
  free -g || true
}

launch_train() {
  local mode=$1 arch=$2 gpu=$3
  echo "LAUNCH mode=$mode arch=$arch gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu MODE=$mode ARCH=$arch bash "$TRAIN" &
  PIDS[$arch]=$!
  ARCHS+=("$arch")
}

wait_wave() {
  local mode=$1
  local arch pid status=0
  for arch in "${ARCHS[@]}"; do
    pid="${PIDS[$arch]:-}"
    if [[ -z "$pid" ]]; then
      continue
    fi
    if ! wait "$pid"; then
      echo "FAILED mode=$mode arch=$arch pid=$pid"
      FAILED+=("${mode}:${arch}")
      status=1
    fi
  done
  unset PIDS
  declare -A PIDS=()
  ARCHS=()
  return "$status"
}

run_wave_parallel3() {
  local mode=$1
  echo "=== WAVE $mode (3 parallel) start $(date -Is) ==="
  log_mem
  launch_train "$mode" tcn 0
  launch_train "$mode" gru 1
  launch_train "$mode" transformer 0
  wait_wave "$mode" || true
  echo "=== WAVE $mode (3 parallel) done $(date -Is) ==="
  log_mem
}

run_wave_parallel2() {
  local mode=$1
  echo "=== WAVE $mode (2+1) start $(date -Is) ==="
  log_mem
  launch_train "$mode" tcn 0
  launch_train "$mode" gru 1
  wait_wave "$mode" || true
  launch_train "$mode" transformer 0
  wait_wave "$mode" || true
  echo "=== WAVE $mode (2+1) done $(date -Is) ==="
  log_mem
}

run_wave() {
  if [[ "$WAVE_PARALLEL" == "3" ]]; then
    run_wave_parallel3 "$1"
  else
    run_wave_parallel2 "$1"
  fi
}

echo "MATRIX start $(date -Is) wave_parallel=$WAVE_PARALLEL"
python -c "import torch; print('torch cuda count', torch.cuda.device_count())" || true
log_mem

run_wave overlay
run_wave baseline

if ((${#FAILED[@]} > 0)); then
  echo "MATRIX finished with failures: ${FAILED[*]}"
  exit 1
fi
echo "MATRIX finished $(date -Is)"
