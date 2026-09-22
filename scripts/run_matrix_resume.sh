#!/bin/bash
# Resume specific matrix legs inside an existing 2-GPU srun shell.
# Example:
#   RUNS="overlay:transformer" bash scripts/run_matrix_resume.sh
#   RUNS="baseline:tcn baseline:gru baseline:transformer" WAVE_PARALLEL=3 bash scripts/run_matrix_resume.sh
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"

TRAIN="$REPO/scripts/train_encoder_20m.sh"
RUNS="${RUNS:-}"
if [[ -z "$RUNS" ]]; then
  echo "ERROR: set RUNS, e.g. RUNS='overlay:transformer baseline:transformer'" >&2
  exit 1
fi

gpu_for_arch() {
  case "$1" in
    tcn) echo 0 ;;
    gru) echo 1 ;;
    transformer) echo 0 ;;
    *) echo "ERROR: unknown arch $1" >&2; exit 1 ;;
  esac
}

declare -a PIDS=()
for item in $RUNS; do
  mode="${item%%:*}"
  arch="${item##*:}"
  gpu="$(gpu_for_arch "$arch")"
  echo "RESUME mode=$mode arch=$arch gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu MODE=$mode ARCH=$arch bash "$TRAIN" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
