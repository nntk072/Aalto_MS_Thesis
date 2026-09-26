#!/usr/bin/env bash
# Run the six checkpoint rescores on the already-allocated GH200 shell.
# Two models at a time, one per GPU. Does not allocate and does not exit the parent shell.
set -u
REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
# shellcheck source=/dev/null
source "$REPO/scripts/triton/activate_venv.sh"
export PYTHONPATH="$REPO"
mkdir -p outputs/rescore_sl_fill/logs

run_pair() {
  local a="$1" b="$2"
  echo "=== pair $a (gpu0) + $b (gpu1) $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=0 python scripts/rescore_sl_fill.py "$a" > "outputs/rescore_sl_fill/logs/${a}.log" 2>&1 &
  local pa=$!
  CUDA_VISIBLE_DEVICES=1 python scripts/rescore_sl_fill.py "$b" > "outputs/rescore_sl_fill/logs/${b}.log" 2>&1 &
  local pb=$!
  local fail=0
  wait "$pa" || fail=1
  wait "$pb" || fail=1
  echo "=== pair done $a $b fail=$fail $(date -Is) ==="
  return "$fail"
}

run_pair base_gru po3_tcn || echo "pair1 failed"
run_pair base_tcn po3_gru || echo "pair2 failed"
run_pair base_tf po3_tf || echo "pair3 failed"
echo "=== all rescores finished $(date -Is) ==="
