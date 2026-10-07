#!/usr/bin/env bash
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"

PARTITION="${PARTITION:-gpu-grace-h200-141g}"
GPUS="${GPUS:-gh200:1}"
MEM="${MEM:-900G}"
CPUS="${CPUS:-64}"
TIME="${TIME:-6:00:00}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/gpu_matrix_18}"

mkdir -p "$OUT_DIR"

SEED="${SEED:-42}"
STEPS="${STEPS:-20000000}"

N_BATCHES=5
NODES_PER_BATCH=2
NODE_A="gpuarm1"
NODE_B="gpuarm2"

FAILURES=()

for BATCH_IDX in $(seq 0 $((N_BATCHES - 1))); do
  echo "=== BATCH $BATCH_IDX start $(date -Is) ==="

  NODE_A_IDX=$((BATCH_IDX * NODES_PER_BATCH))
  NODE_B_IDX=$((NODE_A_IDX + 1))

  set +e
  srun --partition="$PARTITION" \
       --nodelist="$NODE_A" \
       --gpus="$GPUS" \
       --time="$TIME" \
       --mem="$MEM" \
       --ntasks=1 \
       --cpus-per-task="$CPUS" \
       --exclusive \
       env OUT_DIR="$OUT_DIR" SEED="$SEED" STEPS="$STEPS" \
       bash "$REPO/scripts/run_2variant_node.sh" "$NODE_A_IDX" 0 &
  SRUN_A_PID=$!

  srun --partition="$PARTITION" \
       --nodelist="$NODE_B" \
       --gpus="$GPUS" \
       --time="$TIME" \
       --mem="$MEM" \
       --ntasks=1 \
       --cpus-per-task="$CPUS" \
       --exclusive \
       env OUT_DIR="$OUT_DIR" SEED="$SEED" STEPS="$STEPS" \
       bash "$REPO/scripts/run_2variant_node.sh" "$NODE_B_IDX" 1 &
  SRUN_B_PID=$!

  STATUS_A=0
  STATUS_B=0

  wait "$SRUN_A_PID" || STATUS_A=$?
  wait "$SRUN_B_PID" || STATUS_B=$?
  set -e

  echo "BATCH $BATCH_IDX results: node_a=$STATUS_A node_b=$STATUS_B"

  if [[ "$STATUS_A" -ne 0 || "$STATUS_B" -ne 0 ]]; then
    FAILURES+=("batch_$BATCH_IDX")
    echo "FAILED batch=$BATCH_IDX node_a=$STATUS_A node_b=$STATUS_B" | tee "$OUT_DIR/FAILED.txt"
    # Log failed variant details for debugging
    for v in $(sed -n "$((NODE_A_IDX * 2 + 1))p" "$REPO/scripts/variant_list.txt") \
             $(sed -n "$((NODE_A_IDX * 2 + 2))p" "$REPO/scripts/variant_list.txt") \
             $(sed -n "$((NODE_B_IDX * 2 + 1))p" "$REPO/scripts/variant_list.txt") \
             $(sed -n "$((NODE_B_IDX * 2 + 2))p" "$REPO/scripts/variant_list.txt"); do
      if [[ -f "$OUT_DIR/${v}.failed" ]]; then
        echo "--- $v LOG (last 40 lines) ---"
        tail -n 40 "$OUT_DIR/${v}.log" || true
      fi
    done
    break
  fi

  echo "=== BATCH $BATCH_IDX done $(date -Is) ==="
done

echo "=== MATRIX SUMMARY ==="
echo "Done variants: $(find "$OUT_DIR" -maxdepth 1 -name '*.done' | wc -l)"
echo "Failed variants: $(find "$OUT_DIR" -maxdepth 1 -name '*.failed' | wc -l)"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "FAILED batches: ${FAILURES[*]}"
  exit 1
fi
echo "=== ALL 18 VARIANTS COMPLETE ==="
exit 0
