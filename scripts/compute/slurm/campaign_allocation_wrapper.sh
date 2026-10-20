#!/usr/bin/env bash
# Build shared bars and restore only the proven cache-failed identities within
# the already-approved two-node allocation, then run its unchanged controller.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
ALLOCATION_SCRIPT="$REPO/scripts/compute/slurm/campaigns/run_campaign_18way_cpumask_allocation.sh"
OUT_DIR="$REPO/outputs/gpu_campaigns/${CAMPAIGN_ID:?}"
PREPARED="$OUT_DIR/.cache_recovery_prepared_${SLURM_JOB_ID:?}"
FAILED="$OUT_DIR/.cache_recovery_failed_${SLURM_JOB_ID:?}"
DONE_MARKER="$OUT_DIR/allocation_controller.exit"
GPU_GATE_PASSED="$OUT_DIR/gpu_ci_gate.passed"
GPU_GATE_FAILED="$OUT_DIR/gpu_ci_gate.failed"

if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
  while [[ ! -e "$PREPARED" ]]; do sleep 2; done
  while [[ -e "$DONE_MARKER" ]]; do sleep 2; done
  source "$REPO/scripts/compute/slurm/activate_venv.sh"
  export CAMPAIGN_OUT_DIR="$OUT_DIR"
  export CAMPAIGN_WORKER_QUEUE="$OUT_DIR/worker_queue"
  export PATH="$REPO/scripts/compute/slurm/worker_bin:$PATH"
  exec python "$REPO/compute/slurm/worker_dispatch.py" --manager \
    >>"$OUT_DIR/logs/worker_manager_$(hostname -s).log" 2>&1
fi

mkdir -p "$OUT_DIR"
rm -f "$PREPARED" "$FAILED"
current_head="$(git -C "$REPO" rev-parse HEAD)"
if [[ -n "$(git -C "$REPO" status --porcelain)" \
  || ! -s "$GPU_GATE_PASSED" \
  || "$(cat "$GPU_GATE_PASSED")" != "$current_head" ]]; then
  if [[ -s "$GPU_GATE_PASSED" ]]; then
    cp -p "$GPU_GATE_PASSED" "$GPU_GATE_PASSED.stale_${SLURM_JOB_ID}"
    rm "$GPU_GATE_PASSED"
  fi
fi
runtime_retry=1
while ! source "$REPO/scripts/compute/slurm/activate_venv.sh"; do
  touch "$FAILED" "$GPU_GATE_FAILED"
  retry_marker="$OUT_DIR/retry_runtime_setup.$runtime_retry"
  echo "GPU runtime setup failed; fix the cause, then touch $retry_marker"
  while [[ ! -e "$retry_marker" ]]; do sleep 20; done
  mv "$retry_marker" "$retry_marker.consumed"
  rm -f "$FAILED"
  ((runtime_retry+=1))
done

cache_retry=1
while ! python -m market_data.commands.prebuild_campaign_cache; do
  touch "$FAILED" "$GPU_GATE_FAILED"
  retry_marker="$OUT_DIR/retry_bar_cache_prebuild.$cache_retry"
  echo "Bar-cache prebuild failed; fix the cause, then touch $retry_marker"
  while [[ ! -e "$retry_marker" ]]; do sleep 20; done
  mv "$retry_marker" "$retry_marker.consumed"
  rm -f "$FAILED"
  ((cache_retry+=1))
done

if [[ -s "$OUT_DIR/campaign.json" ]]; then
  recovery_retry=1
  while ! python "$REPO/compute/slurm/resume.py" --campaign-dir "$OUT_DIR"; do
    touch "$FAILED" "$GPU_GATE_FAILED"
    retry_marker="$OUT_DIR/retry_campaign_recovery.$recovery_retry"
    echo "Campaign recovery validation failed; fix the cause, then touch $retry_marker"
    while [[ ! -e "$retry_marker" ]]; do sleep 20; done
    mv "$retry_marker" "$retry_marker.consumed"
    rm -f "$FAILED"
    ((recovery_retry+=1))
  done

  index_retry=1
  while ! python "$REPO/scripts/research/thesis/evidence/build_campaign_evidence_index.py" \
    --campaign-dir "$OUT_DIR" \
    --output "$OUT_DIR/reports/thesis_evidence_index_before_${SLURM_JOB_ID}.json"; do
    touch "$FAILED" "$GPU_GATE_FAILED"
    retry_marker="$OUT_DIR/retry_campaign_evidence_index.$index_retry"
    echo "Campaign evidence index failed; fix the cause, then touch $retry_marker"
    while [[ ! -e "$retry_marker" ]]; do sleep 20; done
    mv "$retry_marker" "$retry_marker.consumed"
    rm -f "$FAILED"
    ((index_retry+=1))
  done
else
  echo "New checkpoint-lineage campaign; source checkpoints are validated by the controller."
fi
dispatch_retry=1
while true; do
  if python -m pytest \
    "$REPO/tests/test_triton_campaign_recovery.py" \
    "$REPO/tests/test_triton_prebuild_cache.py" \
    "$REPO/tests/test_triton_campaign_worker_dispatch.py" \
    "$REPO/tests/test_thesis_campaign_evidence_index.py" \
    -q; then
    break
  fi
  touch "$GPU_GATE_FAILED"
  retry_marker="$OUT_DIR/retry_worker_dispatch_gate.$dispatch_retry"
  echo "Worker-dispatch tests failed; fix the cause, then touch $retry_marker"
  while [[ ! -e "$retry_marker" ]]; do sleep 20; done
  mv "$retry_marker" "$retry_marker.consumed"
  ((dispatch_retry+=1))
done
rm -f "$DONE_MARKER"
export CAMPAIGN_OUT_DIR="$OUT_DIR"
export CAMPAIGN_WORKER_QUEUE="$OUT_DIR/worker_queue"
export PATH="$REPO/scripts/compute/slurm/worker_bin:$PATH"
if [[ -e "$CAMPAIGN_WORKER_QUEUE/$SLURM_JOB_ID" ]]; then
  echo "ERROR: worker queue already exists for Slurm job $SLURM_JOB_ID" >&2
  touch "$FAILED"
  exit 1
fi
mkdir -p "$CAMPAIGN_WORKER_QUEUE/$SLURM_JOB_ID"
MANAGER_LOG="$OUT_DIR/logs/worker_manager_$(hostname -s).log"
manager_retry=1
while true; do
  ready_marker="$CAMPAIGN_WORKER_QUEUE/$SLURM_JOB_ID/$(hostname -s)/ready"
  rm -f "$ready_marker"
  python "$REPO/compute/slurm/worker_dispatch.py" --manager \
    >>"$MANAGER_LOG" 2>&1 &
  manager_pid=$!
  ready=0
  for _ in $(seq 1 60); do
    if [[ -e "$ready_marker" ]]; then
      ready=1
      break
    fi
    if ! kill -0 "$manager_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  touch "$FAILED" "$GPU_GATE_FAILED"
  kill "$manager_pid" 2>/dev/null || true
  wait "$manager_pid" || true
  retry_marker="$OUT_DIR/retry_worker_manager.$manager_retry"
  echo "Node worker manager failed readiness; fix the cause, then touch $retry_marker"
  while [[ ! -e "$retry_marker" ]]; do sleep 20; done
  mv "$retry_marker" "$retry_marker.consumed"
  rm -f "$FAILED"
  ((manager_retry+=1))
done
touch "$PREPARED"
set +e
bash "$ALLOCATION_SCRIPT"
allocation_status=$?
set -e
wait "$manager_pid" || manager_status=$?
if ! python "$REPO/scripts/research/thesis/evidence/build_campaign_evidence_index.py" \
  --campaign-dir "$OUT_DIR" \
  --output "$OUT_DIR/reports/thesis_evidence_index_after_${SLURM_JOB_ID}.json"; then
  echo "ERROR: could not finalize the campaign evidence index" >&2
  if [[ "${allocation_status:-0}" -eq 0 ]]; then
    allocation_status=1
  fi
fi
if [[ "$allocation_status" -eq 0 && "${manager_status:-0}" -ne 0 ]]; then
  allocation_status="$manager_status"
fi
exit "${allocation_status:-${manager_status:-0}}"
