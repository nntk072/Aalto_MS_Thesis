#!/usr/bin/env bash
# Run one allocation of the fresh 2025-train/2026-test screen.
# Worker requests are handled by one persistent dispatcher per allocated node.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-replacement18_seed50_train2025_test2026_current_mtf_20261019}"
WORKERS="${CAMPAIGN_WORKERS:-18}"
WORKER_CPUS="${CAMPAIGN_WORKER_CPUS:-5}"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
QUEUE="$OUT_DIR/worker_queue"
JOB_ID="${SLURM_JOB_ID:?}"
DONE="$OUT_DIR/allocation_controller.exit.$JOB_ID"
GATE_DIR="$OUT_DIR/reports/pretraining_gate_$JOB_ID"
GATE_STATUS="$GATE_DIR/status"
GATE_CONTINUE="$GATE_DIR/continue"
OVERRIDE_REASON_FILE="$OUT_DIR/reports/resume_fingerprint_override_reason.txt"
if [[ -z "${CAMPAIGN_RESUME_FINGERPRINT_OVERRIDE_REASON:-}" && -s "$OVERRIDE_REASON_FILE" ]]; then
  CAMPAIGN_RESUME_FINGERPRINT_OVERRIDE_REASON="$(<"$OVERRIDE_REASON_FILE")"
fi
if [[ -n "${CAMPAIGN_RESUME_FINGERPRINT_OVERRIDE_REASON:-}" ]]; then
  if ((${#CAMPAIGN_RESUME_FINGERPRINT_OVERRIDE_REASON} < 12)); then
    echo "fingerprint override reason must be at least 12 characters" >&2
    exit 2
  fi
  export CAMPAIGN_RESUME_FINGERPRINT_OVERRIDE_REASON
fi

cd "$REPO"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/reports"
source "$REPO/scripts/compute/slurm/activate_venv.sh"

export TRITON_REPO="$REPO"
export CAMPAIGN_OUT_DIR="$OUT_DIR"
export CAMPAIGN_WORKER_QUEUE="$QUEUE"
export CAMPAIGN_ALLOCATION_DONE_MARKER="$DONE"
export PATH="$REPO/scripts/compute/slurm/worker_bin:$PATH"
export PYTHONHASHSEED=0

if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
  if [[ ! -s "$OUT_DIR/campaign.json" ]]; then
    mkdir -p "$GATE_DIR"
    while [[ ! -e "$GATE_DIR/abort" ]]; do
      if [[ -s "$GATE_STATUS" && "$(<"$GATE_STATUS")" == "passed" ]]; then break; fi
      sleep 10
    done
    [[ ! -e "$GATE_DIR/abort" ]] || exit 1
    while [[ ! -e "$GATE_CONTINUE" && ! -e "$GATE_DIR/abort" ]]; do sleep 10; done
    [[ ! -e "$GATE_DIR/abort" ]] || exit 1
  fi
  exec python "$REPO/compute/slurm/worker_dispatch.py" --manager \
    >>"$OUT_DIR/logs/worker_manager_$(hostname -s)_${SLURM_JOB_ID}.log" 2>&1
fi

if [[ ! -s "$OUT_DIR/campaign.json" ]]; then
  mkdir -p "$GATE_DIR"
  gate_attempt=0
  while true; do
  ((gate_attempt+=1))
  GATE_RUN_DIR="$GATE_DIR/attempt_$gate_attempt"
  mkdir -p "$GATE_RUN_DIR"
  gate_failed=0
  run_gate_step() {
    local name="$1"
    shift
    local status=0
    if "$@" >"$GATE_RUN_DIR/$name.log" 2>&1; then
      status=0
    else
      status=$?
      gate_failed=1
    fi
    printf '%s\n' "$status" >"$GATE_RUN_DIR/$name.exit"
    echo "pretraining gate attempt=$gate_attempt $name exit=$status"
  }

  run_gate_step ruff_format "$REPO/.venv/bin/ruff" format --check .
  run_gate_step ruff_check "$REPO/.venv/bin/ruff" check .
  run_gate_step mypy "$REPO/.venv/bin/mypy" .
  run_gate_step pytest "$REPO/.venv/bin/pytest" \
    tests/test_envs/test_entry_state_machine.py \
    tests/test_envs/test_baseline_regression.py \
    tests/test_envs/test_bundle_env_equivalence.py \
    tests/test_envs/test_strategy_env.py \
    tests/test_train/test_train_rl_decomposition.py \
    tests/test_matrix_campaign_controller.py \
    tests/test_campaign_artifacts.py \
    tests/test_triton_gh200_capacity.py \
    tests/test_models/test_legacy_checkpoint_migration.py \
    tests/architecture/test_public_imports.py -q

  if ((gate_failed == 0)); then
    printf 'passed\n' >"$GATE_STATUS"
    echo "pretraining CI gate passed; waiting for reviewed commit and push before training"
    break
  fi

  printf 'failed\n' >"$GATE_STATUS"
  echo "pretraining gate failed; preserve logs, repair, then touch $GATE_DIR/retry to rerun" >&2
  while [[ ! -e "$GATE_DIR/retry" && ! -e "$GATE_DIR/abort" ]]; do sleep 10; done
  [[ ! -e "$GATE_DIR/abort" ]] || exit 1
    rm -f "$GATE_DIR/retry" "$GATE_STATUS"
  done

  while [[ ! -e "$GATE_CONTINUE" && ! -e "$GATE_DIR/abort" ]]; do sleep 10; done
  [[ ! -e "$GATE_DIR/abort" ]] || exit 1
fi

python3 - "$DONE" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
finish_allocation() {
  status=$?
  printf '%s\n' "$status" > "$DONE"
  if [[ -n "${manager_pid:-}" ]]; then
    wait "$manager_pid" || manager_status=$?
    if [[ "$status" -eq 0 && "${manager_status:-0}" -ne 0 ]]; then
      status="$manager_status"
    fi
  fi
  echo "replacement allocation job=${SLURM_JOB_ID} exit=$status"
  exit "$status"
}
trap finish_allocation EXIT

# Build the split-specific bar and tick caches once, then keep them for resume.
CACHE_READY="$OUT_DIR/reports/split_cache_ready.json"
if [[ ! -s "$CACHE_READY" ]] || ! python3 -m json.tool "$CACHE_READY" >/dev/null 2>&1; then
  CACHE_BUILD_TMP="$CACHE_READY.job_${SLURM_JOB_ID}.tmp"
  if python -m market_data.commands.prebuild_campaign_cache \
    2>&1 | tee "$OUT_DIR/logs/split_cache_build_${SLURM_JOB_ID}.log" \
    > "$CACHE_BUILD_TMP"; then
    python3 -m json.tool "$CACHE_BUILD_TMP" >/dev/null
    mv "$CACHE_BUILD_TMP" "$CACHE_READY"
  else
    build_status=$?
    rm -f "$CACHE_BUILD_TMP"
    exit "$build_status"
  fi
fi

# Start the local dispatcher and wait until every allocated node has a ready
# dispatcher before the controller can send any worker requests.
mkdir -p "$QUEUE/$SLURM_JOB_ID"
python "$REPO/compute/slurm/worker_dispatch.py" --manager \
  >>"$OUT_DIR/logs/worker_manager_$(hostname -s)_${SLURM_JOB_ID}.log" 2>&1 &
manager_pid=$!
READY="$QUEUE/$SLURM_JOB_ID/$(hostname -s)/ready"
for _ in $(seq 1 60); do
  [[ -e "$READY" ]] && break
  kill -0 "$manager_pid" 2>/dev/null || { echo "local worker manager exited before ready" >&2; exit 1; }
  sleep 1
done
if [[ ! -e "$READY" ]]; then
  echo "local worker manager did not become ready within 60 seconds" >&2
  exit 1
fi

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
MANAGER_DEADLINE=$((SECONDS + 180))
while true; do
  MISSING_NODES=()
  for node in "${ALLOCATED_NODES[@]}"; do
    node_key="${node%%.*}"
    [[ -e "$QUEUE/$SLURM_JOB_ID/$node_key/ready" ]] || MISSING_NODES+=("$node_key")
  done
  if [[ "${#MISSING_NODES[@]}" -eq 0 ]]; then
    break
  fi
  if (( SECONDS >= MANAGER_DEADLINE )); then
    echo "worker managers did not become ready on all allocated nodes; missing=${MISSING_NODES[*]}" >&2
    exit 1
  fi
  sleep 2
done
echo "worker managers ready on all nodes: ${ALLOCATED_NODES[*]}"

# P3 recovery is already complete for this input path; start the requested screen directly.
resume_args=()
if [[ -s "$OUT_DIR/campaign.json" ]]; then
  resume_args+=(--resume)
fi
if [[ -n "${CAMPAIGN_RESUME_FINGERPRINT_OVERRIDE_REASON:-}" ]]; then
  resume_args+=(
    --resume-fingerprint-override-reason
    "$CAMPAIGN_RESUME_FINGERPRINT_OVERRIDE_REASON"
  )
  echo "campaign fingerprint override enabled; resulting training evidence is exploratory"
fi
initial_checkpoint_args=()
if [[ -n "${CAMPAIGN_INITIAL_CHECKPOINTS:-}" && ! -s "$OUT_DIR/campaign.json" ]]; then
  initial_checkpoints="$CAMPAIGN_INITIAL_CHECKPOINTS"
  if [[ "$initial_checkpoints" != /* ]]; then
    initial_checkpoints="$REPO/$initial_checkpoints"
  fi
  if [[ ! -s "$initial_checkpoints" ]]; then
    echo "initial checkpoint map is missing or empty: $initial_checkpoints" >&2
    exit 2
  fi
  initial_checkpoint_args+=(--initial-checkpoints "$initial_checkpoints")
fi

set +e
timeout --signal=TERM --kill-after=2m 20700s \
  python "$REPO/experiments/campaigns/execution.py" \
  --campaign-spec "$REPO/$SPEC" \
  "${resume_args[@]}" \
  "${initial_checkpoint_args[@]}" \
  --workers "$WORKERS" \
  --n-envs 64 \
  --worker-nodes "${SLURM_JOB_NODELIST:?}" \
  --worker-cpus "$WORKER_CPUS" \
  --worker-gpu-type gh200 \
  --worker-gpus-per-node 1 \
  --fast-reports \
  2>&1 | tee -a "$OUT_DIR/logs/controller.log"
controller_status=${PIPESTATUS[0]}
set -e
exit "$controller_status"
