#!/usr/bin/env bash
# Run the frozen selected-pair campaign in one existing GH200 allocation.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261019.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261019}"
WORKERS="${RQ1_WORKERS:-${SLURM_NNODES:-1}}"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
QUEUE="$OUT_DIR/worker_queue"
JOB_ID="${SLURM_JOB_ID:?must run inside an existing Slurm allocation}"
DONE="$OUT_DIR/allocation_controller.exit.$JOB_ID"

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
  exec python "$REPO/compute/slurm/worker_dispatch.py" --manager \
    >>"$OUT_DIR/logs/worker_manager_$(hostname -s)_${JOB_ID}.log" 2>&1
fi

python3 - "$DONE" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).unlink(missing_ok=True)
PY

manager_pid=""
finish_allocation() {
  status=$?
  printf '%s\n' "$status" >"$DONE"
  if [[ -n "$manager_pid" ]]; then
    if wait "$manager_pid"; then
      manager_status=0
    else
      manager_status=$?
    fi
    if [[ "$status" -eq 0 && "$manager_status" -ne 0 ]]; then
      status="$manager_status"
    fi
  fi
  echo "RQ1 pair allocation job=$JOB_ID exit=$status"
  exit "$status"
}
trap finish_allocation EXIT

READY="$QUEUE/$JOB_ID/$(hostname -s)/ready"
mkdir -p "$QUEUE/$JOB_ID"
python "$REPO/compute/slurm/worker_dispatch.py" --manager \
  >>"$OUT_DIR/logs/worker_manager_$(hostname -s)_${JOB_ID}.log" 2>&1 &
manager_pid=$!
for _ in $(seq 1 60); do
  [[ -e "$READY" ]] && break
  kill -0 "$manager_pid" 2>/dev/null || { echo "worker manager exited before ready" >&2; exit 1; }
  sleep 1
done
if [[ ! -e "$READY" ]]; then
  echo "worker manager did not become ready within 60 seconds" >&2
  exit 1
fi

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
deadline=$((SECONDS + 180))
while true; do
  missing=()
  for node in "${ALLOCATED_NODES[@]}"; do
    [[ -e "$QUEUE/$JOB_ID/${node%%.*}/ready" ]] || missing+=("${node%%.*}")
  done
  ((${#missing[@]} == 0)) && break
  if ((SECONDS >= deadline)); then
    echo "worker managers not ready on all allocated nodes: ${missing[*]}" >&2
    exit 1
  fi
  sleep 2
done
echo "worker managers ready on all nodes: ${ALLOCATED_NODES[*]}"

python "$REPO/scripts/maintenance/migrations/migrate_preflight_campaign.py" \
  --campaign-id "$CAMPAIGN_ID" \
  --campaign-spec "$REPO/$SPEC" \
  --expected-fingerprint aa1a343608c41a8b9a596eb410c1648bbd22d6bf8484186ce02fc35d7c80e4ba \
  --workers "$WORKERS" \
  --worker-cpus 47 \
  --worker-gpu-type gh200 \
  --worker-gpus-per-node 1 \
  --fast-reports

resume_args=()
[[ -s "$OUT_DIR/campaign.json" ]] && resume_args+=(--resume)
set +e
timeout --signal=TERM --kill-after=2m 20700s \
  python "$REPO/experiments/campaigns/execution.py" \
  --campaign-spec "$REPO/$SPEC" \
  "${resume_args[@]}" \
  --workers "$WORKERS" \
  --n-envs 64 \
  --worker-nodes "${SLURM_JOB_NODELIST:?}" \
  --worker-cpus 47 \
  --worker-gpu-type gh200 \
  --worker-gpus-per-node 1 \
  --fast-reports \
  2>&1 | tee -a "$OUT_DIR/logs/controller.log"
controller_status=${PIPESTATUS[0]}
set -e
exit "$controller_status"
