#!/usr/bin/env bash
# Prepare and run the acknowledged exploratory matrix in one two-node job.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/18_variant_seed42_20m_purelegacy_shared_bundle_n64_20261014.yaml}"
CAMPAIGN_ID="ladder18_seed42_20m_purelegacy_shared_bundle_n64_20261014"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
PREFLIGHT="$OUT_DIR/reports/preflight_complete.json"
DONE_MARKER="$OUT_DIR/allocation_controller.exit"

# A resumed campaign must run under the source revision recorded in its
# manifest. Use the shared pinned worktree when the active checkout has moved.
MANIFEST="$OUT_DIR/campaign.json"
if [[ -s "$MANIFEST" ]]; then
  expected_sha="$(python3 - "$MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("git_sha", ""))
PY
)"
  current_sha="$(git -C "$REPO" rev-parse HEAD)"
  if [[ -n "$expected_sha" && "$expected_sha" != "$current_sha" ]]; then
    PINNED_REPO="${CAMPAIGN_PINNED_SOURCE_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis_pinned_937bae}"
    if [[ ! -d "$PINNED_REPO" || "$(git -C "$PINNED_REPO" rev-parse HEAD)" != "$expected_sha" ]]; then
      echo "campaign source mismatch: expected=$expected_sha current=$current_sha; no matching pinned checkout" >&2
      exit 2
    fi
    echo "resuming campaign with pinned source $expected_sha at $PINNED_REPO"
    REPO="$PINNED_REPO"
    OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
    PREFLIGHT="$OUT_DIR/reports/preflight_complete.json"
    DONE_MARKER="$OUT_DIR/allocation_controller.exit"
  fi
fi

cd "$REPO"

if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
  while [[ ! -e "$DONE_MARKER" ]]; do sleep 10; done
  exit 0
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/reports"
rm -f "$DONE_MARKER"
trap 'touch "$DONE_MARKER"' EXIT
source "$REPO/scripts/triton/activate_venv.sh"

if [[ ! -s "$PREFLIGHT" ]]; then
  echo "$(date --iso-8601=seconds) rebuilding/auditing the pure-legacy A0 cache"
  python "$REPO/scripts/matrix/rebuild_prebuilt_baseline_cache.py" \
    --backup-dir "$OUT_DIR/artifacts/original_prebuilt_baseline" \
    --report "$OUT_DIR/reports/baseline_cache_audit.json" \
    2>&1 | tee -a "$OUT_DIR/logs/preflight.log"

  echo "$(date --iso-8601=seconds) auditing all-variant frames/bundle parity"
  python "$REPO/scripts/matrix/audit_real_market_bundle_parity.py" \
    --out "$OUT_DIR/reports/frames_bundle_parity.json" \
    --bundle-cache "$OUT_DIR/artifacts/parity_bundle_cache" \
    2>&1 | tee -a "$OUT_DIR/logs/preflight.log"
  python - "$OUT_DIR/reports/frames_bundle_parity.json" "$PREFLIGHT" <<'PY'
import json, pathlib, sys
source = pathlib.Path(sys.argv[1])
data = json.loads(source.read_text(encoding="utf-8"))
items = data.get("variants", [])
if len(items) != 18 or data.get("summary", {}).get("passed") != 18:
    raise SystemExit("preflight parity did not pass for all 18 variants")
if data.get("summary", {}).get("failed") != 0:
    raise SystemExit("preflight parity reported variant failures")
if any(
    trace.get("max_scalar_delta") != 0
    for item in items
    for trace in item.get("traces", [])
):
    raise SystemExit("preflight parity contains a nonzero trace delta")
pathlib.Path(sys.argv[2]).write_text(
    json.dumps({"status": "passed", "variants": 18}, indent=2) + "\n",
    encoding="utf-8",
)
PY
fi

set +e
timeout --signal=TERM --kill-after=2m 20700s \
  python "$REPO/scripts/matrix/campaign_controller.py" \
  --campaign-spec "$REPO/$SPEC" \
  --resume \
  --workers 18 \
  --n-envs 64 \
  --worker-nodes "${SLURM_JOB_NODELIST:?}" \
  --worker-cpus 5 \
  --worker-gpu-type gh200 \
  --worker-gpus-per-node 1 \
  --fast-reports \
  >> "$OUT_DIR/logs/controller.log" 2>&1 &
main_controller_pid=$!

# The two cache-recovery variants have their own immutable campaign manifest.
# Resume them alongside the primary workers so all 18 variants keep advancing.
RECOVERY_ID="ladder2_cache_recovery_seed42_20m_shared_bundle_n64_20261014"
RECOVERY_DIR="$REPO/outputs/gpu_campaigns/$RECOVERY_ID"
RECOVERY_SPEC="$OUT_DIR/recovery_two_variants.yaml"
recovery_controller_pid=""
if [[ -s "$RECOVERY_DIR/campaign.json" && -s "$RECOVERY_SPEC" ]]; then
  timeout --signal=TERM --kill-after=2m 20700s \
    python "$REPO/scripts/matrix/campaign_controller.py" \
    --campaign-spec "$RECOVERY_SPEC" \
    --resume \
    --workers 2 \
    --n-envs 64 \
    --worker-nodes gpuarm1 \
    --worker-cpus 5 \
    --worker-gpu-type gh200 \
    --worker-gpus-per-node 1 \
    --fast-reports \
    >> "$OUT_DIR/logs/recovery_controller.log" 2>&1 &
  recovery_controller_pid=$!
fi

wait "$main_controller_pid"
main_rc=$?
recovery_rc=0
if [[ -n "$recovery_controller_pid" ]]; then
  wait "$recovery_controller_pid"
  recovery_rc=$?
fi
set -e

if (( main_rc != 0 )); then
  exit "$main_rc"
fi
exit "$recovery_rc"
