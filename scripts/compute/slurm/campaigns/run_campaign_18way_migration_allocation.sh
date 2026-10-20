#!/usr/bin/env bash
# Run the unified 18-variant successor campaign in one existing allocation.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SPEC="${CAMPAIGN_SPEC:-config/campaigns/18_variant_seed42_20m_policy_migration_20261014.yaml}"
CAMPAIGN_ID="ladder18_seed42_20m_policy_migration_20261014"
OUT_DIR="$REPO/outputs/gpu_campaigns/$CAMPAIGN_ID"
CHECKPOINT_MAP="$OUT_DIR/checkpoint_migration_map.json"
PREFLIGHT="$OUT_DIR/reports/preflight_complete.json"
DONE_MARKER="$OUT_DIR/allocation_controller.exit"
PARENT_PARITY_CACHE="$REPO/outputs/gpu_campaigns/ladder18_seed42_20m_purelegacy_shared_bundle_n64_20261014/artifacts/parity_bundle_cache"

cd "$REPO"
if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
  while [[ ! -e "$DONE_MARKER" ]]; do sleep 10; done
  exit 0
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/reports"
rm -f "$DONE_MARKER"
trap 'touch "$DONE_MARKER"' EXIT
source "$REPO/scripts/compute/slurm/activate_venv.sh"

if [[ ! -s "$CHECKPOINT_MAP" ]]; then
  python "$REPO/experiments/campaigns/checkpoints.py" \
    --source-campaign "$REPO/outputs/gpu_campaigns/ladder18_seed42_20m_purelegacy_shared_bundle_n64_20261014" \
    --source-campaign "$REPO/outputs/gpu_campaigns/ladder2_cache_recovery_seed42_20m_shared_bundle_n64_20261014" \
    --variant-file "$REPO/config/campaigns/variant_list.txt" \
    --seeds 42 \
    --timesteps 20000000 \
    --output "$CHECKPOINT_MAP" \
    2>&1 | tee -a "$OUT_DIR/logs/checkpoint_migration.log"
fi

if [[ ! -s "$PREFLIGHT" ]]; then
  if python "$REPO/scripts/research/campaigns/audit_real_market_bundle_parity.py" \
    --out "$OUT_DIR/reports/frames_bundle_parity.json" \
    --bundle-cache "$PARENT_PARITY_CACHE" \
    --obs-mmap-dir "$OUT_DIR/artifacts/parity_obs_mmap" \
    --require-existing-bundles \
    2>&1 | tee -a "$OUT_DIR/logs/preflight.log"; then
    echo "Validated and reused the parent campaign's content-keyed shared bundles."
  else
    echo "Parent bundle reuse validation failed; rebuilding parity bundles in successor output."
    python "$REPO/scripts/research/campaigns/audit_real_market_bundle_parity.py" \
      --out "$OUT_DIR/reports/frames_bundle_parity.json" \
      --bundle-cache "$OUT_DIR/artifacts/parity_bundle_cache" \
      --obs-mmap-dir "$OUT_DIR/artifacts/parity_bundle_cache/obs_mmap" \
      2>&1 | tee -a "$OUT_DIR/logs/preflight.log"
  fi
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

timeout --signal=TERM --kill-after=2m 20700s \
  python "$REPO/experiments/campaigns/execution.py" \
  --campaign-spec "$REPO/$SPEC" \
  --initial-checkpoints "$CHECKPOINT_MAP" \
  --resume \
  --workers 18 \
  --n-envs 64 \
  --worker-nodes "${SLURM_JOB_NODELIST:?}" \
  --worker-cpus 5 \
  --worker-gpu-type gh200 \
  --worker-gpus-per-node 1 \
  --fast-reports \
  2>&1 | tee -a "$OUT_DIR/logs/controller.log"
