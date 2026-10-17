#!/usr/bin/env bash
# Run the repository CI gates in one existing Slurm GPU allocation.
set -uo pipefail

ROOT=/scratch/work/nguyenl37/Aalto_MS_Thesis
cd "$ROOT"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this gate must run inside an existing GPU allocation" >&2
  exit 2
fi

source scripts/triton/activate_venv.sh
# Activation enables errexit. Disable it so a failed gate is recorded and the
# remaining gates, including the full pytest run, still execute in this one
# sustained allocation.
set +e
GATE_DIR="$ROOT/outputs/gpu_ci_gates/current_tree_job${SLURM_JOB_ID}"
mkdir -p "$GATE_DIR"
exec > >(tee -a "$GATE_DIR/validation.log") 2>&1
RUN_DATE="$(date +%Y%m%d)"

printf 'job=%s node=%s arch=%s partition=%s python=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(uname -m)" \
  "${SLURM_JOB_PARTITION:-unknown}" "$(command -v python)"
python --version
status=0
run_gate() {
  label="$1"
  shift
  printf '\n===== %s =====\n' "$label"
  "$@"
  gate_status=$?
  printf 'GATE_EXIT %s %s\n' "$label" "$gate_status"
  if [[ "$gate_status" -ne 0 ]]; then status=1; fi
}

run_gate nvidia_smi nvidia-smi
gpu_smoke() {
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
if not any(family in name for family in ("A100", "H100", "H200", "GH200", "B300")):
    raise SystemExit(f"GPU family is not approved for this gate: {name}")
x = torch.ones((256, 256), device="cuda")
print("torch", torch.__version__, "device", name, "cuda_matmul", float((x @ x)[0, 0].item()))
PY
}
run_gate gpu_smoke gpu_smoke
run_gate ruff_format_full ruff format --check .
run_gate ruff_lint_full ruff check .

# Stage the matching Python runtime and Mypy modules on node-local storage to
# avoid intermittent Lustre import stalls. This still runs the required full
# tree `mypy .` check against this checkout and its installed dependencies.
run_mypy_full() {
  local env_dir stage_dir base_dir site_dir stage_status
  env_dir="$(dirname "$(dirname "$(command -v python)")")"
  stage_dir="$(mktemp -d "/tmp/triton-mypy-${USER}-XXXXXX")" || return
  base_dir="$(dirname "$(dirname "$(readlink -f "$env_dir/bin/python")")")"
  site_dir="$env_dir/lib/python3.12/site-packages"
  mkdir -p "$stage_dir/sitepkgs" "$stage_dir/cache" || {
    rm -rf -- "$stage_dir"
    return 1
  }
  cp -a "$base_dir" "$stage_dir/pythonbase" || {
    rm -rf -- "$stage_dir"
    return 1
  }
  cp -a "$site_dir/mypy" "$site_dir/mypyc" "$site_dir/librt" \
    "$site_dir/pathspec" "$site_dir/packaging" "$site_dir/mypy_extensions.py" \
    "$site_dir/typing_extensions.py" "$stage_dir/sitepkgs/" || {
    rm -rf -- "$stage_dir"
    return 1
  }
  find "$site_dir" -maxdepth 1 -name '*__mypyc*.so' \
    -exec cp -a {} "$stage_dir/sitepkgs/" \;
  printf 'mypy_env=%s pythonbase=%s\n' "$env_dir" "$base_dir"
  PYTHONPATH="$stage_dir/sitepkgs:$site_dir" \
    "$stage_dir/pythonbase/bin/python3.12" -m mypy . \
    --cache-dir "$stage_dir/cache" --no-color-output --show-error-codes
  stage_status=$?
  rm -rf -- "$stage_dir"
  return "$stage_status"
}
run_gate mypy_full run_mypy_full
run_gate pytest_full python -m pytest tests/ -q
run_gate p1_tick_coverage_audit python scripts/research/audit_tick_execution_coverage.py \
  --out "docs/thesis/evidence/p1_tick_execution_coverage_${RUN_DATE}_job${SLURM_JOB_ID}.json" \
  --latency-ms 30 --max-quote-wait-ms 60000
run_gate whitespace git diff --check

printf '\nSOFTWARE_GATE_EXIT %s\n' "$status"
printf 'The tick audit disposition is recorded in its JSON report and is separate from this software-gate exit.\n'
exit "$status"
