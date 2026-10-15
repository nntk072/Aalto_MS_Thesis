#!/usr/bin/env bash
# Required repository checks for the CPU-mask campaign fix; run on a GPU node.
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis-main}"
OUT_DIR="${CAMPAIGN_OUT_DIR:?CAMPAIGN_OUT_DIR is required}"
LOG="$OUT_DIR/logs/gpu_ci_gate.log"
cd "$REPO"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "GPU CI gate start host=$(hostname -f) arch=$(uname -m) date=$(date --iso-8601=seconds)"
source "$REPO/scripts/triton/activate_venv.sh"
PYTHON="$VIRTUAL_ENV/bin/python"
RUFF="$VIRTUAL_ENV/bin/ruff"
PYTEST="$VIRTUAL_ENV/bin/pytest"
for shell_script in \
  "$REPO/scripts/matrix/run_campaign_18way_cpumask_allocation.sh" \
  "$REPO/scripts/matrix/run_campaign_18way_cpumask_supervisor.sh" \
  "$REPO/scripts/matrix/validate_cpu_mask_campaign_gpu.sh"; do
  bash -n "$shell_script"
done
"$PYTHON" -m py_compile \
  scripts/matrix/campaign_controller.py \
  scripts/matrix/verify_worker_cpu_affinity.py
"$RUFF" format --check .
"$RUFF" check .

ENV_DIR="$VIRTUAL_ENV"
MYPY_STAGE="$(mktemp -d "/tmp/triton-mypy-${USER}-cpumask-XXXXXX")"
cleanup() {
  rm -rf -- "$MYPY_STAGE"
}
trap cleanup EXIT
mkdir -p "$MYPY_STAGE/sitepkgs" "$MYPY_STAGE/cache"
BASE="$(dirname "$(dirname "$(readlink -f "$ENV_DIR/bin/python")")")"
SITE="$ENV_DIR/lib/python3.12/site-packages"
[[ -d "$SITE/mypy" && -x "$BASE/bin/python3.12" ]]
cp -a "$BASE" "$MYPY_STAGE/pythonbase"
cp -a "$SITE/mypy" "$SITE/mypyc" "$SITE/librt" "$SITE/pathspec" \
  "$SITE/packaging" "$SITE/mypy_extensions.py" "$SITE/typing_extensions.py" \
  "$MYPY_STAGE/sitepkgs/"
find "$SITE" -maxdepth 1 -name '*__mypyc*.so' -exec cp -a {} "$MYPY_STAGE/sitepkgs/" \;
export PYTHONPATH="$MYPY_STAGE/sitepkgs:$SITE"
"$MYPY_STAGE/pythonbase/bin/python3.12" -m mypy . \
  --cache-dir "$MYPY_STAGE/cache" --no-color-output --show-error-codes

"$PYTEST" tests/test_campaign_artifacts.py tests/test_campaign_checkpoint_migration.py -q
echo "GPU CI gate passed host=$(hostname -f) arch=$(uname -m)"
