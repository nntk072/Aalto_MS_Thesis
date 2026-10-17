#!/usr/bin/env bash
set -u

ROOT=/scratch/work/nguyenl37/Aalto_MS_Thesis
RUN_DIR="$ROOT/outputs/gpu_ci_gates/observation_schema_gate_job21012389"
VENV_X86="$ROOT/.venv-x86"
VENV_MI210=/scratch/work/nguyenl37/.venv-mi210
LOG="$RUN_DIR/validation.log"

cd "$ROOT"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "Observation schema gate start: $(date --iso-8601=seconds)"
echo "host=$(hostname -f) arch=$(uname -m) job=${SLURM_JOB_ID:-unset} step=${SLURM_STEP_ID:-unset}"
echo "repo=$ROOT head=$(git -C "$ROOT" rev-parse HEAD)"
echo "git_status_start"
git status --short
echo "tracked_worktree_diff_sha256=$(git diff --binary HEAD | sha256sum | cut -d' ' -f1)"
echo "untracked_file_hashes_start"
git ls-files --others --exclude-standard -z | xargs -0 -r sha256sum
echo "untracked_file_hashes_end"
for executable in \
  "$VENV_X86/bin/ruff" \
  "$VENV_X86/bin/mypy" \
  "$VENV_MI210/bin/python"; do
  if [[ ! -x "$executable" ]]; then
    echo "PREFLIGHT_FAIL missing executable: $executable"
    exit 2
  fi
done

failures=0
run_gate() {
  local name="$1"
  shift
  echo "GATE_START $name $(date --iso-8601=seconds)"
  "$@"
  local rc=$?
  echo "GATE_RESULT $name exit=$rc $(date --iso-8601=seconds)"
  if (( rc != 0 )); then
    failures=$((failures + 1))
  fi
  return 0
}

run_gate full_tree_ruff_format "$VENV_X86/bin/ruff" format --check .
run_gate full_tree_ruff_lint "$VENV_X86/bin/ruff" check .
run_gate full_tree_mypy "$VENV_X86/bin/mypy" .
run_gate mi210_rocm_smoke "$VENV_MI210/bin/python" -c '
import torch

assert torch.cuda.is_available(), "ROCm device is unavailable"
name = torch.cuda.get_device_name(0)
assert "MI210" in name, f"expected MI210, got {name}"
left = torch.randn((256, 256), device="cuda")
right = torch.randn((256, 256), device="cuda")
result = left @ right
torch.cuda.synchronize()
assert torch.isfinite(result).all().item(), "MI210 matmul returned non-finite values"
print(f"ROCm smoke passed: {name}; result_shape={tuple(result.shape)}")
'
run_gate full_pytest "$VENV_MI210/bin/python" -m pytest tests/ -q

echo "Observation schema gate end: $(date --iso-8601=seconds)"
echo "CI gate failures=$failures"
if (( failures != 0 )); then
  exit 1
fi
