#!/usr/bin/env bash
set -u

ROOT=/scratch/work/nguyenl37/Aalto_MS_Thesis
RUN_DIR="$ROOT/outputs/gpu_ci_gates/p1_tick_fold_validation_job21012389"
VENV_X86="$ROOT/.venv-x86"
VENV_MI210=/scratch/work/nguyenl37/.venv-mi210
LOG="$RUN_DIR/validation.log"

cd "$ROOT"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "P1 tick/fold validation start: $(date --iso-8601=seconds)"
echo "host=$(hostname -f) arch=$(uname -m) job=${SLURM_JOB_ID:-unset} step=${SLURM_STEP_ID:-unset}"
echo "repo=$ROOT head=$(git -C "$ROOT" rev-parse HEAD)"
for executable in "$VENV_X86/bin/ruff" "$VENV_X86/bin/mypy" "$VENV_MI210/bin/python"; do
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

run_gate ruff_format_apply "$VENV_X86/bin/ruff" format \
  quant_rl/data/ticks.py quant_rl/data/pipeline.py tests/test_ticks.py \
  docs/thesis/audit_real_market_fold_fit_scope_full_train_2026-10-17.py
run_gate ruff_format_verify "$VENV_X86/bin/ruff" format --check \
  quant_rl/data/ticks.py quant_rl/data/pipeline.py tests/test_ticks.py \
  docs/thesis/audit_real_market_fold_fit_scope_full_train_2026-10-17.py
run_gate ruff_lint "$VENV_X86/bin/ruff" check \
  quant_rl/data/ticks.py quant_rl/data/pipeline.py tests/test_ticks.py \
  docs/thesis/audit_real_market_fold_fit_scope_full_train_2026-10-17.py
run_gate full_tree_mypy "$VENV_X86/bin/mypy" .
run_gate focused_tick_tests "$VENV_MI210/bin/python" -m pytest tests/test_ticks.py -q
run_gate full_training_interval_fold_audit "$VENV_MI210/bin/python" \
  docs/thesis/audit_real_market_fold_fit_scope_full_train_2026-10-17.py

echo "P1 tick/fold validation end: $(date --iso-8601=seconds)"
echo "CI gate failures=$failures"
if (( failures != 0 )); then
  exit 1
fi
