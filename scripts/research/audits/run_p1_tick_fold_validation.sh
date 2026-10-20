#!/usr/bin/env bash
set -u

ROOT=/scratch/work/nguyenl37/Aalto_MS_Thesis
RUN_DIR="$ROOT/outputs/gpu_ci_gates/p1_tick_fold_validation_job${SLURM_JOB_ID:-manual}"
VENV_X86="$ROOT/.venv-x86"
VENV_MI210=/scratch/work/nguyenl37/.venv-mi210
LOG="$RUN_DIR/validation.log"
AUDIT_REPORT="${P1_FOLD_AUDIT_OUTPUT:-$ROOT/docs/thesis/g1_real_market_fold_fit_full_train_post_observation_schema_2026-10-17.json}"
TICK_AUDIT_REPORT="${P1_TICK_AUDIT_OUTPUT:-$ROOT/docs/thesis/g1_tick_quality_current_recovery_2026-10-17.json}"

cd "$ROOT"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "P1 tick/fold validation start: $(date --iso-8601=seconds)"
echo "host=$(hostname -f) arch=$(uname -m) job=${SLURM_JOB_ID:-unset} step=${SLURM_STEP_ID:-unset}"
echo "repo=$ROOT head=$(git -C "$ROOT" rev-parse HEAD)"
echo "pre_gate_tracked_worktree_diff_sha256=$(git -C "$ROOT" diff --binary HEAD | sha256sum | cut -d' ' -f1)"
echo "audit_report=$AUDIT_REPORT"
echo "tick_audit_report=$TICK_AUDIT_REPORT"
for report in "$AUDIT_REPORT" "$TICK_AUDIT_REPORT"; do
  if [[ -e "$report" ]]; then
    echo "PREFLIGHT_FAIL refusing to overwrite audit report: $report"
    exit 2
  fi
done
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

run_gate touched_source_format_apply "$VENV_X86/bin/ruff" format \
  docs/thesis/audit_tick_quality_current_campaign_2026-10-17.py
echo "validated_source_tracked_worktree_diff_sha256=$(git -C "$ROOT" diff --binary HEAD | sha256sum | cut -d' ' -f1)"
run_gate full_tree_ruff_format_verify "$VENV_X86/bin/ruff" format --check .
run_gate full_tree_ruff_lint "$VENV_X86/bin/ruff" check .
run_gate full_tree_mypy "$VENV_X86/bin/mypy" .
run_gate campaign_shell_syntax bash -c \
  'for file in scripts/compute/slurm/campaigns/run_18variant_matrix.sh scripts/compute/slurm/campaigns/run_campaign_18way_cpumask_allocation.sh scripts/compute/slurm/campaigns/run_campaign_18way_cpumask_supervisor.sh scripts/compute/slurm/campaigns/validate_cpu_mask_campaign_gpu.sh scripts/compute/slurm/campaign_srun_gh200_guard.sh scripts/compute/slurm/campaign_allocation_wrapper.sh; do bash -n "$file" || exit; done'
run_gate matrix_launcher_no_scancel bash -c \
  '! rg -n "^[[:space:]]*scancel[[:space:]]|[;&|][[:space:]]*scancel[[:space:]]" scripts/compute/slurm/campaigns/run_18variant_matrix.sh'
if (( failures != 0 )); then
  echo "DATA_AUDITS_SKIPPED software gate failed=$failures"
  exit 1
fi
run_gate focused_tick_tests "$VENV_MI210/bin/python" -m pytest tests/test_ticks.py -q
if (( failures != 0 )); then
  echo "DATA_AUDITS_SKIPPED focused tick tests failed"
  exit 1
fi
run_gate raw_tick_quality_audit "$VENV_MI210/bin/python" \
  docs/thesis/audit_tick_quality_current_campaign_2026-10-17.py \
  --out "$TICK_AUDIT_REPORT"
run_gate full_training_interval_fold_audit "$VENV_MI210/bin/python" \
  docs/thesis/audit_real_market_fold_fit_scope_full_train_2026-10-17.py \
  --out "$AUDIT_REPORT"

echo "P1 tick/fold validation end: $(date --iso-8601=seconds)"
echo "CI gate failures=$failures"
if (( failures != 0 )); then
  exit 1
fi
