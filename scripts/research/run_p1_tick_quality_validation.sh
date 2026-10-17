#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/scratch/work/nguyenl37/Aalto_MS_Thesis
RUN_DIR="$ROOT/outputs/gpu_ci_gates/p1_tick_quality_job${SLURM_JOB_ID:-manual}"
VENV_X86="$ROOT/.venv-x86"
VENV_MI210=/scratch/work/nguyenl37/.venv-mi210
REPORT="$ROOT/docs/thesis/g1_tick_quality_current_recovery_2026-10-17_job${SLURM_JOB_ID:-manual}.json"
LOG="$RUN_DIR/validation.log"

cd "$ROOT"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "P1 tick-integrity/quality audit run start: $(date --iso-8601=seconds)"
echo "host=$(hostname -f) arch=$(uname -m) job=${SLURM_JOB_ID:-unset} step=${SLURM_STEP_ID:-unset}"
echo "repo=$ROOT head=$(git -C "$ROOT" rev-parse HEAD)"
echo "report=$REPORT"

for executable in "$VENV_X86/bin/ruff" "$VENV_MI210/bin/python"; do
  if [[ ! -x "$executable" ]]; then
    echo "PREFLIGHT_FAIL missing executable: $executable"
    exit 2
  fi
done
if [[ -e "$REPORT" ]]; then
  echo "PREFLIGHT_FAIL refusing to overwrite immutable report: $REPORT"
  exit 2
fi

"$VENV_X86/bin/ruff" format --check docs/thesis/audit_tick_quality_current_campaign_2026-10-17.py
"$VENV_X86/bin/ruff" check docs/thesis/audit_tick_quality_current_campaign_2026-10-17.py
"$VENV_X86/bin/ruff" format --check .
"$VENV_X86/bin/ruff" check .
"$VENV_X86/bin/mypy" .
"$VENV_MI210/bin/python" -m pytest tests/ -q
"$VENV_MI210/bin/python" docs/thesis/audit_tick_quality_current_campaign_2026-10-17.py \
  --out "$REPORT"
echo "P1 tick-integrity/quality audit run complete: $(date --iso-8601=seconds)"
