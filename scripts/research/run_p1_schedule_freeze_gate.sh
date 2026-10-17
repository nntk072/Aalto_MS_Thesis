#!/usr/bin/env bash
# Rebuild replacement-data tick coverage and its observed-schedule fingerprint
# on the allocated GH200 node, then run repository code and test checks.
set -u
cd /scratch/work/nguyenl37/Aalto_MS_Thesis || exit 2
source scripts/triton/activate_venv.sh
set +e

validate_only=false
if [[ "${1:-}" == "--validate-only" ]]; then
    validate_only=true
fi

ruff format \
    quant_rl/train/train_rl.py \
    tests/test_envs/test_eligibility_wiring.py \
    scripts/research/fingerprint_tick_decision_schedule.py
format_apply_rc=$?

coverage_out=docs/thesis/evidence/replacement_data_20261017/tick_execution_coverage_schedule_freeze.json
fingerprint_out=docs/thesis/evidence/replacement_data_20261017/decision_schedule_fingerprint.json
if [[ "$validate_only" == true ]]; then
    coverage_out=docs/thesis/evidence/replacement_data_20261017/tick_execution_coverage_schedule_freeze_final.json
    fingerprint_out=docs/thesis/evidence/replacement_data_20261017/decision_schedule_fingerprint_final.json
fi

python -m pytest tests/test_envs/test_eligibility_wiring.py -q
pytest_rc=$?

python scripts/research/audit_tick_execution_coverage.py --out "$coverage_out"
coverage_rc=$?

if [[ $coverage_rc -eq 0 ]]; then
    python scripts/research/fingerprint_tick_decision_schedule.py \
        --out "$fingerprint_out" \
        --coverage-report "$coverage_out"
    fingerprint_rc=$?
else
    fingerprint_rc=125
fi

ruff format --check .
format_rc=$?
ruff check .
lint_rc=$?
mypy .
mypy_rc=$?
python -m pytest tests/ -q
full_pytest_rc=$?

printf '\nCHECK_SUMMARY format_apply=%s pytest=%s coverage=%s fingerprint=%s ruff_format=%s ruff_lint=%s mypy=%s full_pytest=%s\n' \
    "$format_apply_rc" "$pytest_rc" "$coverage_rc" "$fingerprint_rc" "$format_rc" "$lint_rc" "$mypy_rc" "$full_pytest_rc"

if [[ $format_apply_rc -eq 0 && $pytest_rc -eq 0 && $coverage_rc -eq 0 && $fingerprint_rc -eq 0 \
    && $format_rc -eq 0 && $lint_rc -eq 0 && $mypy_rc -eq 0 && $full_pytest_rc -eq 0 ]]; then
    exit 0
fi
exit 1
