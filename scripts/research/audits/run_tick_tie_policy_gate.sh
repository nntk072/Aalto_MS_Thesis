#!/usr/bin/env bash
# Verify the documented equal-millisecond tick rule on full replacement caches.
set -u
cd /scratch/work/nguyenl37/Aalto_MS_Thesis || exit 2
source scripts/compute/slurm/activate_venv.sh
set +e

ruff format \
    market_data/loading/ticks.py \
    tests/test_ticks.py \
    scripts/research/audits/audit_tick_tie_policy.py
format_apply_rc=$?

python -m pytest tests/test_ticks.py tests/test_envs/test_eligibility_wiring.py -q
focused_pytest_rc=$?

python scripts/research/audits/audit_tick_tie_policy.py \
    --coverage-report docs/thesis/evidence/replacement_data_20261017/tick_execution_coverage_schedule_freeze_final.json \
    --prior-tie-report docs/thesis/evidence/replacement_data_20261017/tick_timestamp_ties.json \
    --out docs/thesis/evidence/replacement_data_20261017/tick_timestamp_tie_policy_20261017.json
tie_audit_rc=$?

ruff format --check .
format_check_rc=$?
ruff check .
lint_rc=$?
mypy .
mypy_rc=$?
python -m pytest tests/ -q
full_pytest_rc=$?

printf '\nCHECK_SUMMARY format_apply=%s focused_pytest=%s tie_audit=%s ruff_format=%s ruff_lint=%s mypy=%s full_pytest=%s\n' \
    "$format_apply_rc" "$focused_pytest_rc" "$tie_audit_rc" "$format_check_rc" \
    "$lint_rc" "$mypy_rc" "$full_pytest_rc"

if [[ $format_apply_rc -eq 0 && $focused_pytest_rc -eq 0 && $tie_audit_rc -eq 0 \
    && $format_check_rc -eq 0 && $lint_rc -eq 0 && $mypy_rc -eq 0 \
    && $full_pytest_rc -eq 0 ]]; then
    exit 0
fi
exit 1
