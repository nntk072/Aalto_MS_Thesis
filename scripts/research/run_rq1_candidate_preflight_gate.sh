#!/usr/bin/env bash
# Validate the provisional RQ1 config contrast and the complete current tree.
set -u
cd /scratch/work/nguyenl37/Aalto_MS_Thesis || exit 2
source scripts/triton/activate_venv.sh
set +e

audit_rc=0
python scripts/research/audit_rq1_candidate_config.py \
    --out docs/thesis/evidence/rq1_gate_candidate_config_preflight_20261017.json \
    || audit_rc=$?
format_apply_rc=0
ruff format . || format_apply_rc=$?
format_rc=0
ruff format --check . || format_rc=$?
lint_rc=0
ruff check . || lint_rc=$?
mypy_rc=0
mypy . || mypy_rc=$?
pytest_rc=0
python -m pytest tests/ -q || pytest_rc=$?

printf '\nCHECK_SUMMARY rq1_config_audit=%s format_apply=%s format_check=%s lint=%s mypy=%s full_pytest=%s\n' \
    "$audit_rc" "$format_apply_rc" "$format_rc" "$lint_rc" "$mypy_rc" "$pytest_rc"

if [[ $audit_rc -eq 0 && $format_apply_rc -eq 0 && $format_rc -eq 0 && $lint_rc -eq 0 \
    && $mypy_rc -eq 0 && $pytest_rc -eq 0 ]]; then
    exit 0
fi
exit 1
