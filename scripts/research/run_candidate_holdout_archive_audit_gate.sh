#!/usr/bin/env bash
# Scan pre-intake archive text for holdout-period references, then gate the tree.
set -u
cd /scratch/work/nguyenl37/Aalto_MS_Thesis || exit 2
source scripts/triton/activate_venv.sh
set +e

audit_rc=0
ruff format scripts/research/audit_candidate_holdout_archive_exposure.py
format_apply_rc=$?
python scripts/research/audit_candidate_holdout_archive_exposure.py \
    --out docs/thesis/evidence/replacement_data_20261017/candidate_holdout_archive_exposure_20261017_triaged.json \
    || audit_rc=$?

ruff format --check .
format_rc=$?
ruff check .
lint_rc=$?
mypy .
mypy_rc=$?
python -m pytest tests/ -q
pytest_rc=$?

printf '\nCHECK_SUMMARY format_apply=%s archive_audit=%s ruff_format=%s ruff_lint=%s mypy=%s full_pytest=%s\n' \
    "$format_apply_rc" "$audit_rc" "$format_rc" "$lint_rc" "$mypy_rc" "$pytest_rc"

if [[ $format_apply_rc -eq 0 && $audit_rc -eq 0 && $format_rc -eq 0 && $lint_rc -eq 0 \
    && $mypy_rc -eq 0 && $pytest_rc -eq 0 ]]; then
    exit 0
fi
exit 1
