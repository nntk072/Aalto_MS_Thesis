#!/usr/bin/env bash
# Validate the 18-variant mapping to four full-period prebuilt feature profiles.
set -u
cd /scratch/work/nguyenl37/Aalto_MS_Thesis || exit 2
source scripts/triton/activate_venv.sh
set +e

ruff format --check .
format_rc=$?
ruff check .
lint_rc=$?
mypy .
mypy_rc=$?
python -m pytest tests/ -q
pytest_rc=$?
python scripts/matrix/rebuild_named_prebuilt_feature_caches.py --plan-only \
    > outputs/data_refresh_20261017/cache_profile_18variant_plan_20261017_retry1.json
plan_rc=$?

printf '\nCHECK_SUMMARY format=%s lint=%s mypy=%s full_pytest=%s cache_plan=%s\n' \
    "$format_rc" "$lint_rc" "$mypy_rc" "$pytest_rc" "$plan_rc"

if [[ $format_rc -eq 0 && $lint_rc -eq 0 && $mypy_rc -eq 0 \
    && $pytest_rc -eq 0 && $plan_rc -eq 0 ]]; then
    exit 0
fi
exit 1
