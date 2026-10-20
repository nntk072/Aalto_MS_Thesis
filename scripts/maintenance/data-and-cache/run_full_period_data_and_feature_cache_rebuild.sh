#!/usr/bin/env bash
# Refresh every configured full-period bar/tick cache and all four named feature profiles.
set -euo pipefail
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
source scripts/compute/slurm/activate_venv.sh

python scripts/maintenance/data-and-cache/rebuild_full_period_data_cache.py
python scripts/maintenance/campaigns-or-cache/rebuild_named_prebuilt_feature_caches.py \
  --backup-dir data/prebuilt_feature_cache/backups/pre-18-profile-rebuild_20261017_job21020179_rerun1 \
  --report docs/thesis/evidence/replacement_data_20261017/named_feature_cache_rebuild_18variant_20261017_job21020179_rerun1.json
