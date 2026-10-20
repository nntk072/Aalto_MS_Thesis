#!/usr/bin/env bash
# Transactionally rebuild all feature profiles used by the 18-variant thesis screen.
set -euo pipefail
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
source scripts/compute/slurm/activate_venv.sh
python scripts/maintenance/campaigns-or-cache/rebuild_named_prebuilt_feature_caches.py \
    --backup-dir data/prebuilt_feature_cache/backups/pre-18-profile-rebuild_20261017_job21020179 \
    --report docs/thesis/evidence/replacement_data_20261017/named_feature_cache_rebuild_18variant_20261017_job21020179.json
