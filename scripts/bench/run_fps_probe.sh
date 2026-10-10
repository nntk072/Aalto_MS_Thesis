#!/usr/bin/env bash
# Throughput probe on a GPU node, following the production training configuration.
#
# Usage (inside an existing GPU allocation):
#   BUNDLE_DIR=cache/shared_env/<key> VARIANT=ladder_a2_tp bash scripts/bench/run_fps_probe.sh 200000
#
# BUNDLE_DIR is required: the bundle key depends on the variant's features and
# strategy, so a hard-coded path from another configuration would measure the
# wrong environment. Build it once per variant with train_rl, then point here.
set -uo pipefail
cd "$(dirname "$0")/../.."

TOTAL_TIMESTEPS="${1:-100000}"
: "${BUNDLE_DIR:?set BUNDLE_DIR to an existing shared env bundle for this variant}"
VARIANT="${VARIANT:-}"
OUT="${OUT:-outputs/fps_$(hostname)_${VARIANT:-default}.json}"

source scripts/triton/activate_venv.sh
mkdir -p "$(dirname "$OUT")"

echo "HOST=$(hostname) VARIANT=${VARIANT:-default} BUNDLE_DIR=$BUNDLE_DIR"
VARIANT_ARGS=()
[[ -n "$VARIANT" ]] && VARIANT_ARGS=(--variant "$VARIANT")

.venv-x86/bin/python scripts/bench/measure_ppo_throughput.py \
    --device cuda \
    --n-envs 64 \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --bundle-dir "$BUNDLE_DIR" \
    "${VARIANT_ARGS[@]}" \
    --output "$OUT"
echo "EXIT=$?"
