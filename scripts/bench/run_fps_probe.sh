#!/usr/bin/env bash
# Run the production-config PPO throughput probe on a GPU node.
#
# Usage (inside an existing GPU allocation):
#   bash scripts/bench/run_fps_probe.sh [TOTAL_TIMESTEPS]
#
# Reuses the already-built shared bundle so the probe does not need to reload
# the full train-split frames (~473 GiB peak).
set -uo pipefail
cd "$(dirname "$0")/../.."

TOTAL_TIMESTEPS="${1:-100000}"
BUNDLE="${BUNDLE_DIR:-cache/shared_env/025309d67bc1715c16603a45c41ddbb95d18f0b679ff53abd31ee93cdd7dcadd}"
OUT="outputs/fps_production_n64_$(hostname).json"

source scripts/triton/activate_venv.sh
mkdir -p outputs

echo "HOST=$(hostname)"
.venv-x86/bin/python - <<'PY'
import torch

print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

.venv-x86/bin/python scripts/bench/measure_ppo_throughput.py \
    --device cuda \
    --n-envs 64 \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --bundle-dir "$BUNDLE" \
    --output "$OUT"
echo "EXIT=$?"
