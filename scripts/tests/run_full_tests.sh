#!/bin/bash
# Full CI test bar on the GPU node (ruff + mypy + pytest tests/ -v).
# Run after srun (see scripts/run_full_tests_tmux.sh). Do not scancel.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
mkdir -p outputs

arch="$(uname -m)"
# shellcheck source=/dev/null
source "$REPO/scripts/triton/activate_venv.sh"

export PYTHONUNBUFFERED=1
python -c "import torch; print('arch', '$arch', 'CUDA', torch.cuda.is_available())"

python -m ruff format --check .
python -m ruff check .
python -m mypy .
python -m pytest tests/ -v

echo "=== full tests finished ==="
