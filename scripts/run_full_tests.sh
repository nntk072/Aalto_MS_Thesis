#!/bin/bash
# Full CI test bar on the GPU node (ruff + mypy + pytest tests/ -v).
# Run after srun (see scripts/run_full_tests_tmux.sh). Do not scancel.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
mkdir -p outputs

arch="$(uname -m)"
if [[ "$arch" == "aarch64" ]]; then
  # GH200: project .venv is aarch64.
  # shellcheck source=/dev/null
  source "$REPO/.venv/bin/activate"
elif [[ -n "${X86_VENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${X86_VENV}/bin/activate"
else
  echo "ERROR: x86 node — set X86_VENV to an x86 venv; do not source GH200 .venv" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
python -c "import torch; print('arch', '$arch', 'CUDA', torch.cuda.is_available())"

python -m ruff format --check .
python -m ruff check .
python -m mypy .
python -m pytest tests/ -v

echo "=== full tests finished ==="
