#!/usr/bin/env bash
# Create / refresh the x86_64 virtualenv at .venv-x86 (H200, local, CI login).
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"

machine="$(uname -m)"
if [[ "$machine" != "x86_64" ]]; then
  echo "ERROR: setup_venv_x86.sh must run on x86_64 (got $machine)" >&2
  exit 1
fi

echo "Syncing $REPO/.venv-x86 ..."
UV_PROJECT_ENVIRONMENT="$REPO/.venv-x86" uv sync --extra dev

echo "Done. Activate with:"
echo "  source scripts/triton/activate_venv.sh"
echo "  # or: source .venv-x86/bin/activate"
