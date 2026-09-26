#!/usr/bin/env bash
# Create / refresh the arch-matching project virtualenv.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
machine="$(uname -m)"

case "$machine" in
  aarch64)
    cd "$REPO"
    echo "Syncing $REPO/.venv (aarch64) ..."
    uv sync --extra dev
    echo "Done. Activate with: source .venv/bin/activate"
    ;;
  x86_64)
    exec "$REPO/scripts/setup_venv_x86.sh"
    ;;
  *)
    echo "ERROR: unsupported architecture: $machine" >&2
    exit 1
    ;;
esac
