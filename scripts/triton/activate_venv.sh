#!/usr/bin/env bash
# Source the arch-matching project venv.
#
# Usage (inside a GPU shell or training script):
#   source scripts/triton/activate_venv.sh
#
# aarch64 (GH200) -> $REPO/.venv
# x86_64 (H200, local, CI) -> ${X86_VENV:-$REPO/.venv-x86}
#
# Do not source the wrong-arch venv on login (Exec format error).
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
machine="$(uname -m)"

if [[ "$machine" == "aarch64" ]]; then
  venv="$REPO/.venv"
elif [[ "$machine" == "x86_64" ]]; then
  venv="${X86_VENV:-$REPO/.venv-x86}"
else
  echo "ERROR: unsupported architecture: $machine" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -x "$venv/bin/python" ]]; then
  if [[ "$machine" == "x86_64" ]]; then
    echo "ERROR: x86 venv not found at $venv — run: scripts/setup_venv_x86.sh" >&2
  else
    echo "ERROR: aarch64 venv not found at $venv — run: uv sync (on aarch64)" >&2
  fi
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$venv/bin/activate"
