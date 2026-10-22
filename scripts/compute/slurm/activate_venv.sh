#!/usr/bin/env bash
# Source the arch-matching project venv.
#
# Usage (inside a GPU shell or training script):
#   source scripts/compute/slurm/activate_venv.sh
#
# aarch64 (GH200) -> $REPO/.venv
# x86_64 MI210 (gpu-amd) -> ${MI210_VENV:-/scratch/work/nguyenl37/.venv-mi210}
# other x86_64 (H100/H200, local, CI) -> ${X86_VENV:-$REPO/.venv-x86}
#
# Do not source the wrong-arch venv on login (Exec format error).
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
machine="$(uname -m)"

if [[ "$machine" == "aarch64" ]]; then
  venv="$REPO/.venv"
elif [[ "$machine" == "x86_64" ]]; then
  if [[ "${SLURM_JOB_PARTITION:-}" == "gpu-amd" ]]; then
    venv="${MI210_VENV:-/scratch/work/nguyenl37/.venv-mi210}"
  else
    venv="${X86_VENV:-$REPO/.venv-x86}"
  fi
else
  echo "ERROR: unsupported architecture: $machine" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -x "$venv/bin/python" ]]; then
  if [[ "$machine" == "x86_64" ]]; then
    if [[ "${SLURM_JOB_PARTITION:-}" == "gpu-amd" ]]; then
      echo "ERROR: MI210 ROCm venv not found at $venv — run: tools/setup-venv-mi210 on the MI210 node" >&2
    else
      echo "ERROR: x86 venv not found at $venv — run: tools/setup-venv-x86" >&2
    fi
  else
    echo "ERROR: aarch64 venv not found at $venv — run: uv sync (on aarch64)" >&2
  fi
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$venv/bin/activate"

# A resumed campaign keeps the recorded Git identity only while its own source
# hashes still match. This lets documentation-only commits coexist with an
# active campaign without weakening the controller's source-hash fingerprint.
if [[ -n "${CAMPAIGN_ID:-}" && -z "${CAMPAIGN_RESUME_FINGERPRINT_OVERRIDE_REASON:-}" ]]; then
  # shellcheck source=/dev/null
  source "$REPO/scripts/compute/slurm/campaign_git_identity.sh"
fi
