#!/usr/bin/env bash
# Create a separate locked-dependency test environment for Triton's MI210 nodes.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
MI210_ENV="${TRITON_MI210_ENV:-/scratch/work/nguyenl37/.venv-mi210}"
cd "$REPO"

machine="$(uname -m)"
if [[ "$machine" != "x86_64" ]]; then
  echo "ERROR: MI210 setup requires x86_64 (got $machine)" >&2
  exit 1
fi

rocm_version="$(cat /opt/rocm/.info/version 2>/dev/null || true)"
if [[ "$rocm_version" != 6.3.* ]]; then
  echo "ERROR: expected Triton ROCm 6.3.x for the pinned MI210 build (got ${rocm_version:-unknown})" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required on the MI210 compute node" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="$MI210_ENV"
uv python install 3.12
if [[ -x "$MI210_ENV/bin/python" ]]; then
  "$MI210_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
elif [[ -e "$MI210_ENV" ]]; then
  echo "ERROR: $MI210_ENV exists without a Python interpreter; inspect it before replacing it" >&2
  exit 1
else
  uv venv --python 3.12 "$MI210_ENV"
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/aalto-mi210-env.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
uv export --locked --extra dev --no-emit-project --no-emit-package torch \
  -o "$tmp_dir/locked-requirements.txt"

# The repository lock selects CUDA 13.0. Retain its non-accelerator packages,
# then install the ROCm PyTorch wheel separately; do not install CUDA libraries.
awk '
/^[[:alnum:]][[:alnum:]_.-]*(\[[^]]+\])?([<=>!~;[:space:]]|$)/ {
  name=$0
  sub(/\[[^]]*\]/, "", name)
  sub(/[<=>!~;[:space:]].*/, "", name)
  skip=(name == "torch" || name == "torchvision" || name == "triton" ||
        name == "pytorch-triton" || name == "pytorch-triton-rocm" ||
        name ~ /^(nvidia-|cuda-)/)
}
!skip { print }
' "$tmp_dir/locked-requirements.txt" > "$tmp_dir/rocm-requirements.txt"

uv pip sync --python "$MI210_ENV/bin/python" "$tmp_dir/rocm-requirements.txt"
uv pip install --python "$MI210_ENV/bin/python" torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/rocm6.3
uv pip install --python "$MI210_ENV/bin/python" -e "$REPO" --no-deps

"$MI210_ENV/bin/python" - <<'PY'
import torch

assert torch.__version__.startswith("2.7.1+rocm6.3"), torch.__version__
assert torch.version.hip is not None, torch.version.hip
assert torch.cuda.is_available()
assert "MI210" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
left = torch.randn((256, 256), device="cuda")
right = left @ left
torch.cuda.synchronize()
assert torch.isfinite(right).all()
print(
    f"MI210 ready: torch={torch.__version__}, hip={torch.version.hip}, "
    f"device={torch.cuda.get_device_name(0)}"
)
PY
