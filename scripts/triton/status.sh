#!/usr/bin/env bash
# Inspect Triton GPU / Slurm / tmux state (login-node safe — no srun).
# Usage: bash scripts/triton/status.sh
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-quant-rl-train}"

echo "=== user / host ==="
echo "USER=${USER:-unknown}  HOST=$(hostname)  ARCH=$(uname -m)"
echo "REPO=$REPO"
echo

echo "=== squeue (this user) ==="
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER}" -o '%.18i %.12P %.10T %.10M %.6D %R %b %m' 2>/dev/null || squeue -u "${USER}"
else
  echo "(squeue not available)"
fi
echo

echo "=== tmux sessions ==="
if command -v tmux >/dev/null 2>&1; then
  tmux ls 2>/dev/null || echo "(no tmux sessions)"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo
    echo "=== windows in $SESSION ==="
    tmux list-windows -t "$SESSION" 2>/dev/null || true
  fi
else
  echo "(tmux not available)"
fi
echo

echo "=== hint ==="
echo "Reuse an existing GPU srun/tmux if present. Do not scancel finished shells."
echo "Only allocate via: bash scripts/triton/attach_or_alloc.sh"
