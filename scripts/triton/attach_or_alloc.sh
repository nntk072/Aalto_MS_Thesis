#!/usr/bin/env bash
# Reuse an existing GPU tmux/srun shell, or allocate ONE long interactive srun.
#
# GPU priority: H200 first (if usable), else GH200.
# Always 1 GPU. Time: 12h if partition MaxTime allows, else 6h.
#
# Env overrides:
#   PARTITION  GPUS  MEM  CPUS  TIME
#   TMUX_SESSION  TMUX_WINDOW  TRITON_REPO
#   SKIP_PICK=1   # keep PARTITION/GPUS/TIME as given (no auto-pick)
#
# Usage (from login node; agents need user OK before running this for a new alloc):
#   bash scripts/triton/attach_or_alloc.sh
#   MEM=512G CPUS=48 bash scripts/triton/attach_or_alloc.sh
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
SESSION="${TMUX_SESSION:-quant-rl-train}"
WINDOW="${TMUX_WINDOW:-gpu-shell}"
MEM="${MEM:-128G}"
CPUS="${CPUS:-8}"
SKIP_PICK="${SKIP_PICK:-0}"

cd "$REPO" || {
  echo "ERROR: cannot cd $REPO" >&2
  exit 1
}

# --- pick partition / GRES / time from availability (unless overridden) ---
_pick_resources() {
  local idle_h200=0 idle_gh200=0
  # Count idle nodes (STATE column often "idle" / "idle~")
  idle_h200=$(sinfo -p gpu-h200-141g-short -h -t idle -o '%D' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  idle_gh200=$(sinfo -p gpu-grace-h200-141g -h -t idle -o '%D' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  # Also treat lightly mixed H200 short as usable if any node not fully allocated
  local h200_up
  h200_up=$(sinfo -p gpu-h200-141g-short -h -o '%a %D' 2>/dev/null | awk '$1=="up"{s+=$2} END{print s+0}')

  if [[ -n "${PARTITION:-}" && -n "${GPUS:-}" && "${SKIP_PICK}" == "1" ]]; then
    TIME="${TIME:-12:00:00}"
    return 0
  fi

  if [[ "${idle_h200}" -gt 0 || "${h200_up}" -gt 0 ]]; then
    # Prefer H200 when the partition is up and has capacity signal.
    # If fully jammed, user can OVERRIDE with PARTITION=gpu-grace-h200-141g.
    if [[ "${idle_h200}" -gt 0 || "${h200_up}" -gt 0 ]]; then
      PARTITION="${PARTITION:-gpu-h200-141g-short}"
      GPUS="${GPUS:-h200:1}"
    fi
  fi

  if [[ -z "${PARTITION:-}" || -z "${GPUS:-}" ]]; then
    PARTITION="${PARTITION:-gpu-grace-h200-141g}"
    GPUS="${GPUS:-gh200:1}"
  fi

  # Time: 12h if MaxTime >= 12h, else 6h (or leave TIME if set)
  if [[ -z "${TIME:-}" ]]; then
    local max_min
    max_min=$(scontrol show partition "$PARTITION" 2>/dev/null \
      | tr ' ' '\n' | awk -F= '/^MaxTime=/{print $2; exit}')
    # MaxTime forms: 5-00:00:00 | 2-00:00:00 | 12:00:00 | 8:00:00 | UNLIMITED
    local hours=0
    if [[ "$max_min" == "UNLIMITED" || -z "$max_min" ]]; then
      hours=999
    elif [[ "$max_min" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
      hours=$(( ${BASH_REMATCH[1]} * 24 + 10#${BASH_REMATCH[2]} ))
    elif [[ "$max_min" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
      hours=$((10#${BASH_REMATCH[1]}))
    fi
    if [[ "$hours" -ge 12 ]]; then
      TIME="12:00:00"
    else
      TIME="6:00:00"
    fi
  fi
}

echo "=== preflight (no new srun yet) ==="
bash "$REPO/scripts/triton/status.sh" || true
echo
echo "=== availability (H200 preferred, then GH200) ==="
sinfo -p gpu-h200-141g-short,gpu-h200-141g-m,gpu-grace-h200-141g \
  -o '%P %a %D %T %G %l %m %c' 2>/dev/null || true

_pick_resources
PARTITION="${PARTITION:?}"
GPUS="${GPUS:?}"
TIME="${TIME:?}"

echo
echo "Selected: partition=$PARTITION gpus=$GPUS mem=$MEM cpus=$CPUS time=$TIME"
if [[ "$GPUS" == h200:* ]]; then
  echo "NOTE: H200 is x86_64 — do not source the GH200 aarch64 .venv; use an x86 env."
elif [[ "$GPUS" == gh200:* ]]; then
  echo "NOTE: GH200 is aarch64 — source $REPO/.venv on the compute node."
fi

if command -v squeue >/dev/null 2>&1; then
  RUNNING=$(squeue -u "${USER}" -h -t R -o '%i' 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${RUNNING}" -gt 0 ]]; then
    echo
    echo "REUSE: you already have ${RUNNING} running job(s)."
    echo "Attach: tmux attach -t $SESSION"
    echo "Then run training inside the existing GPU bash — do not allocate another srun."
    echo "Do not scancel finished or idle GPU shells unless the user asks."
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux list-windows -t "$SESSION" || true
    fi
    exit 0
  fi
fi

echo
echo "ALLOC: no running job — requesting ONE long interactive allocation (1 GPU):"
echo "  partition=$PARTITION gpus=$GPUS mem=$MEM cpus=$CPUS time=$TIME"
echo "  tmux=$SESSION:$WINDOW"
echo

tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n bootstrap
if ! tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$WINDOW"; then
  tmux new-window -t "$SESSION" -n "$WINDOW"
fi

SRUN_CMD=$(cat <<EOF
cd $REPO && \
srun --partition=$PARTITION --gpus=$GPUS --time=$TIME --mem=$MEM --ntasks=1 --cpus-per-task=$CPUS --pty bash -lc '
  echo "=== allocated on \$(hostname) ==="
  nvidia-smi -L 2>/dev/null || true
  uname -m
  if [[ "\$(uname -m)" == "aarch64" ]]; then
    source $REPO/.venv/bin/activate
  else
    echo "x86 node: activate your x86 venv (not GH200 .venv)"
  fi
  python -c "import torch; print(\"CUDA\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" 2>/dev/null || true
  echo "Leave this shell open after training. Do not exit unless replacing the allocation."
  exec bash
'
EOF
)

PANE_TAIL=$(tmux capture-pane -t "$SESSION:$WINDOW" -p 2>/dev/null | tail -5 || true)
if echo "$PANE_TAIL" | grep -Eqi 'srun:|CUDA True|gpuarm|gh200|h200'; then
  echo "Window $SESSION:$WINDOW already looks GPU-backed — not sending another srun."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux send-keys -t "$SESSION:$WINDOW" C-c 2>/dev/null || true
sleep 0.2
tmux send-keys -t "$SESSION:$WINDOW" "$SRUN_CMD" Enter

echo "Sent srun into tmux $SESSION:$WINDOW."
echo "Attach: tmux attach -t $SESSION  (select window $WINDOW)"
echo "After CUDA is True, run train_rl in that same bash. Keep the shell when done."
