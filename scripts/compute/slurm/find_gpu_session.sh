#!/usr/bin/env bash
# Find whether this user already has a GPU-backed Slurm job and/or tmux GPU pane.
# Exit 0 if a reusable GPU allocation likely exists; exit 1 if a new srun may be needed.
# Login-node safe — no srun.
set -euo pipefail

SESSION="${TMUX_SESSION:-quant-rl-train}"
FOUND_JOB=0
FOUND_TMUX=0

echo "=== looking for running GPU / compute jobs ==="
if command -v squeue >/dev/null 2>&1; then
  # Running jobs for this user that mention gpu or grace partitions
  mapfile -t JOBS < <(squeue -u "${USER}" -h -t R -o '%i %P %b %N' 2>/dev/null || true)
  if ((${#JOBS[@]})); then
    FOUND_JOB=1
    printf '%s\n' "${JOBS[@]}"
  else
    echo "(no running jobs)"
  fi
else
  echo "(squeue not available)"
fi
echo

echo "=== looking for tmux session $SESSION ==="
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
  FOUND_TMUX=1
  tmux list-windows -t "$SESSION"
  echo
  echo "--- recent pane tails (first 3 windows) ---"
  mapfile -t WINS < <(tmux list-windows -t "$SESSION" -F '#{window_index}:#{window_name}' 2>/dev/null || true)
  n=0
  for w in "${WINS[@]}"; do
    idx="${w%%:*}"
    name="${w#*:}"
    echo ">>>> $SESSION:$idx ($name)"
    tmux capture-pane -t "$SESSION:$idx" -p 2>/dev/null | tail -15 || true
    echo
    n=$((n + 1))
    (( n >= 3 )) && break
  done
else
  echo "(session $SESSION not found)"
fi

echo "=== verdict ==="
if [[ "$FOUND_JOB" -eq 1 ]]; then
  echo "REUSE: running Slurm job(s) found — attach tmux and run train inside that GPU bash."
  echo "Do NOT start a new srun. Do NOT scancel unless the user asks."
  exit 0
fi
if [[ "$FOUND_TMUX" -eq 1 ]]; then
  echo "CHECK: tmux exists but no running squeue GPU job — inspect panes for an interactive shell."
  echo "If a pane is already on a compute node with CUDA, reuse it; else allocate once."
  exit 0
fi
echo "ALLOC: no GPU job / useful session found — may run attach_or_alloc.sh once."
exit 1
