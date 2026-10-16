#!/usr/bin/env bash
# Source through BASH_ENV to avoid submitting the campaign allocation until
# two nodes can satisfy the requested one-GH200/48-CPU/512-GiB shape.

if [[ "${CAMPAIGN_SRUN_GH200_GUARD_LOADED:-}" == "1" ]]; then
  return 0
fi
export CAMPAIGN_SRUN_GH200_GUARD_LOADED=1

campaign_srun_gh200_guard() {
  local guarded=0
  local arg
  for arg in "$@"; do
    case "$arg" in
      --partition=gpu-grace-h200-141g) guarded=$((guarded | 1)) ;;
      --nodes=2) guarded=$((guarded | 2)) ;;
      --gpus-per-node=gh200:1) guarded=$((guarded | 4)) ;;
      --cpus-per-task=48) guarded=$((guarded | 8)) ;;
      --mem=512G) guarded=$((guarded | 16)) ;;
    esac
  done
  [[ "$guarded" -eq 31 ]] || return 0

  local checker="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}/scripts/triton/check_gh200_capacity.py"
  local last_notice=0 now result
  while true; do
    if result="$(python3 "$checker" --partition gpu-grace-h200-141g \
      --nodes 2 --gpu-type gh200 --gpus-per-node 1 \
      --cpus-per-node 48 --memory-mb 524288 2>&1)"; then
      printf 'GH200 preflight passed: %s\n' "$result" >&2
      return 0
    else
      local status=$?
      now="$(date +%s)"
      if (( now - last_notice >= 300 )); then
        printf 'GH200 preflight waiting; no allocation submitted: %s (check=%s)\n' \
          "$result" "$status" >&2
        last_notice="$now"
      fi
      sleep 60
    fi
  done
}

srun() {
  campaign_srun_gh200_guard "$@" || return $?

  local repo="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
  local -a forwarded=("$@")
  local i
  for ((i = 0; i + 1 < ${#forwarded[@]}; i++)); do
    if [[ "${forwarded[i]}" == "bash" \
      && "${forwarded[i + 1]}" == "$repo/scripts/matrix/run_campaign_18way_cpumask_allocation.sh" ]]; then
      forwarded[i + 1]="$repo/scripts/triton/campaign_allocation_wrapper.sh"
      break
    fi
  done
  command srun "${forwarded[@]}"
}
