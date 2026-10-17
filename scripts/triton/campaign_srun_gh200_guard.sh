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
  local result other_gh200
  other_gh200="$(squeue -h -u "${USER:?}" -o '%T %b' | \
    awk '$1 ~ /^(RUNNING|PENDING|CONFIGURING)$/ && $2 ~ /gres\/gpu:gh200=[1-9]/')"
  if [[ -n "$other_gh200" ]]; then
    printf 'GH200 request blocked by an existing or pending GH200 job: %s\n' \
      "$other_gh200" >&2
    return 75
  fi
  if ! result="$(python3 "$checker" --partition gpu-grace-h200-141g \
    --nodes 2 --gpu-type gh200 --gpus-per-node 1 \
    --cpus-per-node 48 --memory-mb 524288 2>&1)"; then
    printf 'GH200 preflight unavailable; no allocation submitted and no retry: %s\n' \
      "$result" >&2
    return 75
  fi
  CAMPAIGN_GH200_SELECTED_NODES="${result#available: }"
  if [[ "$CAMPAIGN_GH200_SELECTED_NODES" == "$result" \
    || -z "$CAMPAIGN_GH200_SELECTED_NODES" ]]; then
    printf 'GH200 preflight returned an invalid node list: %s\n' "$result" >&2
    return 2
  fi
  printf 'GH200 per-node preflight passed: %s\n' "$result" >&2
  command srun --test-only \
    --partition=gpu-grace-h200-141g \
    --nodelist="$CAMPAIGN_GH200_SELECTED_NODES" \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --cpus-per-task=48 --gpus-per-node=gh200:1 --mem=512G \
    --time=06:00:00 true
}

srun() {
  campaign_srun_gh200_guard "$@" || return $?

  local repo="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
  local -a forwarded=("$@")
  local i arg has_nodelist=0
  for arg in "${forwarded[@]}"; do
    [[ "$arg" == --nodelist=* ]] && has_nodelist=1
  done
  for ((i = 0; i + 1 < ${#forwarded[@]}; i++)); do
    if [[ "${forwarded[i]}" == "bash" \
      && "${forwarded[i + 1]}" == "$repo/scripts/matrix/run_campaign_18way_cpumask_allocation.sh" ]]; then
      if [[ -n "${CAMPAIGN_GH200_SELECTED_NODES:-}" && "$has_nodelist" -eq 0 ]]; then
        forwarded=(
          "${forwarded[@]:0:i}"
          "--nodelist=$CAMPAIGN_GH200_SELECTED_NODES"
          "${forwarded[@]:i}"
        )
        i=$((i + 1))
      fi
      forwarded[i + 1]="$repo/scripts/triton/campaign_allocation_wrapper.sh"
      break
    fi
  done
  command srun "${forwarded[@]}"
}
