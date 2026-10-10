#!/usr/bin/env bash
set -euo pipefail

# Unified, resource-aware matrix launcher.
# Selection order: GPU family GH200 -> H200 -> H100, then mode 3 -> 4 -> 6 -> 2 -> 1 -> 5.
#
# Modes:
#   1: one GPU, 12h, 900G, four trainers, n_envs=32
#   2: two GPUs on one node, 6h, 900G, two trainers/GPU, n_envs=32
#   3: one GPU on each of two nodes, 6h/900G/node, two trainers/node, n_envs=64
#   4: one GPU on each of two nodes, 6h/600G/node, two trainers/node, n_envs=32
#   5: one GPU, 12h, 600G, 32 CPUs, two trainers, n_envs=64 (lowest-priority fallback, floor 550G)
#   6: one GPU, 12h, 900G, 32 CPUs, three trainers, n_envs=64 (prefer 900G, floor 850G)
#
# Usage:
#   run_18variant_matrix.sh [all|variant1,variant2,...]
# With no argument, all variants in scripts/variant_list.txt are selected.
# Use "list" as the argument to print the available variants and exit.

REPO="${TRITON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/gpu_matrix_auto_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
SEED="${SEED:-42}"
# Empty = steps declared per variant in config/experiments.yaml (resolved by the launcher).
STEPS="${STEPS:-}"
VARIANT_LIST="$REPO/scripts/variant_list.txt"
SELECTION="${1:-all}"

cd "$REPO"
mkdir -p "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/all.log") 2>&1

mapfile -t ALL_VARIANTS < "$VARIANT_LIST"
if [[ "$SELECTION" == "list" ]]; then
  printf '%s\n' "${ALL_VARIANTS[@]}"
  exit 0
fi

if [[ "$SELECTION" == "all" ]]; then
  SELECTED_VARIANTS=("${ALL_VARIANTS[@]}")
else
  IFS=',' read -r -a SELECTED_VARIANTS <<< "$SELECTION"
fi

if [[ "${#SELECTED_VARIANTS[@]}" -eq 0 ]]; then
  echo "ERROR: no variants selected" >&2
  exit 2
fi
for variant in "${SELECTED_VARIANTS[@]}"; do
  if ! printf '%s\n' "${ALL_VARIANTS[@]}" | grep -Fxq "$variant"; then
    echo "ERROR: unknown variant '$variant'; use '$0 list'" >&2
    exit 2
  fi
done

REMAINING_VARIANTS=()
for variant in "${SELECTED_VARIANTS[@]}"; do
  if [[ -f "$OUT_DIR/${variant}__seed${SEED}.done" ]]; then
    echo "SKIP completed variant=$variant"
  else
    REMAINING_VARIANTS+=("$variant")
  fi
done
SELECTED_VARIANTS=("${REMAINING_VARIANTS[@]}")
if [[ "${#SELECTED_VARIANTS[@]}" -eq 0 ]]; then
  echo "ALL SELECTED VARIANTS COMPLETE"
  exit 0
fi

configure_gpu() {
  case "$1" in
    gh200) PARTITION=gpu-grace-h200-141g; GPU_TYPE=gh200; NODE_A=gpuarm1; NODE_B=gpuarm2 ;;
    h200)  PARTITION=gpu-h200-141g-m; GPU_TYPE=h200; NODE_A=gpu51; NODE_B=gpu52 ;;
    h100)  PARTITION=gpu-h100-80g; GPU_TYPE=h100; NODE_A=gpu45; NODE_B=gpu46 ;;
    *) return 1 ;;
  esac
}

configure_mode() {
  case "$1" in
    1) GPUS=1; TIME=12:00:00; MEM=900G; CPUS=32; ENVS=32; PER_BATCH=4 ;;
    2) GPUS=2; TIME=6:00:00; MEM=900G; CPUS=32; ENVS=32; PER_BATCH=4 ;;
    3) GPUS=1; TIME=6:00:00; MEM=900G; CPUS=16; ENVS=64; PER_BATCH=4 ;;
    4) GPUS=1; TIME=6:00:00; MEM=600G; CPUS=16; ENVS=32; PER_BATCH=4 ;;
    5) GPUS=1; TIME=12:00:00; MEM=600G; CPUS=32; ENVS=64; PER_BATCH=2 ;;
    6) GPUS=1; TIME=12:00:00; MEM=900G; CPUS=32; ENVS=64; PER_BATCH=3 ;;
    *) return 1 ;;
  esac
}

# RAM tolerance: accept up to 50G below the requested MEM.
MEM_TOL_GB=50
MIN_MEM=880G
MIN_BATCH_REMAINING_SECONDS=$((4 * 3600 + 30 * 60))

# Minimum acceptable MEM for a requested value (requested - 50G).
min_mem_for() {
  local requested_mb minimum_mb
  requested_mb=$(mem_mb "$1")
  minimum_mb=$((requested_mb - MEM_TOL_GB * 1024))
  (( minimum_mb > 0 )) || minimum_mb="$requested_mb"
  if (( minimum_mb % 1024 == 0 )); then
    echo "$((minimum_mb / 1024))G"
  else
    echo "${minimum_mb}M"
  fi
}

resource_test() {
  local gpu="$1" mode="$2"
  local test_mem test_mem_b mode_min
  configure_gpu "$gpu" || return 1
  configure_mode "$mode" || return 1
  mode_min="$(min_mem_for "$MEM")"
  if [[ "$mode" == 5 || "$mode" == 6 ]]; then
    # Mode 5/6 single-GPU fallback may run on either node of the pair.
    if node_has_capacity "$NODE_A" "$GPU_TYPE" "$GPUS" "$mode_min" "$CPUS"; then
      :
    elif node_has_capacity "$NODE_B" "$GPU_TYPE" "$GPUS" "$mode_min" "$CPUS"; then
      local swap="$NODE_A"; NODE_A="$NODE_B"; NODE_B="$swap"
    else
      return 1
    fi
  else
    node_has_capacity "$NODE_A" "$GPU_TYPE" "$GPUS" "$mode_min" "$CPUS" || return 1
  fi
  test_mem="$(select_memory_request "$NODE_A" "$MEM" "$mode_min")" || return 1
  if [[ "$mode" == 3 || "$mode" == 4 ]]; then
    node_has_capacity "$NODE_B" "$GPU_TYPE" 1 "$mode_min" "$CPUS" || return 1
    test_mem_b="$(select_memory_request "$NODE_B" "$MEM" "$mode_min")" || return 1
    [[ "$test_mem_b" == 880G ]] && test_mem=880G
    srun --test-only --partition="$PARTITION" --nodelist="$NODE_A" \
      --gpus="$GPU_TYPE:1" --time="$TIME" --mem="$test_mem" \
      --ntasks=1 --cpus-per-task="$CPUS" true >/dev/null 2>&1 || return 1
    srun --test-only --partition="$PARTITION" --nodelist="$NODE_B" \
      --gpus="$GPU_TYPE:1" --time="$TIME" --mem="$test_mem" \
      --ntasks=1 --cpus-per-task="$CPUS" true >/dev/null 2>&1 || return 1
  else
    srun --test-only --partition="$PARTITION" --nodelist="$NODE_A" \
      --gpus="$GPU_TYPE:$GPUS" --time="$TIME" --mem="$test_mem" \
      --ntasks=1 --cpus-per-task="$CPUS" true >/dev/null 2>&1 || return 1
  fi
}

select_memory_request() {
  local node="$1" requested="$2" minimum="$3" info cfg alloc cfg_mem alloc_mem free_mem
  info="$(scontrol show node -o "$node" 2>/dev/null)" || return 1
  cfg="$(sed -n 's/.*CfgTRES=\([^ ]*\).*/\1/p' <<< "$info")"
  alloc="$(sed -n 's/.*AllocTRES=\([^ ]*\).*/\1/p' <<< "$info")"
  cfg_mem="$(mem_mb "$(sed -n 's/.*mem=\([0-9]*[MG]\).*/\1/p' <<< "$cfg")")"
  alloc_mem="$(mem_mb "$(sed -n 's/.*mem=\([0-9]*[MG]\).*/\1/p' <<< "$alloc")")"
  free_mem=$((cfg_mem - alloc_mem))
  if (( free_mem >= $(mem_mb "$requested") )); then
    echo "$requested"
  elif (( free_mem >= $(mem_mb "$minimum") )); then
    echo "$minimum"
  else
    return 1
  fi
}

tres_value() {
  local tres="$1" key="$2"
  if [[ "$tres" =~ $key=([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo 0
  fi
}

mem_mb() {
  local value="$1" amount unit
  [[ "$value" =~ ^([0-9]+)([MG])$ ]] || { echo 0; return; }
  amount="${BASH_REMATCH[1]}"
  unit="${BASH_REMATCH[2]}"
  if [[ "$unit" == G ]]; then
    echo $((amount * 1024))
  else
    echo "$amount"
  fi
}

node_has_capacity() {
  local node="$1" gpu_type="$2" gpu_count="$3" memory="$4" cpus="$5"
  local info state cfg alloc cfg_gpu alloc_gpu cfg_cpu alloc_cpu
  local cfg_mem alloc_mem requested_mem
  info="$(scontrol show node -o "$node" 2>/dev/null)" || return 1
  state="$(sed -n 's/.*State=\([^ ]*\).*/\1/p' <<< "$info")"
  [[ "$state" != *PLANNED* ]] || return 1
  cfg="$(sed -n 's/.*CfgTRES=\([^ ]*\).*/\1/p' <<< "$info")"
  alloc="$(sed -n 's/.*AllocTRES=\([^ ]*\).*/\1/p' <<< "$info")"
  cfg_gpu="$(tres_value "$cfg" "gres/gpu:$gpu_type")"
  alloc_gpu="$(tres_value "$alloc" "gres/gpu:$gpu_type")"
  cfg_cpu="$(tres_value "$cfg" "cpu")"
  alloc_cpu="$(tres_value "$alloc" "cpu")"
  cfg_mem="$(mem_mb "$(sed -n 's/.*mem=\([0-9]*[MG]\).*/\1/p' <<< "$cfg")")"
  alloc_mem="$(mem_mb "$(sed -n 's/.*mem=\([0-9]*[MG]\).*/\1/p' <<< "$alloc")")"
  requested_mem="$(mem_mb "$memory")"
  (( cfg_gpu - alloc_gpu >= gpu_count )) || return 1
  (( cfg_cpu - alloc_cpu >= cpus )) || return 1
  (( cfg_mem - alloc_mem >= requested_mem )) || return 1
}

duration_seconds() {
  local value="$1" days=0 hours=0 minutes=0 seconds=0
  if [[ "$value" == *-* ]]; then
    days="${value%%-*}"
    value="${value#*-}"
  fi
  IFS=: read -r -a parts <<< "$value"
  case "${#parts[@]}" in
    3) hours="${parts[0]}"; minutes="${parts[1]}"; seconds="${parts[2]}" ;;
    2) minutes="${parts[0]}"; seconds="${parts[1]}" ;;
    1) seconds="${parts[0]}" ;;
    *) return 1 ;;
  esac
  echo $((days * 86400 + hours * 3600 + minutes * 60 + seconds))
}

allocation_remaining_seconds() {
  local time_left
  [[ -n "${SLURM_JOB_ID:-}" ]] || return 1
  time_left="$(squeue -h -j "$SLURM_JOB_ID" -o '%L' 2>/dev/null | head -n 1)"
  duration_seconds "$time_left"
}

monitor_srun_allocation() {
  local srun_pid="$1" job_name="$2" job_id remaining
  while kill -0 "$srun_pid" 2>/dev/null; do
    job_id="$(squeue -h -u "$USER" -n "$job_name" -o '%A' 2>/dev/null | head -n 1)"
    if [[ -n "$job_id" ]]; then
      remaining="$(squeue -h -j "$job_id" -o '%L' 2>/dev/null | head -n 1)"
      remaining="$(duration_seconds "$remaining" 2>/dev/null || echo 0)"
      if (( remaining < MIN_BATCH_REMAINING_SECONDS )); then
        echo "ROLLOVER: srun job=$job_id name=$job_name has ${remaining}s remaining; stopping before hard timeout"
        scancel "$job_id"
        wait "$srun_pid" 2>/dev/null || true
        return 75
      fi
    fi
    sleep 30
  done
  wait "$srun_pid"
}

current_allocation() {
  local job_id="${1:-${SLURM_JOB_ID:-}}" job_info alloc tres node_list
  local allocated_gpus allocated_cpus allocated_mem
  local -a allocation_nodes
  [[ -n "$job_id" ]] || return 1
  job_info="$(scontrol show job -o "$job_id" 2>/dev/null)" || return 1
  [[ "$job_info" == *"JobState=RUNNING"* ]] || return 1
  alloc="$(sed -n 's/.*AllocTRES=\([^ ]*\).*/\1/p' <<< "$job_info")"
  tres="$(sed -n 's/.*ReqTRES=\([^ ]*\).*/\1/p' <<< "$job_info")"
  node_list="$(sed -n 's/.*NodeList=\([^ ]*\).*/\1/p' <<< "$job_info")"
  mapfile -t allocation_nodes < <(scontrol show hostnames "$node_list" 2>/dev/null)
  ((${#allocation_nodes[@]} > 0)) || return 1
  NODE_A="${allocation_nodes[0]}"
  NODE_B="${allocation_nodes[1]:-}"

  SELECTED_GPU=
  for candidate in gh200 h200 h100; do
    if [[ "$alloc" == *"gres/gpu:${candidate}="* ||
          "$tres" == *"gres/gpu:${candidate}="* ]]; then
      SELECTED_GPU="$candidate"
      GPU_TYPE="$candidate"
      break
    fi
  done
  [[ -n "$SELECTED_GPU" ]] || return 1

  GPUS="$(tres_value "$alloc" "gres/gpu:${GPU_TYPE}")"
  [[ "$GPUS" -gt 0 ]] || GPUS="$(tres_value "$tres" "gres/gpu:${GPU_TYPE}")"
  CPUS="$(tres_value "$alloc" "cpu")"
  [[ "$CPUS" -gt 0 ]] || CPUS="$(tres_value "$tres" "cpu")"
  MEM="$(sed -n 's/.*mem=\([0-9]*[MG]\).*/\1/p' <<< "$alloc")"
  [[ -n "$MEM" ]] || MEM="$(sed -n 's/.*mem=\([0-9]*[MG]\).*/\1/p' <<< "$tres")"
  PARTITION="$(sed -n 's/.*Partition=\([^ ]*\).*/\1/p' <<< "$job_info")"
  case "$PARTITION" in
    gpu-grace-h200-141g|gpu-h200-141g-m|gpu-h100-80g) ;;
    *) return 1 ;;
  esac
  (( $(mem_mb "$MEM") >= $(mem_mb 600G) )) || return 1
  (( GPUS >= 1 && GPUS <= 2 )) || return 1
  (( CPUS >= 16 )) || return 1
  allocated_gpus="$GPUS"
  allocated_cpus="$CPUS"
  allocated_mem="$MEM"
  if [[ -n "$NODE_B" ]]; then
    [[ "$GPUS" -eq 2 ]] || return 1
    if (( $(mem_mb "$MEM") >= $(mem_mb 900G) )); then
      SELECTED_MODE=3
    else
      SELECTED_MODE=4
    fi
    configure_mode "$SELECTED_MODE"
  elif [[ "$GPUS" -ge 2 ]]; then
    SELECTED_MODE=2
    configure_mode "$SELECTED_MODE"
  elif (( $(mem_mb "$MEM") >= $(mem_mb 850G) )) && (( CPUS >= 32 )); then
    SELECTED_MODE=6
    configure_mode "$SELECTED_MODE"
    ENVS=64
  else
    SELECTED_MODE=5
    configure_mode "$SELECTED_MODE"
  fi
  GPUS="$allocated_gpus"
  CPUS="$allocated_cpus"
  MEM="$allocated_mem"

  REUSE_JOB_ID="$job_id"
  echo "REUSE current allocation job=$job_id state=RUNNING"
  echo "REUSE resources: gpu=$SELECTED_GPU nodes=$NODE_A${NODE_B:+,$NODE_B} gpus=$GPUS cpus=$CPUS mem=$MEM mode=$SELECTED_MODE"
  return 0
}

find_reusable_allocation() {
  local job_id remaining_seconds
  while read -r job_id; do
    [[ -n "$job_id" ]] || continue
    remaining_seconds="$(squeue -h -j "$job_id" -o '%L' 2>/dev/null | head -n 1)"
    remaining_seconds="$(duration_seconds "$remaining_seconds" 2>/dev/null || echo 0)"
    (( remaining_seconds >= MIN_BATCH_REMAINING_SECONDS )) || continue
    if current_allocation "$job_id"; then
      return 0
    fi
  done < <(squeue -h -u "$USER" -t RUNNING -o '%A' 2>/dev/null)
  return 1
}

SELECTED_GPU=
SELECTED_MODE=
IN_CURRENT_ALLOCATION=0
if current_allocation; then
  IN_CURRENT_ALLOCATION=1
elif find_reusable_allocation; then
  IN_CURRENT_ALLOCATION=2
else
  for gpu in gh200 h200 h100; do
    for mode in 3 4 6 2 1 5; do
      echo "CHECK gpu=$gpu mode=$mode"
      if resource_test "$gpu" "$mode"; then
        SELECTED_GPU="$gpu"
        SELECTED_MODE="$mode"
        break 2
      fi
    done
  done
fi

if [[ -z "$SELECTED_GPU" ]]; then
  echo "NO TRAINING: no requested GPU/resource shape is currently feasible." >&2
  exit 3
fi

if [[ "$IN_CURRENT_ALLOCATION" -eq 0 ]]; then
  configure_gpu "$SELECTED_GPU"
  configure_mode "$SELECTED_MODE"
  if [[ "$SELECTED_MODE" == 5 || "$SELECTED_MODE" == 6 ]]; then
    # configure_gpu reset NODE_A/NODE_B; re-pick the feasible single node.
    _mode_min="$(min_mem_for "$MEM")"
    if node_has_capacity "$NODE_A" "$GPU_TYPE" "$GPUS" "$_mode_min" "$CPUS"; then
      :
    elif node_has_capacity "$NODE_B" "$GPU_TYPE" "$GPUS" "$_mode_min" "$CPUS"; then
      _swap="$NODE_A"; NODE_A="$NODE_B"; NODE_B="$_swap"
    fi
  fi
  MEM="$(select_memory_request "$NODE_A" "$MEM" "$(min_mem_for "$MEM")")"
  if [[ "$SELECTED_MODE" == 3 || "$SELECTED_MODE" == 4 ]]; then
    MEM_B="$(select_memory_request "$NODE_B" "$MEM" "$(min_mem_for "$MEM")")"
    if [[ "$MEM_B" == 880G && "$MEM" == 900G ]]; then MEM=880G; fi
  fi
fi
mkdir -p "$OUT_DIR"
echo "SELECTED gpu=$SELECTED_GPU mode=$SELECTED_MODE partition=$PARTITION nodes=$NODE_A,$NODE_B"
echo "SELECTED resources: gpus=$GPUS time=$TIME mem=$MEM cpus=$CPUS n_envs=$ENVS"
echo "SELECTED variants (${#SELECTED_VARIANTS[@]}): ${SELECTED_VARIANTS[*]}"

if [[ "$IN_CURRENT_ALLOCATION" -eq 2 ]]; then
  echo "REUSE: entering existing allocation $REUSE_JOB_ID with an srun step"
  exec srun --jobid="$REUSE_JOB_ID" --overlap --ntasks=1 \
    --cpus-per-task="$CPUS" --chdir="$REPO" \
    env TRITON_REPO="$REPO" RUN_STAMP="$RUN_STAMP" OUT_DIR="$OUT_DIR" \
    LOG_DIR="$LOG_DIR" SEED="$SEED" STEPS="$STEPS" \
    bash "$REPO/scripts/matrix/run_18variant_matrix.sh" "$SELECTION"
fi

run_variant() {
  # No resume: train_rl has no checkpoint-restore path. A rolled-over job restarts
  # its variant from scratch, and the seed-keyed .done marker keeps finished seeds.
  local variant="$1" gpu_id="$2"
  VARIANT="$variant" SEED="$SEED" STEPS="$STEPS" GPU_ID="$gpu_id" \
    QUANT_RL_MAX_N_ENVS="$ENVS" OUT_DIR="$OUT_DIR" LOG_DIR="$LOG_DIR" \
    bash "$REPO/scripts/train/train_one_variant.sh"
}

run_same_node_batch() {
  local start="$1" i gpu_id
  local -a pids=()
  for ((i = 0; i < PER_BATCH && start + i < ${#SELECTED_VARIANTS[@]}; i++)); do
    gpu_id=0
    [[ "$SELECTED_MODE" == 2 ]] && gpu_id=$((i / 2))
    run_variant "${SELECTED_VARIANTS[start + i]}" "$gpu_id" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || return 1; done
}

run_split_batch() {
  local start="$1" node="$2"
  run_variant "${SELECTED_VARIANTS[start]}" 0 &
  local p1=$!
  run_variant "${SELECTED_VARIANTS[start + 1]}" 0 &
  local p2=$!
  wait "$p1" && wait "$p2"
}

run_same_node_allocation() {
  local start="$1"
  if [[ "$IN_CURRENT_ALLOCATION" -eq 1 ]]; then
    run_same_node_batch "$start"
    return
  fi
  local job_name="matrix-${RUN_STAMP}-${start}"
  srun --job-name="$job_name" --kill-on-bad-exit=1 \
    --partition="$PARTITION" --nodelist="$NODE_A" \
    --gpus="$GPU_TYPE:$GPUS" --time="$TIME" --mem="$MEM" \
    --ntasks=1 --cpus-per-task="$CPUS" --chdir="$REPO" \
    env OUT_DIR="$OUT_DIR" SEED="$SEED" STEPS="$STEPS" ENVS="$ENVS" \
    LOG_DIR="$LOG_DIR" \
    SELECTED_MODE="$SELECTED_MODE" bash -c \
    "$(declare -f run_variant run_same_node_batch); \
      REPO='$REPO'; OUT_DIR='$OUT_DIR'; SEED='$SEED'; STEPS='$STEPS'; \
      LOG_DIR='$LOG_DIR'; ENVS='$ENVS'; SELECTED_MODE='$SELECTED_MODE'; PER_BATCH='$PER_BATCH'; \
      SELECTED_VARIANTS=($(printf '%q ' "${SELECTED_VARIANTS[@]}")); \
      run_same_node_batch '$start'" &
  local srun_pid=$!
  monitor_srun_allocation "$srun_pid" "$job_name"
}

run_split_allocation() {
  local start="$1" node="$2"
  if [[ "$IN_CURRENT_ALLOCATION" -eq 1 ]]; then
    run_split_batch "$start" "$node"
    return
  fi
  srun --partition="$PARTITION" --nodelist="$node" \
    --gpus="$GPU_TYPE:1" --time="$TIME" --mem="$MEM" \
    --ntasks=1 --cpus-per-task="$CPUS" --chdir="$REPO" \
    env OUT_DIR="$OUT_DIR" SEED="$SEED" STEPS="$STEPS" ENVS="$ENVS" \
    LOG_DIR="$LOG_DIR" \
    bash -c \
    "$(declare -f run_variant run_split_batch); \
      REPO='$REPO'; OUT_DIR='$OUT_DIR'; SEED='$SEED'; STEPS='$STEPS'; \
      LOG_DIR='$LOG_DIR'; ENVS='$ENVS'; SELECTED_VARIANTS=($(printf '%q ' "${SELECTED_VARIANTS[@]}")); \
      run_split_batch '$start' '$node'"
}

rollover_allocation() {
  local start="$1" remaining_selection child_status
  remaining_selection="$(IFS=,; echo "${SELECTED_VARIANTS[*]:start}")"
  echo "ROLLOVER: allocation ${SLURM_JOB_ID:-unknown} has less than 4.5h remaining; requesting same config for variants starting at index $start"
  echo "ROLLOVER resources: partition=$PARTITION node=$NODE_A gpus=$GPU_TYPE:$GPUS mem=$MEM cpus=$CPUS time=$TIME"

  set +e
  srun --partition="$PARTITION" --nodelist="$NODE_A" \
    --gpus="$GPU_TYPE:$GPUS" --time="$TIME" --mem="$MEM" \
    --ntasks=1 --cpus-per-task="$CPUS" --chdir="$REPO" \
    env TRITON_REPO="$REPO" RUN_STAMP="$RUN_STAMP" OUT_DIR="$OUT_DIR" \
    LOG_DIR="$LOG_DIR" SEED="$SEED" STEPS="$STEPS" \
    bash "$REPO/scripts/matrix/run_18variant_matrix.sh" "$remaining_selection"
  child_status=$?
  set -e

  if [[ "$child_status" -eq 0 ]]; then
    echo "ROLLOVER: replacement allocation completed; cancelling old allocation ${SLURM_JOB_ID:-unknown}"
    [[ -n "${SLURM_JOB_ID:-}" ]] && scancel "$SLURM_JOB_ID" || true
  fi
  return "$child_status"
}

FAILURES=()
for ((start = 0, batch = 0; start < ${#SELECTED_VARIANTS[@]}; start += PER_BATCH, batch++)); do
  echo "=== MATRIX batch=$batch start=$(date -Is) ==="
  if [[ "$IN_CURRENT_ALLOCATION" -eq 1 ]]; then
    remaining_seconds="$(allocation_remaining_seconds || echo 0)"
    if (( remaining_seconds < MIN_BATCH_REMAINING_SECONDS )); then
      rollover_allocation "$start"
      exit $?
    fi
  fi
  set +e
  if [[ "$SELECTED_MODE" == 3 || "$SELECTED_MODE" == 4 ]]; then
    run_split_allocation "$start" "$NODE_A" & pid_a=$!
    run_split_allocation "$((start + 2))" "$NODE_B" & pid_b=$!
    wait "$pid_a"; status_a=$?
    wait "$pid_b"; status_b=$?
    status=$((status_a != 0 || status_b != 0))
  else
    run_same_node_allocation "$start"
    status=$?
  fi
  set -e
  if [[ "$status" -ne 0 ]]; then
    if [[ "$status" -eq 75 ]]; then
      echo "ROLLOVER: retrying batch=$batch from checkpoints with a fresh allocation"
      continue
    fi
    FAILURES+=("batch_$batch")
    echo "FAILED batch=$batch status=$status" | tee "$OUT_DIR/FAILED.txt"
    break
  fi
  echo "=== MATRIX batch=$batch done $(date -Is) ==="
done

echo "=== MATRIX SUMMARY gpu=$SELECTED_GPU mode=$SELECTED_MODE ==="
echo "Done variants: $(find "$OUT_DIR" -maxdepth 1 -name '*.done' | wc -l)"
echo "Failed variants: $(find "$OUT_DIR" -maxdepth 1 -name '*.failed' | wc -l)"
if [[ "${#FAILURES[@]}" -gt 0 ]]; then exit 1; fi
echo "=== SELECTED VARIANTS COMPLETE ==="
