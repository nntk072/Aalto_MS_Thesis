#!/usr/bin/env bash
# Benchmark harness: baseline vs Stages 1-3 (lazy VAE import + parent cleanup + thread caps)
# Measures: parent RSS, worker RSS, rollout throughput, rollout/learn time split, GPU usage.
# Uses the project's existing train_one_variant.sh + SB3 callbacks + /proc instrumentation.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"

# Configuration - short run for benchmarking
VARIANT="${VARIANT:-ladder_a0_flat}"
SEED="${SEED:-42}"
STEPS="${STEPS:-10000}"  # Short for benchmark, ~1-2 minutes
N_ENVS="${N_ENVS:-64}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/stages_bench_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# RAM-relevant files for Stages 2 + 3
RAM_FILES=(
  "quant_rl/models/agent.py"
  "quant_rl/train/train_rl.py"
  "quant_rl/envs/trading_env.py"  # Stage 1: lazy VAE import
)

# === Worker Process Monitoring ===
# Collect RSS from worker processes via /proc/<pid>/status
# Workers are spawned by SubprocVecEnv; we capture their PIDs from the fork

# Function to find worker PIDs from SubprocVecEnv
find_worker_pids() {
  # SubprocVecEnv processes will have cmdline containing python and the env factory
  # This is a best-effort approach since we can't directly access SubprocVecEnv internals
  local worker_pids=()
  
  # Look for python processes that are children of our current process
  local current_pid=$$
  for pid in $(ps -eo pid,ppid,comm | awk -v parent="$current_pid" '$2 == parent && $3 == "python" {print $1}'); do
    # Check if this looks like a worker (has SubprocVecEnv in its stack or env parameters)
    if [[ -f "/proc/$pid/cmdline" ]]; then
      local cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      if echo "$cmdline" | grep -q "make_env\|TradingEnv"; then
        worker_pids+=("$pid")
      fi
    fi
  done
  
  echo "${worker_pids[@]}"
}

# Function to get RSS for a specific PID
get_pid_rss_kb() {
  local pid="$1"
  if [[ -f "/proc/$pid/status" ]]; then
    grep "VmRSS:" "/proc/$pid/status" | awk '{print $2}' || echo "0"
  else
    echo "0"
  fi
}

# Function to get VmHWM for a specific PID
get_pid_vmhwm_kb() {
  local pid="$1"
  if [[ -f "/proc/$pid/status" ]]; then
    grep "VmHWM:" "/proc/$pid/status" | awk '{print $2}' || echo "0"
  else
    echo "0"
  fi
}

# Function to monitor worker RSS during training
monitor_workers() {
  local label="$1"
  local worker_log="$LOG_DIR/${label}_workers.log"
  
  echo "timestamp,pid,vmrss_kb,vmhwm_kb" > "$worker_log"
  
  # Monitor for up to 5 minutes in background
  local end_time=$((SECONDS + 300))
  while [[ $SECONDS -lt $end_time ]]; do
    local worker_pids=($(find_worker_pids))
    local timestamp=$(date +%s%N)
    
    for pid in "${worker_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        local rss=$(get_pid_rss_kb "$pid")
        local vmhwm=$(get_pid_vmhwm_kb "$pid")
        echo "${timestamp},${pid},${rss},${vmhwm}" >> "$worker_log"
      fi
    done
    sleep 5
  done &
  echo $!
}

# Function to get GPU metrics using nvidia-smi
get_gpu_metrics() {
  if command -v nvidia-smi &>/dev/null; then
    local gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$gpu_info" ]]; then
      echo "gpu_usage: $gpu_info"
    fi
  fi
}

# Function to run one benchmark iteration
run_one() {
  local label="$1"
  local log="$LOG_DIR/${label}.log"
  local start_ts end_ts
  local start_rss end_rss start_vmhwm end_vmhwm
  
  echo "=== [$label] Starting train VARIANT=$VARIANT SEED=$SEED STEPS=$STEPS N_ENVS=$N_ENVS ==="
  
  # Get starting parent memory
  start_rss=$(grep "VmRSS:" /proc/self/status | awk '{print $2}' || echo "0")
  start_vmhwm=$(grep "VmHWM:" /proc/self/status | awk '{print $2}' || echo "0")
  echo "[$label] Parent start RSS: ${start_rss} KB, VmHWM: ${start_vmhwm} KB"
  
  # Get GPU start metrics
  get_gpu_metrics > "$LOG_DIR/${label}_gpu_start.txt" || true
  
  start_ts=$(date +%s%N)
  
  # Start worker monitoring in background
  local monitor_pid=$(monitor_workers "$label")
  
  # Run training
  VARIANT="$VARIANT" SEED="$SEED" STEPS="$STEPS" \
  QUANT_RL_MAX_N_ENVS="$N_ENVS" \
  OUT_DIR="$OUT_DIR/$label" \
  LOG_DIR="$LOG_DIR" \
    bash scripts/train/train_one_variant.sh 2>&1 | tee "$log"
  
  # Stop worker monitoring
  if [[ -n "$monitor_pid" && "$monitor_pid" -ne 0 ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  
  end_ts=$(date +%s%N)
  
  # Get ending parent memory
  end_rss=$(grep "VmRSS:" /proc/self/status | awk '{print $2}' || echo "0")
  end_vmhwm=$(grep "VmHWM:" /proc/self/status | awk '{print $2}' || echo "0")
  echo "[$label] Parent end RSS: ${end_rss} KB, VmHWM: ${end_vmhwm} KB"
  
  # Get GPU end metrics
  get_gpu_metrics > "$LOG_DIR/${label}_gpu_end.txt" || true
  
  # Calculate metrics
  local wall_ms=$(( (end_ts - start_ts) / 1000000 ))
  echo "[$label] wall_ms=$wall_ms" | tee -a "$log"
  echo "[$label] parent_start_rss_kb=$start_rss" | tee -a "$log"
  echo "[$label] parent_end_rss_kb=$end_rss" | tee -a "$log"
  echo "[$label] parent_start_vmhwm_kb=$start_vmhwm" | tee -a "$log"
  echo "[$label] parent_end_vmhwm_kb=$end_vmhwm" | tee -a "$log"
  
  echo "[$label] Training complete"
}

# === Main Benchmark Execution ===

echo "=== Stages 1-3 Benchmark: Baseline vs Staged ==="
echo "Stages:"
echo "  1: Lazy VAE/torch import in trading_env.py"
echo "  2: Parent memory cleanup (del data, features, primary_m1, secondary_m1 + gc.collect)"
echo "  3: Worker thread caps (OMP/MKL/OPENBLAS_NUM_THREADS=1 + torch.set_num_threads(1))"
echo ""

# --- Baseline: stash Stages 1-3 changes, run, then restore ---
echo "=== Stashing Stages 1-3 changes for baseline ==="
git stash push -- "${RAM_FILES[@]}" -m "stages_bench_baseline" || true

run_one "baseline"

echo "=== Restoring Stages 1-3 changes for staged ==="
git stash pop || true

run_one "staged"

# --- Parse and Report Results ---
echo "=== Parsing results ==="
python3 "$REPO/scripts/bench/parse_stages_bench.py" "$LOG_DIR" "$OUT_DIR/summary.md" "$OUT_DIR/summary.json"

echo ""
echo "=== Benchmark complete ==="
echo "Results: $OUT_DIR/summary.md"
echo "JSON: $OUT_DIR/summary.json"
echo "Reproduce: VARIANT=$VARIANT SEED=$SEED STEPS=$STEPS N_ENVS=$N_ENVS bash scripts/bench/run_stages_benchmark.sh"