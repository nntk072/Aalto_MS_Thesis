#!/usr/bin/env bash
set -euo pipefail
# Parallel batch runner: run isolated Docker containers in parallel.
#
# Builds each service into its own image, then runs them concurrently.
# Useful for running independent experiments (e.g. multi-symbol backtests).
#
# Usage:
#   ./scripts/docker/parallel-run.sh <service> [args...]
#
# Examples:
#   # Run 4 baseline jobs in parallel across symbols
#   ./scripts/docker/parallel-run.sh baseline-macd \
#     --symbol EURUSD --symbol GBPUSD --symbol USDJPY --symbol AUDUSD
#
#   # Parallel training with different seeds
#   printf "42\n73\n99\n" | xargs -P4 -I{} \
#     docker compose run --rm -e SEED={} train-rl

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.yml"
SERVICE="${1:?Usage: $0 <service> [args...]}"
shift || true

PARALLEL_LOG_DIR="outputs/parallel-logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARALLEL_LOG_DIR"

echo "Building $SERVICE image..."
docker compose -f "$COMPOSE_FILE" build "$SERVICE"

echo "Running $SERVICE (parallel batch)..."
echo "Args: $@"
echo "Logs: $PARALLEL_LOG_DIR/"
echo ""

# Run with docker compose run --rm
# If no additional args, run once
if [ $# -eq 0 ]; then
  docker compose -f "$COMPOSE_FILE" run --name "${SERVICE}-batch" --rm "$SERVICE" \
    2>&1 | tee "$PARALLEL_LOG_DIR/${SERVICE}.log"
  exit "${PIPESTATUS[0]}"
fi

# Split remaining args into parallel jobs (one per arg value)
# when --key repeated values are provided on the CLI.
# Hack: detect repeated --flag values and split across jobs.
declare -a JOB_ARGS=()
JOB_VALUES=()
LAST_FLAG=""

for arg in "$@"; do
  if [[ "$arg" == --* ]]; then
    LAST_FLAG="$arg"
  else
    JOB_VALUES+=("$arg")
  fi
done

if [ ${#JOB_VALUES[@]} -eq 0 ]; then
  docker compose -f "$COMPOSE_FILE" run --name "${SERVICE}-batch" --rm "$SERVICE" \
    "$@" 2>&1 | tee "$PARALLEL_LOG_DIR/${SERVICE}.log"
  exit "${PIPESTATUS[0]}"
fi

echo "Found ${#JOB_VALUES[@]} values for parallel execution"
echo ""

PIDS=()
INDEX=0
for value in "${JOB_VALUES[@]}"; do
  log_file="$PARALLEL_LOG_DIR/${SERVICE}-${INDEX}.log"
  container_name="${SERVICE}-job-${INDEX}"

  docker compose -f "$COMPOSE_FILE" run \
    --name "$container_name" \
    --rm \
    -e "PARALLEL_INDEX=$INDEX" \
    "$SERVICE" \
    "$LAST_FLAG" "$value" > "$log_file" 2>&1 &

  pid=$!
  PIDS+=("$pid:$INDEX:$value")
  echo "  [$INDEX] $SERVICE -- $LAST_FLAG $value  (PID $pid, log: $log_file)"

  INDEX=$((INDEX + 1))
done

echo ""
echo "Waiting for ${#PIDS[@]} parallel jobs..."
echo ""

FAILED=()
for entry in "${PIDS[@]}"; do
  pid="${entry%%:*}"
  rest="${entry#*:}"
  idx="${rest%%:*}"
  value="${rest##*:}"

  if wait "$pid"; then
    echo "  ✓ [$idx] $value passed"
  else
    code=$?
    echo "  ✗ [$idx] $value FAILED (exit $code)"
    FAILED+=("$idx")
    echo "    log: $PARALLEL_LOG_DIR/${SERVICE}-${idx}.log"
    tail -10 "$PARALLEL_LOG_DIR/${SERVICE}-${idx}.log" | sed 's/^/    /'
  fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All ${#PIDS[@]} jobs passed"
else
  echo "${#FAILED[@]}/${#PIDS[@]} jobs FAILED: ${FAILED[*]}"
fi

[ ${#FAILED[@]} -eq 0 ]
