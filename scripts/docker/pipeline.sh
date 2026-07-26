#!/usr/bin/env bash
set -euo pipefail
# Pipeline runner: orchestrate Docker Compose services in parallel.
# Usage:
#   ./scripts/docker/pipeline.sh [stage...]
#   ./scripts/docker/pipeline.sh          # run full pipeline
#   ./scripts/docker/pipeline.sh test     # run only tests
#   ./scripts/docker/pipeline.sh train-rl backtest  # run specific stages

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.yml"
PIPELINE_LOG_DIR="outputs/pipeline-logs"
mkdir -p "$PIPELINE_LOG_DIR"

# Default: run all pipeline stages in dependency order
DEFAULT_STAGES=(
  prepare-data
  discover-symbols
  lint
  typecheck
  test
  baseline-macd
)

if [ $# -gt 0 ]; then
  STAGES=("$@")
else
  STAGES=("${DEFAULT_STAGES[@]}")
fi

echo "========================================"
echo " Pipeline stages: ${STAGES[*]}"
echo "========================================"

# Build images first
docker compose -f "$COMPOSE_FILE" build base dev test

RUN_IDS=()
for stage in "${STAGES[@]}"; do
  log_file="$PIPELINE_LOG_DIR/${stage}.log"

  # Determine compose service name
  service="$stage"

  echo ""
  echo "--- Starting: $stage ---"

  docker compose -f "$COMPOSE_FILE" run --rm \
    -T \
    "$service" > "$log_file" 2>&1 &

  pid=$!
  RUN_IDS+=("$stage:$pid")
  echo "  launched (PID $pid, log: $log_file)"
done

echo ""
echo "========================================"
echo " Waiting for all stages to finish..."
echo "========================================"

FAILED=()
for entry in "${RUN_IDS[@]}"; do
  stage="${entry%%:*}"
  pid="${entry##*:}"
  log_file="$PIPELINE_LOG_DIR/${stage}.log"

  if wait "$pid"; then
    echo "  ✓ $stage passed"
  else
    echo "  ✗ $stage FAILED (exit code $?)"
    FAILED+=("$stage")
    echo "    last 20 lines of $log_file:"
    tail -20 "$log_file" | sed 's/^/    /'
  fi
done

echo ""
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo " Pipeline: ALL STAGES PASSED"
else
  echo " Pipeline: FAILED stages: ${FAILED[*]}"
fi
echo " Logs: $PIPELINE_LOG_DIR/"
echo "========================================"

[ ${#FAILED[@]} -eq 0 ]
