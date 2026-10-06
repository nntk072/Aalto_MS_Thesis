#!/usr/bin/env bash
# Re-export the two Oct 3 runs from their saved PPO checkpoints.
#
# Both runs completed 20M PPO steps but were killed inside save_run() while
# rendering charts: Idea 1 never reached its testing split and Idea 2 stopped
# part-way through it, so neither wrote training_log.json. The checkpoints are
# intact, so this re-evaluates ppo_final.zip instead of retraining.
#
# Chart volume is deliberately small. A per-trade chart costs ~20s of CPU and
# HTML roughly doubles that; 50 per split keeps the job far inside the
# allocation. HTML is off, matching output.save_html in default.yaml.
#
# This is CPU work (env rollout + matplotlib). It runs on a GPU node only
# because the project venv is arch-specific: aarch64 -> .venv, and the login
# node is x86_64.
#
# Usage:
#   sbatch scripts/triton/reexport_oct3_runs.sh
#   MAX_CHARTS=200 bash scripts/triton/reexport_oct3_runs.sh   # inside a shell
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
RUNS="${RUNS:-20261003_171903_512479_rl_train_seed50_tcn 20261003_171903_426375_rl_train_seed50_tcn}"
MAX_CHARTS="${MAX_CHARTS:-50}"
LOGDIR="${LOGDIR:-$REPO/outputs}"

cd "$REPO"
# shellcheck source=/dev/null
source scripts/triton/activate_venv.sh

for run in $RUNS; do
  dir="$REPO/outputs/$run"
  if [[ ! -d "$dir" ]]; then
    echo "SKIP $run: run directory not found"
    continue
  fi
  if [[ ! -f "$dir/model/ppo_final.zip" ]]; then
    echo "SKIP $run: no model/ppo_final.zip"
    continue
  fi
  echo "=== re-export $run (max_charts=$MAX_CHARTS, no HTML) ==="
  python -u -m quant_rl.eval.eval_run \
    --run "$dir" \
    --no-save-html \
    --max-charts "$MAX_CHARTS" \
    2>&1 | tee "$LOGDIR/reexport_${run}.log"
done

echo "All re-exports finished."
