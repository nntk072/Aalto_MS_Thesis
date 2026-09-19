#!/usr/bin/env bash
# Thin orchestration for thesis ablation / walk-forward / OOS cost smoke.
# Uses main entrypoints only (no feature/plan-6 validation package).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BARS_CSV="${BARS_CSV:-data/us100_2025.csv}"
FEATURES_CSV="${FEATURES_CSV:-}"
STEPS="${STEPS:-8192}"
SEEDS="${SEEDS:-42}"
MODEL_PATH="${MODEL_PATH:-}"
OUT_ABLATIONS="${OUT_ABLATIONS:-results/ablations}"
OUT_OOS="${OUT_OOS:-results/oos_report.json}"

echo "== ablation smoke (steps=${STEPS} seeds=${SEEDS}) =="
ABLATION_ARGS=(
  --bars-csv "$BARS_CSV"
  --experiments config/experiments.yaml
  --steps "$STEPS"
  --seeds $SEEDS
  --out-dir "$OUT_ABLATIONS"
  --variants ablation_baseline_unconditional ablation_discrete_actions
)
if [[ -n "$FEATURES_CSV" ]]; then
  ABLATION_ARGS+=(--features-csv "$FEATURES_CSV")
fi
uv run python scripts/ablation_runner.py "${ABLATION_ARGS[@]}"
uv run python scripts/report_ablations.py --ablations-dir "$OUT_ABLATIONS"

echo "== purged walk-forward (main train_rl) =="
uv run python -m quant_rl.train.train_rl --mvp --walk-forward --wf-splits 3 \
  --purge-bars 60 --embargo-bars 20 --seed 42

if [[ -n "$MODEL_PATH" ]]; then
  echo "== OOS cost sensitivity =="
  OOS_ARGS=(
    --model-path "$MODEL_PATH"
    --bars-csv "$BARS_CSV"
    --algo ppo
    --spreads 0.6 1.0
    --slippages 0.0 0.1
    --out "$OUT_OOS"
  )
  if [[ -n "$FEATURES_CSV" ]]; then
    OOS_ARGS+=(--features-csv "$FEATURES_CSV")
  fi
  uv run python scripts/test_oos.py "${OOS_ARGS[@]}"
else
  echo "skip OOS cost grid (set MODEL_PATH=outputs/<run>/model/ppo_final.zip)"
fi

echo "done."
