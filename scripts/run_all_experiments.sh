#!/usr/bin/env bash
# Run the full experiment pipeline end-to-end (PLAN 6 + PLAN 7).
# Usage: BARS=data/us100_2025.csv FEATURES=... ./scripts/run_all_experiments.sh
set -euo pipefail

BARS="${BARS:?set BARS=/path/to/bars.csv}"
FEATURES="${FEATURES:-$BARS}"
SEEDS="${SEEDS:-42 43 44 45 46}"
STEPS="${STEPS:-100000}"
MODEL="${MODEL:-models/rl_runs/best/model.zip}"
OOS_BARS="${OOS_BARS:-data/us100_2026.csv}"
OOS_FEATURES="${OOS_FEATURES:-$OOS_BARS}"

echo "=== 1/5 Baselines ==="
for s in buy_hold ema_macd_rsi breakout; do
    python scripts/run_baseline.py --strategy "$s" --bars-csv "$BARS" \
        --features-csv "$FEATURES" --out "results/baselines/$s.json"
done

echo "=== 2/5 Encoder comparison ==="
python scripts/compare_encoders.py --bars-csv "$BARS" --features-csv "$FEATURES" \
    --steps "$STEPS" --out results/encoder_comparison.json

echo "=== 3/5 Walk-forward validation ==="
python scripts/run_walk_forward.py --bars-csv "$BARS" --features-csv "$FEATURES" \
    --steps "$STEPS" --seeds $SEEDS --out results/walk_forward.json

echo "=== 4/5 Ablation matrix ==="
python scripts/ablation_runner.py --bars-csv "$BARS" --features-csv "$FEATURES" \
    --steps "$STEPS" --out-dir results/ablations

echo "=== 5/5 OOS sensitivity ==="
python scripts/test_oos.py --model-path "$MODEL" --bars-csv "$OOS_BARS" \
    --features-csv "$OOS_FEATURES" --out results/oos_report.json

echo "All experiments complete. See results/."
