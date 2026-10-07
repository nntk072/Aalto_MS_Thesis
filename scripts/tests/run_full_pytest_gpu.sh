#!/bin/bash
# Full pytest on ONE gh200:1 allocation (run inside that allocation).
#
# Split by marker so a slow or hanging group cannot hide the rest:
#   fast  -> the pure-CPU unit suites (fast feedback on sizing/reward/env)
#   slow  -> model-training smoke tests (SAC/PPO/LSTM), minutes each
# The CPU tests are expected to PASS. The slow group contains known failures
# that also reproduce on a clean HEAD, so treat it as informational.
set -uo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
source "$REPO/scripts/triton/activate_venv.sh"

LOG_DIR="${LOG_DIR:-$REPO/outputs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=========================================================="
echo "node=$(hostname)  $(date '+%F %T')"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
echo "=========================================================="

# -p no:cacheprovider: keep the tree clean on a shared scratch dir.
# --timeout: fail a stuck test instead of hanging the 6h allocation.
COMMON=(-q -p no:cacheprovider --timeout=300 -rf --tb=line)

echo
echo "########## GROUP 1/2: fast CPU unit suites ##########"
python -m pytest "${COMMON[@]}" \
  tests/test_envs tests/test_risk.py tests/test_rollout.py tests/test_reward.py \
  tests/test_guardrails.py tests/test_backtest tests/test_data tests/test_features \
  tests/test_causal_features.py tests/test_config_typo_safety.py \
  tests/test_cost_model.py tests/test_session_filter.py tests/test_structure.py \
  tests/test_trade_metrics.py tests/test_data_split.py tests/test_ticks.py \
  tests/test_macd_baseline.py tests/test_plots_export.py tests/test_report_g3.py \
  tests/test_training_plots.py tests/test_orchestration_weekly.py \
  tests/test_eval tests/test_utils tests/test_validation tests/test_orchestra \
  2>&1 | tee "$LOG_DIR/pytest_fast_$STAMP.log"
FAST_RC=${PIPESTATUS[0]}
echo "GROUP 1 rc=$FAST_RC"

echo
echo "########## GROUP 2/2: model training smoke (slow) ##########"
python -m pytest "${COMMON[@]}" \
  tests/test_train tests/test_integration tests/test_models \
  tests/test_baselines tests/test_evaluation tests/test_live \
  2>&1 | tee "$LOG_DIR/pytest_slow_$STAMP.log"
SLOW_RC=${PIPESTATUS[0]}
echo "GROUP 2 rc=$SLOW_RC"

echo
echo "=========================================================="
echo "FAST (expect pass) rc=$FAST_RC"
echo "SLOW (known pre-existing failures) rc=$SLOW_RC"
echo "logs: $LOG_DIR/pytest_fast_$STAMP.log  $LOG_DIR/pytest_slow_$STAMP.log"
echo "finished $(date '+%F %T')"
echo "=========================================================="