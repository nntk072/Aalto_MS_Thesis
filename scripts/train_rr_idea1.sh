#!/bin/bash
# Idea 1 (po3_ifvg), one TCN, 20M steps. Runs INSIDE its own gh200:1 allocation
# started by scripts/run_rr_two_gpu_tmux.sh (window idea1, node gpuarm1).
#
# Sizing is FIXED at $500/trade, so 1R == $500 at every stop width and cumulative
# R is a clean objective. Under the default fractional sizing 1R drifts with
# equity ($450 at $90k, $600 at $120k) and the reward would partly pay for
# position size instead of decisions.
#
# reward_mode=rr: each closed trade contributes its realized R. Losing trades
# cost -1R, so the agent is pushed toward hit rate rather than size.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"

# shellcheck source=/dev/null
source "$REPO/scripts/triton/activate_venv.sh"

export PYTHONUNBUFFERED=1
export QUANT_RL_MAX_N_ENVS="${QUANT_RL_MAX_N_ENVS:-32}"

STEPS="${STEPS:-20000000}"
SEED="${SEED:-50}"
ARCH="${ARCH:-tcn}"
FIXED_RISK="${FIXED_RISK:-500}"
# Safety cap on trades per session, not a sizing rule: sizing is already fixed,
# so 5 trades risk at most $2,500 against the $5,000 FTMO daily limit.
MAX_ENTRIES="${MAX_ENTRIES:-5}"

echo "=== Idea 1 po3_ifvg | $ARCH | ${STEPS} steps | 1R = \$$FIXED_RISK | reward=rr ==="
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

exec python -u -m quant_rl.train.train_rl \
  --config config/features_full_po3_mtf.yaml \
  --strategy po3_ifvg \
  --seed "$SEED" \
  --arch "$ARCH" \
  --out outputs \
  features.include_session_ohlc=true \
  features.liquidity.enabled=true \
  features.po3_state_mtf.enabled=true \
  features.ifvg_mtf.enabled=true \
  env.n_envs="$QUANT_RL_MAX_N_ENVS" \
  env.reward_mode=rr \
  env.risk_mode=fixed \
  env.allow_ema_exit=false \
  env.fixed_risk_usd="$FIXED_RISK" \
  env.max_entries_per_session="$MAX_ENTRIES" \
  ppo.total_timesteps="$STEPS"
