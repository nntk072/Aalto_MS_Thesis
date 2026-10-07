#!/bin/bash
# Idea 2 (distribution), one TCN, 20M steps. Runs INSIDE its own gh200:1
# allocation started by scripts/run_rr_two_gpu_tmux.sh (window idea2, node gpuarm2).
#
# Identical sizing/reward policy to Idea 1: fixed $500/trade so 1R == $500, and
# reward_mode=rr so the objective is cumulative realized R rather than Sharpe or
# dollar PnL. Only the strategy differs, so the two runs are comparable.
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
MAX_ENTRIES="${MAX_ENTRIES:-5}"

echo "=== Idea 2 distribution | $ARCH | ${STEPS} steps | 1R = \$$FIXED_RISK | reward=rr ==="
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

exec python -u -m quant_rl.train.train_rl \
  --config config/features/full_po3_mtf.yaml \
  --strategy distribution \
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
