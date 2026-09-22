#!/bin/bash
# One GH200 slot: merged overlay, then the unconstrained baseline, 20M each.
# ARCH selects the encoder (tcn, gru, or transformer). Default is tcn.
# PPO trailing drawdown is a training-only fail. Reported train and test
# rolls follow FTMO. Do not pass --mvp. Do not scancel.
# Launch from the login node with scripts/run_encoder_slot_tmux.sh.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
mkdir -p outputs

machine="$(uname -m)"
if [[ "$machine" == "aarch64" ]]; then
  # shellcheck source=/dev/null
  source "$REPO/.venv/bin/activate"
elif [[ -n "${X86_VENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${X86_VENV}/bin/activate"
else
  echo "ERROR: x86 node — set X86_VENV to an x86 venv; do not source GH200 .venv" >&2
  exit 1
fi

export QUANT_RL_MAX_N_ENVS="${QUANT_RL_MAX_N_ENVS:-64}"
export PYTHONUNBUFFERED=1
ENCODER="${ARCH:-tcn}"
echo "encoder=$ENCODER"

python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python - <<'PY'
from quant_rl.config import load_config

cfg = load_config()
td = float(cfg.ftmo.trailing_dd_limit)
soft = float(cfg.ftmo.soft_trailing_dd_limit)
if abs(td - 0.07) > 1e-12:
    raise SystemExit(f"trailing_dd_limit={td}, expected 0.07")
if abs(soft) > 1e-12:
    raise SystemExit(f"soft_trailing_dd_limit={soft}, expected 0 (hard peak DD only)")
ith = float(cfg.env.entry_intensity_threshold)
if abs(ith) > 1e-12:
    raise SystemExit(f"entry_intensity_threshold={ith}, expected 0 (no intensity hold band)")
print(
    "ftmo trailing_dd_limit=%.2f soft=%.2f daily_loss_limit=%.0f max_loss_limit=%.0f "
    "log_std_min=%.2f entry_intensity_threshold=%.3f"
    % (
        td,
        soft,
        float(cfg.ftmo.daily_loss_limit),
        float(cfg.ftmo.max_loss_limit),
        float(cfg.ppo.log_std_min),
        float(cfg.env.entry_intensity_threshold),
    )
)
PY

echo "=== overlay 20M encoder=$ENCODER ==="
python -u -m quant_rl.train.train_rl \
  --config config/features_full_po3_mtf.yaml \
  --strategy po3_ifvg \
  --seed 50 \
  --arch "$ENCODER" \
  --out outputs \
  features.include_session_ohlc=true \
  features.liquidity.enabled=true \
  features.po3_state_mtf.enabled=true \
  features.ifvg_mtf.enabled=true \
  env.n_envs=64 \
  2>&1 | tee "outputs/overlay_${ENCODER}_20m.log"

# SKIP_BASELINE=1 trains the overlay only. SKIP_IDEA3 is the old name.
if [[ "${SKIP_BASELINE:-${SKIP_IDEA3:-0}}" != "1" ]]; then
  echo "=== baseline 20M encoder=$ENCODER ==="
  python -u -m quant_rl.train.train_rl \
    --config quant_rl/config/default.yaml \
    --seed 50 \
    --arch "$ENCODER" \
    --out outputs \
    env.n_envs=64 \
    2>&1 | tee "outputs/baseline_${ENCODER}_20m.log"
else
  echo "=== skip baseline ==="
fi

echo "=== overlay and baseline finished ==="
