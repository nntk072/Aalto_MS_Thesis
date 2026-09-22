#!/bin/bash
# Single 20M train for the encoder matrix. Requires MODE and ARCH.
# MODE=overlay|baseline  ARCH=tcn|gru|transformer
# CUDA_VISIBLE_DEVICES is set by the orchestrator. Do not scancel.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
mkdir -p outputs

MODE="${MODE:-}"
ARCH="${ARCH:-}"
if [[ -z "$MODE" || -z "$ARCH" ]]; then
  echo "ERROR: MODE and ARCH are required (MODE=overlay|baseline ARCH=tcn|gru|transformer)" >&2
  exit 1
fi
if [[ "$MODE" != "overlay" && "$MODE" != "baseline" ]]; then
  echo "ERROR: MODE must be overlay or baseline, got: $MODE" >&2
  exit 1
fi
case "$ARCH" in
  tcn | gru | transformer) ;;
  *)
    echo "ERROR: ARCH must be tcn, gru, or transformer, got: $ARCH" >&2
    exit 1
    ;;
esac

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

echo "mode=$MODE encoder=$ARCH cuda_visible=${CUDA_VISIBLE_DEVICES:-all}"
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

LOG="outputs/${MODE}_${ARCH}_20m.log"
echo "=== ${MODE} 20M encoder=$ARCH log=$LOG ==="

if [[ "$MODE" == "overlay" ]]; then
  python -u -m quant_rl.train.train_rl \
    --config config/features_full_po3_mtf.yaml \
    --strategy po3_ifvg \
    --seed 50 \
    --arch "$ARCH" \
    --out outputs \
    features.include_session_ohlc=true \
    features.liquidity.enabled=true \
    features.po3_state_mtf.enabled=true \
    features.ifvg_mtf.enabled=true \
    env.n_envs=64 \
    2>&1 | tee "$LOG"
else
  python -u -m quant_rl.train.train_rl \
    --config quant_rl/config/default.yaml \
    --seed 50 \
    --arch "$ARCH" \
    --out outputs \
    env.n_envs=64 \
    2>&1 | tee "$LOG"
fi

echo "=== ${MODE} 20M encoder=$ARCH finished ==="
