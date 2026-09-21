#!/bin/bash
# Sequential Idea 1 / 2 / 3 year-episode 20M trains (trailing 7% DD).
# Run on the GPU node after srun (see scripts/run_idea123_20m_tmux.sh).
# Do not pass --mvp. Do not scancel.
set -euo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
cd "$REPO"
mkdir -p outputs

arch="$(uname -m)"
if [[ "$arch" == "aarch64" ]]; then
  # GH200: project .venv is aarch64.
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

python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python - <<'PY'
from quant_rl.config import load_config

cfg = load_config()
td = float(cfg.ftmo.trailing_dd_limit)
if abs(td - 0.07) > 1e-12:
    raise SystemExit(f"trailing_dd_limit={td}, expected 0.07")
print(
    "ftmo trailing_dd_limit=%.2f soft=%.2f daily_loss_limit=%.0f max_loss_limit=%.0f"
    % (
        td,
        float(cfg.ftmo.soft_trailing_dd_limit),
        float(cfg.ftmo.daily_loss_limit),
        float(cfg.ftmo.max_loss_limit),
    )
)
PY

echo "=== Idea 1 PO3/IFVG 20M ==="
python -u -m quant_rl.train.train_rl \
  --config config/features_full_po3_mtf.yaml \
  --strategy po3_ifvg \
  --seed 50 \
  --out outputs \
  features.include_session_ohlc=true \
  features.liquidity.enabled=true \
  features.po3_state_mtf.enabled=true \
  features.ifvg_mtf.enabled=true \
  env.n_envs=64 \
  2>&1 | tee outputs/idea1_20m_trailing_dd.log

echo "=== Idea 2 distribution 20M ==="
python -u -m quant_rl.train.train_rl \
  --config config/idea2_distribution.yaml \
  --strategy distribution \
  --seed 50 \
  --out outputs \
  env.n_envs=64 \
  2>&1 | tee outputs/idea2_20m_trailing_dd.log

echo "=== Idea 3 unconstrained baseline 20M ==="
python -u -m quant_rl.train.train_rl \
  --config quant_rl/config/default.yaml \
  --seed 50 \
  --out outputs \
  env.n_envs=64 \
  2>&1 | tee outputs/idea3_20m_trailing_dd.log

echo "=== all three arms finished ==="
