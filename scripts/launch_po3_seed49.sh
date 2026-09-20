#!/bin/bash
set -euo pipefail
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
source .venv/bin/activate
export QUANT_RL_MAX_N_ENVS=32
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
exec python -u -m quant_rl.train.train_rl \
  --config config/features_full_po3_mtf.yaml \
  --strategy po3_ifvg \
  --seed 49 \
  --out outputs \
  features.include_session_ohlc=true \
  features.liquidity.enabled=true \
  features.po3_state_mtf.enabled=true \
  features.ifvg_mtf.enabled=true \
  env.n_envs=32
