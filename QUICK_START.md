# Quick Terminal Reference - RL Structure SL/TP/Lot Sizing

## Setup (One Time)
```bash
cd /home/l2nguyen/Aalto_MS_Thesis
source $HOME/.local/bin/env  # Activate environment
```

## Run Tests (1 minute)
```bash
# All tests
.venv/bin/python -m pytest tests/test_structure.py tests/test_risk.py -v

# Individual test files
.venv/bin/python -m pytest tests/test_structure.py -v
.venv/bin/python -m pytest tests/test_risk.py -v
```

**Expected:** 12/12 tests passing ✅

---

## Run Baseline (10 seconds)
```bash
# Quick (no charts saved)
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd --no-save

# Full (with chart generation)
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd

# View results
ls -1 outputs/baseline_macd_seed*/test/orders/trade_*.png | head -3
```

**Expected:** 
- Training Sharpe: -5 to -10
- Test Sharpe: -8 to -12
- Charts with SL/TP lines visible ✅

---

## Train RL Agent - MVP (1 minute)
```bash
# Quick MVP training (8k timesteps, 30 days)
.venv/bin/python -m quant_rl.train.train_rl --mvp --seed=42

# View model output
ls -1 outputs/ppo_model_seed42
```

**Expected:** Model saved, test charts generated ✅

---

## Train RL Agent - Full (10+ minutes)
```bash
# Full training (500k timesteps, all data)
.venv/bin/python -m quant_rl.train.train_rl --seed=42
```

---

## Run Backtest (5 seconds)
```bash
# Quick backtest
.venv/bin/python -m quant_rl.train.run_backtest --no-save

# Full backtest with charts
.venv/bin/python -m quant_rl.train.run_backtest --seed=42
```

---

## View Results

### List Latest Run
```bash
ls -lh outputs/ | tail -1
LATEST=$(ls -d outputs/*/ | tail -1)
echo $LATEST
```

### Check Metrics
```bash
cat outputs/LATEST/test/metrics.json
```

### View Per-Trade Charts
```bash
# PNG charts
ls -1 outputs/LATEST/test/orders/trade_*.png | head -5

# HTML charts (interactive)
ls -1 outputs/LATEST/test/orders/trade_*.html | head -5

# Open first chart
eog outputs/LATEST/test/orders/trade_0001*.png &
# or: firefox outputs/LATEST/test/orders/trade_0001*.html &
```

### Inspect Trade Log
```bash
# Show first 10 trades
head -10 outputs/LATEST/test/trades.csv

# Count trades with SL/TP
.venv/bin/python << 'EOF'
import pandas as pd
df = pd.read_csv('outputs/LATEST/test/trades.csv')
open_trades = df[df['type'] == 'open']
print(f"Total open trades: {len(open_trades)}")
print(f"With SL price: {open_trades['sl_price'].notna().sum()}")
print(f"With TP price: {open_trades['tp_price'].notna().sum()}")
print(f"\nSample trade:")
print(open_trades[['direction', 'price', 'sl_price', 'tp_price']].iloc[0])
EOF
```

---

## Configuration Overrides

### Change Max Loss Per Trade
```bash
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd \
  backtest.validation.max_loss_per_trade_usd=50.0
```

### Change Initial Balance
```bash
.venv/bin/python -m quant_rl.train.run_backtest \
  account.initial_balance=50000.0
```

### Change Risk Settings
```bash
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd \
  risk.default_risk_frac=0.02 \
  risk.swing_buffer_pts=2.0
```

### Change Observation Window
```bash
.venv/bin/python -m quant_rl.train.train_rl --mvp \
  env.obs_window=30
```

---

## Troubleshooting

### "Module not found" errors
```bash
# Rebuild feature cache (forces recomputation)
.venv/bin/python -m quant_rl.train.run_backtest --force

# Clean cache
rm -rf cache/*.parquet
```

### Training hangs or out of memory
```bash
# Reduce MVP timesteps
.venv/bin/python -m quant_rl.train.train_rl --mvp \
  training.total_timesteps_mvp=4096

# Reduce batch size
.venv/bin/python -m quant_rl.train.train_rl --mvp \
  ppo.batch_size=32
```

### No charts generated
```bash
# Check output directory exists
ls -la outputs/*/test/orders/

# Regenerate with explicit paths
.venv/bin/python -m quant_rl.train.run_backtest --out=outputs
```

---

## What Each Command Does

| Command | Purpose | Time | Output |
|---------|---------|------|--------|
| `pytest tests/test_*.py` | Validate structure & risk math | <1s | Pass/fail |
| `run_baselines --strategy macd` | Validate engine & generate charts | 10s | PNG/HTML charts |
| `train_rl --mvp` | Quick RL training on small data | 45s | PPO model + charts |
| `train_rl` (no --mvp) | Full RL training on all data | 10+ min | Production model |
| `run_backtest` | Random policy backtest for comparison | 5s | Baseline metrics |

---

## Key Files to Know

```
quant_rl/
├── features/structure.py       ← Swing detection
├── backtest/risk.py            ← SL/TP/lots formulas
├── backtest/engine.py          ← Per-trade enforcement
├── envs/trading_env.py         ← RL environment
├── train/train_rl.py           ← PPO training
├── train/run_baselines.py      ← MACD/EMA validation
└── config/default.yaml         ← All settings

tests/
├── test_structure.py           ← Swing tests
└── test_risk.py                ← Sizing tests

outputs/                         ← Results saved here
├── baseline_macd_seed42/       ← Baseline results
├── ppo_model_seed42            ← Trained model
└── 20260718_*/                 ← Run results
    ├── training/orders/        ← PNG/HTML charts
    ├── test/orders/            ← PNG/HTML charts
    └── metrics.json            ← Performance metrics
```

---

## For the Thesis Report

1. **Show SL/TP enforcement**: Charts in `outputs/baseline_macd_seed42/test/orders/`
2. **Show hold times**: Compare `random_seed42` vs `baseline_macd_seed42`
3. **Show metrics**: Sharpe, Max DD, trade count from `metrics.json`
4. **Show architecture**: Diagram in `trade_volume_duration_fix_10741966.plan.md`
5. **Show code**: Reference implementation files in repo

---

**READY TO RUN! Pick a command above and execute.** ✅
