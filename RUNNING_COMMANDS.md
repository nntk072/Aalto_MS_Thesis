# Running the RL Structure-Based SL/TP/Lot Sizing Project

## Setup

```bash
cd /home/l2nguyen/Aalto_MS_Thesis
source $HOME/.local/bin/env  # Activate environment
# Or: source .venv/bin/activate
```

## Phase Testing & Validation

### 1. Run Unit Tests (Structure & Risk)

```bash
# Test structure features (causal swing detection)
.venv/bin/python -m pytest tests/test_structure.py -v

# Test risk sizing (SL/TP/lots computation)
.venv/bin/python -m pytest tests/test_risk.py -v

# Run both together
.venv/bin/python -m pytest tests/test_structure.py tests/test_risk.py -v
```

**Expected output:**
- 2 structure tests passing
- 10 risk tests passing

---

## Phase 4: Baseline Validation

### 2. Run MACD Baseline Strategy

```bash
# Quick test (uses default config)
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd --no-save

# Full run with chart generation
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd

# Run EMA baseline
.venv/bin/python -m quant_rl.train.run_baselines --strategy ema
```

**What this does:**
- Runs MACD momentum-based policy with structure SL/TP
- Generates per-trade charts (PNG + HTML) showing realistic hold times
- Verifies structure SL/TP enforcement works correctly

**Output:** `outputs/baseline_macd_seed42/`
- `training/orders/trade_*.png` — Per-trade charts with SL/TP levels
- `training/orders/trade_*.html` — Interactive charts
- Metrics: Sharpe ratio, max drawdown, trade count

---

## Phase 3: RL Training

### 3a. Run Quick MVP (Minimal Viable Product) Training

```bash
# Quick 8k timestep training on last 30 days of data
.venv/bin/python -m quant_rl.train.train_rl --mvp --seed=42

# With custom override
.venv/bin/python -m quant_rl.train.train_rl --mvp training.max_days_mvp=15 --seed=42
```

### 3b. Run Full RL Training

```bash
# Full training (500k timesteps on all data)
.venv/bin/python -m quant_rl.train.train_rl --seed=42

# Save model and generate test charts
.venv/bin/python -m quant_rl.train.train_rl --seed=42
```

**Output:**
- `outputs/ppo_model_seed42` — Trained PPO policy
- `outputs/20260718_*/testing/orders/trade_*.png` — Evaluation charts

---

## Phase 2: Full Backtest with Random Policy

### 4. Run Random Policy Backtest (baseline for comparison)

```bash
# Quick test
.venv/bin/python -m quant_rl.train.run_backtest --no-save

# Full run with chart export
.venv/bin/python -m quant_rl.train.run_backtest --seed=42

# Test on specific date range
.venv/bin/python -m quant_rl.train.run_backtest data.split.train_end=2025-11-30 --seed=42
```

**Output:**
- `outputs/random_seed42/`
- Per-trade charts with MAE/MFE/SL/TP overlays
- Equity curve, metrics summary

---

## Advanced: Custom Configurations

### Override Config via Command Line

```bash
# Change max loss per trade
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd \
  backtest.validation.max_loss_per_trade_usd=50.0

# Change initial balance
.venv/bin/python -m quant_rl.train.run_backtest \
  account.initial_balance=50000.0

# Change risk settings
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd \
  risk.default_risk_frac=0.02 \
  risk.swing_buffer_pts=2.0

# Combine multiple overrides
.venv/bin/python -m quant_rl.train.train_rl --mvp \
  account.initial_balance=100000.0 \
  env.obs_window=30 \
  ppo.total_timesteps=16384
```

---

## Full Testing Pipeline (Run in Sequence)

```bash
#!/bin/bash
cd /home/l2nguyen/Aalto_MS_Thesis

echo "=== Running Unit Tests ==="
.venv/bin/python -m pytest tests/test_structure.py tests/test_risk.py -v

echo "=== Running MACD Baseline ==="
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd

echo "=== Running MVP RL Training ==="
.venv/bin/python -m quant_rl.train.train_rl --mvp --seed=42

echo "=== All Done! ==="
```

---

## Project Structure

```
quant_rl/
├── backtest/
│   ├── engine.py          ← Per-trade SL/TP enforcement
│   ├── risk.py            ← NEW: SL/TP/lots computation
│   └── broker.py          ← Extended: Position.sl_price/tp_price
├── features/
│   ├── structure.py       ← NEW: Swing detection
│   └── build.py           ← Updated: wire structure features
├── envs/
│   ├── trading_env.py     ← NEW: Gymnasium environment
│   └── reward.py          ← DSR reward
├── train/
│   ├── run_backtest.py    ← Updated: structure params
│   ├── run_baselines.py   ← NEW: MACD/EMA baseline
│   └── train_rl.py        ← NEW: PPO training
├── eval/
│   ├── trade_metrics.py   ← Updated: prefer per-trade SL/TP
│   ├── plots.py           ← Per-trade PNG charts
│   └── plots_interactive.py ← Per-trade HTML charts
└── config/
    └── default.yaml       ← Updated: max_loss=100, risk section

tests/
├── test_structure.py      ← NEW: 2 tests
└── test_risk.py           ← NEW: 10 tests
```

---

## Debugging & Monitoring

### Check Backtest Results

```bash
# List latest run
ls -lah outputs/ | tail -5

# Check metrics
.venv/bin/python -c "
import pandas as pd
import glob
latest = sorted(glob.glob('outputs/*/'))[-1]
metrics_file = latest + '/test/metrics.json'
df = pd.read_json(metrics_file, orient='index')
print(df)
"

# Inspect trade log
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('outputs/LATEST/test/trades.csv')
print(df.head())
print(f'SL/TP present: {df[\"sl_price\"].notna().sum()}/{len(df)}')
"
```

### View Generated Charts

```bash
# List per-trade charts
ls -1 outputs/LATEST/test/orders/trade_*.png | head -10

# Open in image viewer
eog outputs/LATEST/test/orders/trade_0001*.png &

# Open HTML in browser
firefox outputs/LATEST/test/orders/trade_0001*.html &
```

---

## Key Verification Points

After running each phase, verify:

✓ **Unit Tests**: All 12 tests pass (2 structure + 10 risk)
✓ **Config**: `max_loss_per_trade_usd=100.0`, risk section loaded
✓ **Features**: `last_swing_low_price`, `last_swing_high_price` present
✓ **Baseline**: MACD trades complete with realistic durations (5-60+ min)
✓ **Trade Log**: `sl_price`, `tp_price`, `last_swing_low/high` columns populated
✓ **Charts**: SL/TP lines visible on per-trade overlays
✓ **RL Training**: Model saves successfully to `outputs/ppo_model_*`

---

## Performance Expectations

- **Unit tests**: < 1 second
- **Baseline MACD**: 5-10 seconds (small subset: 10k train, 5k test bars)
- **MVP RL training**: 30-60 seconds (8k timesteps)
- **Full RL training**: 10+ minutes (500k timesteps)
- **Chart generation**: ~100 charts = 30-60 seconds

---

## Troubleshooting

### "Module not found" errors
```bash
# Rebuild feature cache (forces recomputation)
.venv/bin/python -m quant_rl.train.run_backtest --force

# Clean cache
rm -rf cache/*.parquet
```

### Charts not generated
```bash
# Check output directory
ls -lah outputs/*/test/orders/

# Regenerate with explicit output
.venv/bin/python -m quant_rl.train.run_backtest --out=outputs --no-save false
```

### Training hangs/OOM
```bash
# Reduce timesteps for MVP
.venv/bin/python -m quant_rl.train.train_rl --mvp training.total_timesteps_mvp=4096

# Reduce observation window
.venv/bin/python -m quant_rl.train.train_rl --mvp env.obs_window=30
```

---

## Next Steps

1. **Run tests** to verify correctness
2. **Run baselines** to see realistic hold times
3. **Train RL agent** on MVP data first, then full data
4. **Analyze charts** to confirm SL/TP behavior
5. **Compare strategies**: random vs baseline vs trained agent

---

For questions or debugging, check the plan: `/home/l2nguyen/.cursor/plans/trade_volume_duration_fix_10741966.plan.md`
