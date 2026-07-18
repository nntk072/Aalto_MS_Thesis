# Running the RL Structure-Based SL/TP/Lot Sizing Project

## Setup

```bash
cd /home/l2nguyen/Aalto_MS_Thesis
source $HOME/.local/bin/env  # Activate environment
# Or: source .venv/bin/activate
```

## Phase Testing & Validation

### 1. Run Unit Tests (Structure, Risk, MACD Baseline)

```bash
# Test structure features (causal swing detection)
.venv/bin/python -m pytest tests/test_structure.py -v

# Test risk sizing (SL/TP/lots computation)
.venv/bin/python -m pytest tests/test_risk.py -v

# Test MACD baseline strategy
.venv/bin/python -m pytest tests/test_macd_baseline.py -v

# Run all together
.venv/bin/python -m pytest tests/test_structure.py tests/test_risk.py tests/test_macd_baseline.py -v
```

**Expected output:**
- 2 structure tests passing
- 10 risk tests passing
- 7 MACD baseline tests passing

### 1b. Run Ruff and mypy

```bash
# Install or refresh the dev environment
uv sync --extra dev

# Lint the checked package scope
uv run ruff check quant_rl

# Type-check the full checked package
uv run mypy quant_rl
```

---

## Phase 4: Baseline Validation

### 2. Run MACD Baseline Strategy

The MACD baseline implements a true crossover strategy with the following rules:

#### Indicators:
- **MACD Line:** EMA(12) - EMA(26) on close
- **Signal Line:** SMA(9) on MACD line (not EMA — this is important)
- **Trend Filter:** EMA(50) on close

#### Entry / Exit Rules:
1. **Long entry:** `close > EMA50` **AND** bullish MACD cross
   - Bullish cross: MACD crosses above signal (macd[t-1] ≤ signal[t-1], macd[t] > signal[t])
2. **Long exit:** Bearish MACD cross only (goes flat, does NOT flip to short)
3. **Short entry:** `close < EMA50` **AND** bearish MACD cross
4. **Short exit:** Bullish MACD cross only (goes flat, does NOT flip to long)
5. **Cooldown:** After any exit, wait ≥5 M1 bars (5 minutes) before next entry
6. **No SL/TP:** Baseline uses no stop loss or take profit levels

#### Per-trade Charts (All Strategies):
Every trade chart (PNG and HTML) displays a 2-panel layout:
- **Top panel:** Candlesticks with EMA50 overlay, entry/exit arrows, MAE/MFE/SL/TP lines
- **Bottom panel:** MACD line, signal line, histogram (green if positive, red if negative)

#### Run commands:

```bash
# Quick test (no charts saved, uses default config)
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd --no-save

# Full run with chart generation
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd

# View generated charts
ls -1 outputs/baseline_macd_seed*/test/orders/trade_*.png | head -5
open outputs/baseline_macd_seed42/test/orders/trade_0001_*.png  # View first trade

# Run other baselines for comparison
.venv/bin/python -m quant_rl.train.run_baselines --strategy ema
.venv/bin/python -m quant_rl.train.run_baselines --strategy rsi
```

**What this does:**
- Executes MACD crossover policy (no RL model)
- Generates per-trade charts (PNG + HTML) showing:
  - Candlesticks, EMA50 line, entry/exit markers
  - MACD, signal, histogram in bottom panel
  - Trade info (direction, open/close prices, volume, duration)
- Typical output: 80-150 trades per split with realistic hold times
- Verifies order chart layout works correctly

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

echo "=== Running Ruff ==="
uv run ruff check quant_rl

echo "=== Running mypy ==="
uv run mypy quant_rl

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
