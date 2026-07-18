# Running the Full Project - Complete Guide

This document shows how to verify and run all components of the RL Structure-Based SL/TP/Lot Sizing implementation.

## Prerequisites

```bash
cd /home/l2nguyen/Aalto_MS_Thesis
source $HOME/.local/bin/env  # or: source ~/.venv/bin/activate
```

---

## 1. Run Unit Tests (Verify Structure & Risk Sizing)

```bash
# Test structure features (swing detection)
pytest tests/test_structure.py -v

# Test risk sizing (SL/TP computation, lot sizing)
pytest tests/test_risk.py -v

# Run both together
pytest tests/test_structure.py tests/test_risk.py -v
```

**Expected output:**
- All 12 tests should PASS
- Tests verify:
  - Causal swing detection (no lookahead)
  - SL/TP price computation
  - Lot sizing from equity and SL distance
  - Risk capping by USD limits

---

## 2. Run Baseline Strategies (MACD/EMA)

These validate that the engine holds trades for reasonable durations with structure-aware SL/TP.

### MACD Strategy
```bash
python -m quant_rl.train.run_baselines --strategy macd --seed=42
```

### EMA Strategy
```bash
python -m quant_rl.train.run_baselines --strategy ema --seed=42
```

**Output:**
- `outputs/baseline_macd_seed42/` or `outputs/baseline_ema_seed42/`
- Contains:
  - `training/` and `testing/` subdirectories
  - `orders/` with per-trade PNG and HTML charts
  - Metrics: Sharpe, MaxDD, Trade count, Return

**What to verify in charts:**
- Volume varies (not always 1.00 lot)
- Trade duration: 5-60+ minutes (not 1-2m)
- SL lines at swing low/high levels
- TP lines at entry ± RR ratio × risk distance

---

## 3. Run Random Policy Backtest (Original)

For comparison, run the original random policy:

```bash
python -m quant_rl.train.run_backtest --seed=42
```

**Output:**
- `outputs/random_seed42/`
- Charts show structure-based SL/TP even with random entries

---

## 4. Train PPO Agent (Full)

Train a PPO policy with structure features:

```bash
# Full training (default: 500k timesteps)
python -m quant_rl.train.train_rl --seed=42

# Or MVP mode (quick iteration: 8k timesteps on recent 30 days)
python -m quant_rl.train.train_rl --mvp --seed=42
```

**Output:**
- Model saved: `outputs/ppo_model_seed42`
- Test evaluation results logged
- Expected: Better Sharpe, longer hold times, variable lot sizes

---

## 5. Full End-to-End Verification Script

Run all verifications in sequence:

```bash
# 1. Tests
echo "=== Unit Tests ==="
pytest tests/test_structure.py tests/test_risk.py -q

# 2. Baseline MACD
echo -e "\n=== Baseline MACD ==="
python -m quant_rl.train.run_baselines --strategy macd --no-save --seed=42

# 3. Random policy
echo -e "\n=== Random Policy ==="
python -m quant_rl.train.run_backtest --seed=42 --no-save

echo -e "\n=== All verifications complete ==="
```

---

## 6. Inspect Generated Charts

After running any strategy, inspect per-trade charts:

```bash
# List all generated charts
ls -lah outputs/*/training/orders/trade_*.png | head -10
ls -lah outputs/*/testing/orders/trade_*.png | head -10

# Open a specific chart in your viewer
open outputs/baseline_macd_seed42/testing/orders/trade_0001*.png

# Interactive HTML charts
open outputs/baseline_macd_seed42/testing/orders/trade_0001*.html
```

---

## 7. Configuration & Fine-Tuning

All settings in `quant_rl/config/default.yaml`:

```yaml
backtest:
  validation:
    max_loss_per_trade_usd: 100.0    # Structure SL primary, this is safety cap
    take_profit_per_trade_usd: null  # TP from structure + rr_ratio

risk:
  min_lot: 0.01                      # Min position size
  max_lot: 100.0                     # Max position size
  default_risk_frac: 0.01            # 1% equity risk default
  min_sl_atr_mult: 0.5               # Reject stops tighter than 0.5*ATR
  swing_buffer_pts: 1.0              # Buffer beyond swing price
  rr_ratio_default: 2.0              # Default Risk:Reward ratio
```

---

## 8. Key Files & Their Roles

| File | Purpose |
|------|---------|
| `quant_rl/features/structure.py` | Causal swing detection for SL/TP |
| `quant_rl/backtest/risk.py` | SL/TP price computation, lot sizing |
| `quant_rl/backtest/engine.py` | Backtest engine with per-trade SL/TP enforcement |
| `quant_rl/envs/trading_env.py` | Gymnasium environment for RL (hybrid action space) |
| `quant_rl/train/train_rl.py` | PPO training loop |
| `quant_rl/train/run_baselines.py` | MACD/EMA baseline strategies |
| `quant_rl/train/run_backtest.py` | Run backtest with any policy |
| `quant_rl/eval/trade_metrics.py` | Compute MAE/MFE/SL/TP per trade |
| `quant_rl/eval/export.py` | Export run results + generate charts |
| `tests/test_structure.py` | Unit tests for swing detection |
| `tests/test_risk.py` | Unit tests for risk sizing |

---

## 9. Expected Results Summary

### Unit Tests
```
✓ test_compute_structure_features
✓ test_structure_features_no_lookahead
✓ test_long_sl_from_swing_low
✓ test_short_sl_from_swing_high
✓ test_long_sl_fallback_no_swing
✓ test_short_sl_fallback_no_swing
✓ test_long_tp_price
✓ test_short_tp_price
✓ test_basic_lot_calculation
✓ test_lot_capped_by_max_loss
✓ test_lot_clipping_min_max
✓ test_zero_sl_distance
```

### Baseline MACD Metrics (Training)
```
Sharpe: ~-5 to -6
MaxDD: ~10%
Trades: ~8000-10000
Return: ~-10%
```

Note: Negative returns are expected with random-like MACD on this data; the important thing is:
- Trades execute with varying lot sizes
- Hold times are 5-60+ minutes (realistic)
- SL/TP levels shown on charts

### Per-Trade Charts
- **Entry marker**: Green arrow (long) or green downward (short)
- **Exit marker**: Red arrow (long) or red upward (short)
- **SL line**: Red dashed horizontal
- **TP line**: Green dashed horizontal
- **MAE line**: Red dotted (worst price during trade)
- **MFE line**: Green dotted (best price during trade)
- **Trade info box**: Direction, Open, Close, Volume (lots), Duration

---

## 10. Troubleshooting

### "Connection to Cursor server failed" / WSL issues
```bash
# Check WSL memory and resources
free -h
df -h

# Or increase WSL memory in ~/.wslconfig
[wsl2]
memory=16GB
processors=8
```

### Tests fail: "ImportError: No module named quant_rl"
```bash
# Add project to path
export PYTHONPATH=/home/l2nguyen/Aalto_MS_Thesis:$PYTHONPATH
pytest tests/test_structure.py -v
```

### "Data not found" error
```bash
# Ensure data files exist
ls -la data/US100.cash_M1_*.csv
ls -la data/US500.cash_M1_*.csv

# Or force rebuild
python -m quant_rl.train.run_backtest --force
```

---

## 11. Next Steps

1. **Verify everything works**: Run unit tests → baseline strategies → random backtest
2. **Inspect charts**: Verify SL/TP levels, volumes, durations match expectations
3. **Train PPO**: Once satisfied with baselines, run `train_rl.py` for RL agent
4. **Compare results**: Random vs MACD vs PPO side-by-side
5. **Thesis integration**: Use generated metrics + charts in thesis report

---

## Quick Start (Copy-Paste Ready)

```bash
cd /home/l2nguyen/Aalto_MS_Thesis

# 1. Activate environment
source $HOME/.local/bin/env

# 2. Run all tests
echo "Testing..." && pytest tests/test_structure.py tests/test_risk.py -q

# 3. Run MACD baseline
echo "MACD baseline..." && python -m quant_rl.train.run_baselines --strategy macd --seed=42 --no-save

# 4. Run random policy
echo "Random policy..." && python -m quant_rl.train.run_backtest --seed=42 --no-save

# 5. Check outputs
ls -la outputs/ | tail -5
```

---

Done! All components of the RL Structure-Based SL/TP/Lot Sizing plan are now integrated and ready to verify.
