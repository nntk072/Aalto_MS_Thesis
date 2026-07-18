# Quick Terminal Reference - Run Everything

## Setup
```bash
cd /home/l2nguyen/Aalto_MS_Thesis
source $HOME/.local/bin/env  # Activate environment
```

---

## 1️⃣  UNIT TESTS (2-3 seconds)
```bash
pytest tests/test_structure.py tests/test_risk.py -v
```
✓ 12 tests: swing detection, SL/TP computation, lot sizing

---

## 2️⃣  BASELINE MACD STRATEGY (10-15 seconds)
```bash
python -m quant_rl.train.run_baselines --strategy macd --seed=42
```
✓ Generates per-trade charts with structure-based SL/TP
✓ Output: `outputs/baseline_macd_seed42/`

---

## 3️⃣  BASELINE EMA STRATEGY (10-15 seconds)
```bash
python -m quant_rl.train.run_baselines --strategy ema --seed=42
```
✓ Alternative baseline for comparison
✓ Output: `outputs/baseline_ema_seed42/`

---

## 4️⃣  RANDOM POLICY BACKTEST (5-10 seconds)
```bash
python -m quant_rl.train.run_backtest --seed=42
```
✓ Compare with baseline strategies
✓ Output: `outputs/random_seed42/`

---

## 5️⃣  PPO TRAINING - MVP (30-60 seconds)
```bash
python -m quant_rl.train.train_rl --mvp --seed=42
```
✓ Quick iteration on recent 30 days
✓ Model saved: `outputs/ppo_model_seed42`
✓ Output: metrics + test evaluation

---

## 6️⃣  PPO TRAINING - FULL (2-5 minutes)
```bash
python -m quant_rl.train.train_rl --seed=42
```
✓ Full training (500k timesteps)
✓ Best for thesis validation

---

## 🔍 INSPECT OUTPUTS

### View Generated Charts
```bash
# List charts
ls -lah outputs/baseline_macd_seed42/testing/orders/trade_*.png

# Open PNG chart
open outputs/baseline_macd_seed42/testing/orders/trade_0001*.png

# Open interactive HTML
open outputs/baseline_macd_seed42/testing/orders/trade_0001*.html
```

### View Metrics
```bash
# Find all output directories
ls -d outputs/*/

# View metrics file
cat outputs/baseline_macd_seed42/testing/metrics.txt
```

---

## 🚀 RUN ALL AT ONCE (Complete Verification)

```bash
cd /home/l2nguyen/Aalto_MS_Thesis
source $HOME/.local/bin/env

echo "=== 1. Unit Tests ===" && \
pytest tests/test_structure.py tests/test_risk.py -q && \

echo "=== 2. MACD Baseline ===" && \
python -m quant_rl.train.run_baselines --strategy macd --seed=42 --no-save && \

echo "=== 3. Random Policy ===" && \
python -m quant_rl.train.run_backtest --seed=42 --no-save && \

echo "=== ✓ COMPLETE ===" && \
ls -d outputs/*/ | tail -3
```

**Total time**: ~30-40 seconds

---

## 📋 KEY CONFIGS IN `default.yaml`

```yaml
backtest.validation.max_loss_per_trade_usd: 100.0    # Safety cap
risk.default_risk_frac: 0.01                         # 1% equity risk
risk.swing_buffer_pts: 1.0                           # Buffer beyond swings
risk.rr_ratio_default: 2.0                           # Risk:Reward ratio
```

---

## 📊 WHAT TO EXPECT

### Tests ✓
- All 12 pass
- Swing detection causal (no lookahead)
- Risk sizing correct

### Baselines ✓
- 8000-10000 trades
- Hold times: 5-60+ minutes (not 1-2m)
- Variable lot sizes (not always 1.00)
- SL/TP lines on charts

### Charts ✓
- Green entry arrow, red exit arrow
- Red dashed SL line (swing-based)
- Green dashed TP line (entry ± rr_ratio × risk)
- Trade info: Direction, Open, Close, Volume, Duration

---

## 🔗 GITHUB

Commit: `git log --oneline -1`
Push: `git push origin main`
Status: `git status`

---

## 💡 TIPS

1. **Skip chart export**: Add `--no-save` to run faster
2. **Quick test**: Run MVP PPO: `python -m quant_rl.train.train_rl --mvp`
3. **Compare strategies**: Run MACD, EMA, and random → compare metrics
4. **Debug**: Check `outputs/*/testing/metrics.txt` for results

---

Ready to verify the full implementation! 🎯
