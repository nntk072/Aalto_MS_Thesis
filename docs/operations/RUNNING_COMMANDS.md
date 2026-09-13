# Running Commands

Command catalog for research, evaluation, and live sessions. For architecture
and config details see [architecture.md](../architecture.md) and
[config.md](../config.md).

---

## Setup

```bash
cd Aalto_MS_Thesis
uv sync
source .venv/bin/activate
```

---

## CI gate (merge bar)

Mirrors `.github/workflows/ci.yml` and `orchestra/ci_gate.py`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest tests/ -v
```

Local shortcut (not the merge bar): `pytest -m "not slow"`

---

## Data pipeline

```bash
# Raw CSV → parquet bar cache → feature cache
python scripts/prepare_data.py

# Force rebuild (ignore cache)
python scripts/prepare_data.py --force
# or: uv run python -m quant_rl.train.train_rl --force
```

---

## Unit and integration tests

```bash
# Full suite
uv run pytest tests/ -v

# By domain
uv run pytest tests/test_features/ -v
uv run pytest tests/test_envs/ -v
uv run pytest tests/test_integration/ -v
uv run pytest tests/test_orchestra/ -v
uv run pytest tests/test_live/ -v

# Targeted
uv run pytest tests/test_risk.py tests/test_macd_baseline.py -v
```

---

## Baseline strategies

```bash
# MACD crossover (quick, no charts)
uv run python -m quant_rl.train.run_baselines --strategy macd --no-save

# Full run with per-trade charts
uv run python -m quant_rl.train.run_baselines --strategy macd

# Other baselines
uv run python -m quant_rl.train.run_baselines --strategy ema
uv run python -m quant_rl.train.run_baselines --strategy rsi
```

MACD rules: EMA12/26 MACD + SMA9 signal + EMA50 trend filter. Long when
`close > EMA50` and bullish cross. Cooldown ≥5 bars after exit.

---

## Random-policy backtest

```bash
uv run python -m quant_rl.train.run_backtest --no-save
uv run python -m quant_rl.train.run_backtest --seed=42
```

---

## RL training

```bash
# MVP smoke test (~8k timesteps, 30 days)
uv run python -m quant_rl.train.train_rl --mvp --seed=42

# Full training (500k timesteps)
uv run python -m quant_rl.train.train_rl --seed=42

# Algorithm / encoder / reward
uv run python -m quant_rl.train.train_rl --mvp --algo sac --arch gru --reward sweep --seed=42

# Strategy overlay (Idea 1: PO3 + IFVG, Idea 2: distribution)
uv run python -m quant_rl.train.train_rl --mvp --strategy po3_ifvg --seed=42
uv run python -m quant_rl.train.train_rl --mvp --strategy distribution --seed=42

# Feature variant config (MTF expansion)
uv run python -m quant_rl.train.train_rl --mvp \
    --config config/features_full_po3_mtf.yaml --seed=42

# Purged walk-forward validation
uv run python -m quant_rl.train.train_rl --walk-forward --wf-splits 5 \
    --purge-bars 60 --embargo-bars 20 --seed=42

# Weights & Biases logging
uv run python -m quant_rl.train.train_rl --mvp --wandb
```

### Config overrides (OmegaConf key=value)

```bash
uv run python -m quant_rl.train.train_rl --mvp \
    account.initial_balance=100000.0 \
    env.obs_window=30 \
    training.total_timesteps_mvp=4096
```

---

## Evaluation

```bash
# Re-evaluate an existing trained run (no retraining)
uv run python -m quant_rl.eval.eval_run --run outputs/<run_dir>

# With a specific checkpoint
uv run python -m quant_rl.eval.eval_run --run outputs/<run_dir> \
    --checkpoint model/ppo_ckpt_328350_steps.zip

# View latest metrics
LATEST=$(ls -d outputs/*/ | tail -1)
cat "$LATEST/test/metrics.json"
```

---

## Engine cross-validation

```bash
uv run python -m quant_rl.backtest.cross_validation.run
```

---

## Encoder comparison

```bash
uv run python scripts/compare_encoders.py
```

---

## Run-report gate (G3)

```bash
uv run python scripts/report_g3.py --runs-dir outputs --sharpe-threshold 1.0
```

---

## Live / paper trading

### RL agent (`live_trading_rl.py`)

```bash
# One dry-run cycle (PAPER_TRADING defaults to true)
PAPER_TRADING=true RL_MODEL_PATH=outputs/<run>/model/ppo_final \
    python live_trading_rl.py --once

# Continuous paper loop
PAPER_TRADING=true RL_MODEL_PATH=outputs/<run>/model/ppo_final python live_trading_rl.py

# Real orders — only after DEPLOYMENT.md criteria pass
PAPER_TRADING=false RL_MODEL_PATH=outputs/<run>/model/ppo_final python live_trading_rl.py
```

### Rule-based baseline (`live_trading.py`)

```bash
PAPER_TRADING=true STRATEGY_TYPE=combined python live_trading.py --once
# STRATEGY_TYPE: crossover | smc | trend_breakout | combined
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for the paper-to-live promotion protocol.

---

## Orchestra (multi-model pipeline)

```bash
orchestra doctor              # Tier-0 health check (no inference)
orchestra run "task description"   # full pipeline (verify runs CI gate)
```

See [orchestra/README.md](../orchestra/README.md).

---

## Troubleshooting

```bash
# Rebuild feature cache
uv run python -m quant_rl.train.run_backtest --force
rm -rf cache/*.parquet

# Reduce MVP memory / time
uv run python -m quant_rl.train.train_rl --mvp env.obs_window=30 \
    training.total_timesteps_mvp=4096
```

---

## Quick reference

| Command | Purpose | Typical time |
|---------|---------|--------------|
| `pytest tests/ -v` | Full test suite | ~2 min |
| `ruff check .` / `mypy .` | Lint + typecheck | <1 min |
| `run_baselines --strategy macd` | MACD baseline | ~10 s |
| `train_rl --mvp` | RL smoke test | ~1 min |
| `train_rl` (no `--mvp`) | Full RL training | 10+ min |
| `eval_run --run outputs/...` | Re-evaluate checkpoint | ~1 min |
| `orchestra run "task"` | Multi-model pipeline | varies |
