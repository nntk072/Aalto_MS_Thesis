# Aalto MS Thesis — Quantitative RL Trading System

**Reinforcement learning trading agent** with multi-timeframe PO3/IFVG signal detection,
built on top of cleaned US100 (Nasdaq-100) M1 data and trained with Stable-Baselines3 PPO/SAC.

This repository implements the full pipeline: data ingestion → multi-timeframe feature
engineering → PO3 (Price Order Block 3) / FVG / IFVG signal detection → backtesting →
RL training → out-of-sample evaluation → chart visualization.

---

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [PO3 / FVG / IFVG Detection](#po3--fvg--ifvg-detection)
- [RL Agent](#rl-agent)
- [Backtesting](#backtesting)
- [Visualization](#visualization)
- [Testing](#testing)
- [CI / Code Quality](#ci--code-quality)

---

## Overview

This project builds a reinforcement learning agent that trades US100 using **multi-timeframe
PO3 (Price Order Block 3) signals** — a price-action methodology combining:

| Layer | Timeframe | What it detects |
|-------|-----------|-----------------|
| **HTF** | M15 | Fair Value Gaps (FVG) — imbalance zones that act as support/resistance |
| **LTF** | M5 | IFVG (Inversion FVG) confirmations — LTF confluence for HTF zones |
| **Entry** | M1 | Retest, close-through, and LTF-FVG entry triggers |

The agent observes a 60-bar rolling window of technical + structure + PO3 features and
learns position sizing, entry timing, and stop-loss/take-profit placement.
   
## Key Components

| Module | Purpose |
|--------|---------|
| `quant_rl/data/` | Load and resample M1 → M5/M15/M30/H1/H4/D1 OHLCV bars |
| `quant_rl/features/` | Feature engineering: indicators, structure levels, PO3/FVG/IFVG detection |
| `quant_rl/envs/` | Gymnasium trading environment with structure-aware SL/TP |
| `quant_rl/models/` | SB3 PPO/SAC agent with TCN/Transformer/GRU encoders |
| `quant_rl/backtest/` | Event-driven backtest engine with realistic costs |
| `quant_rl/eval/` | OOS evaluation, trade metrics, chart visualization |
| `quant_rl/train/` | Training scripts, baselines, callbacks |

---

## Requirements

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- MetaTrader 5 data (exported M1 CSV files in `data/`)

---

## Installation

```bash
git clone https://github.com/nntk072/Aalto_MS_Thesis.git
cd Aalto_MS_Thesis

# Create virtual environment and install all dependencies
uv sync

# Activate (Linux/macOS)
source .venv/bin/activate
```

---

## Quick Start & Commands

### One-Time Setup
```bash
cd /home/nguyenl37/Aalto_MS_Thesis
source .venv/bin/activate        # or: uv sync && source .venv/bin/activate
```

### Run Tests (~1 minute)
```bash
# All tests
.venv/bin/python -m pytest tests/ -v

# Specific modules
.venv/bin/python -m pytest tests/test_features/ -v      # PO3 + structure
.venv/bin/python -m pytest tests/test_eval/ -v          # Evaluation + plots
.venv/bin/python -m pytest tests/test_integration/ -v   # Smoke tests
```

**Expected:** All tests passing ✅

### Lint + Type Check (~1 minute)
```bash
# Ruff linting
uv run ruff check quant_rl

# Type checking
uv run mypy quant_rl
```

**Expected:** All checks pass ✅

### MACD Baseline Strategy (~10 seconds)
```bash
# Quick run (no charts saved)
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd --no-save

# Full run (with chart generation)
.venv/bin/python -m quant_rl.train.run_baselines --strategy macd

# View results
ls outputs/baseline_macd_seed*/test/orders/trade_*.{png,html} | head -5
```

**Rules:** MACD (EMA12/26) + Signal (SMA9) + EMA50 trend filter. Long when `close > EMA50` + bullish cross. Cooldown ≥5 bars after exit.

### Train RL Agent — MVP (~1 minute)
```bash
# Quick training (8k timesteps, 30 days)
.venv/bin/python -m quant_rl.train.train_rl --mvp --seed=42

# View model output
ls outputs/ppo_model_seed42
```

### Train RL Agent — Full (~10+ minutes)
```bash
# Full training (500k timesteps, all data)
.venv/bin/python -m quant_rl.train.train_rl --seed=42
```

### Run Backtest (~5 seconds)
```bash
# Quick backtest
.venv/bin/python -m quant_rl.train.run_backtest --no-save

# Full backtest with charts
.venv/bin/python -m quant_rl.train.run_backtest --seed=42
```

### Evaluate OOS Performance
```bash
# Run OOS evaluation + generate charts
.venv/bin/python -m quant_rl.eval.eval_run

# View latest results
LATEST=$(ls -d outputs/*/ | tail -1)
cat $LATEST/test/metrics.json
```

### View Per-Trade Charts
```bash
# List charts
ls outputs/$LATEST/test/orders/trade_*.png   # Static
ls outputs/$LATEST/test/orders/trade_*.html  # Interactive
```

### Command Reference

| Command | Purpose | Time | Output |
|---------|---------|------|--------|
| `pytest tests/` | Validate all modules | ~2 min | Pass/fail |
| `ruff check quant_rl` | Lint check | <1s | Issues |
| `mypy quant_rl` | Type check | ~30s | Issues |
| `run_baselines --strategy macd` | Baseline backtest | 10s | PNG/HTML charts |
| `train_rl --mvp` | Quick RL training | ~1 min | PPO model + charts |
| `train_rl` (no --mvp) | Full RL training | 10+ min | Production model |
| `run_backtest` | Random policy baseline | 5s | Baseline metrics |
| `eval_run` | OOS evaluation | ~1 min | Metrics + charts |

### Train/Test Split Configuration

The project uses a **locked out-of-sample (OOS) split**:
- **Training period:** ≤ 2025-12-31 (in-sample)
- **Test period:** ≥ 2026-01-01 (out-of-sample)

Defined in `quant_rl/config/default.yaml`, enforced by `quant_rl/data/split.py`.

### Purged Walk-Forward Validation

```python
from quant_rl.evaluation.walkforward import purged_walk_forward

for split in purged_walk_forward(n=len(data), n_splits=5, purge_bars=60, embargo_bars=20):
    train_idx = split.train_idx
    test_idx = split.test_idx
```

- `purge_bars=60`: Removes last 60 bars from training (prevents leakage)
- `embargo_bars=20`: Removes first 20 bars from test (market reset)
- `n_splits=5`: Number of folds

Useful for hyperparameter tuning without touching the locked 2026 OOS set.

## Project Structure

```
Aalto_MS_Thesis/
├── quant_rl/                    # Core library
│   ├── data/                    # Data loading & resampling (M1 → HTF)
│   ├── features/                # Feature engineering
│   │   ├── build.py             # Feature pipeline entry point
│   │   ├── indicators.py        # Technical indicators (RSI, MACD, etc.)
│   │   ├── structure.py         # Session/liquidity level detection
│   │   ├── po3_config.py        # PO3/FVG/IFVG detection + zone builder
│   │   ├── smt.py               # SMT divergence detection
│   │   └── normalize.py         # Feature normalization
│   ├── envs/                    # Gymnasium environments
│   │   ├── trading_env.py       # Main RL trading environment
│   │   ├── reward.py            # Differential Sharpe Ratio reward
│   │   └── sweep_reward.py      # Sweep confirmation reward
│   ├── models/                  # RL model architectures
│   │   ├── agent.py             # SB3 PPO/SAC agent builder
│   │   ├── encoder.py           # TCN/Transformer/GRU sequence encoders
│   │   ├── base.py              # Base policy class
│   │   ├── auxiliary.py         # Auxiliary task heads
│   │   └── vae.py               # Variational autoencoder feature extractor
│   ├── backtest/                # Event-driven backtest engine
│   │   ├── engine.py            # Core matching/execution
│   │   ├── account.py           # Account state tracking
│   │   ├── broker.py            # Broker simulation
│   │   ├── costs.py             # Spread + commission model
│   │   ├── risk.py              # Position sizing & SL/TP
│   │   └── guardrails.py        # Drawdown limits
│   ├── eval/                    # Evaluation & visualization
│   │   ├── eval_run.py          # OOS evaluation pipeline
│   │   ├── rollout.py           # Policy rollout
│   │   ├── trade_metrics.py     # Trade-level analytics
│   │   ├── plots.py             # Static matplotlib charts
│   │   ├── plots_interactive.py # Interactive Plotly charts
│   │   ├── po3_plots.py         # PO3/FVG/IFVG signal charts
│   │   ├── export.py            # Results export
│   │   └── training_plots.py    # Training curve plots
│   ├── train/                   # Training scripts
│   │   ├── train_rl.py          # RL training entry point
│   │   ├── run_backtest.py      # Backtest entry point
│   │   ├── run_baselines.py     # Baseline strategy comparison
│   │   └── callbacks.py         # Custom SB3 callbacks
│   ├── config/                  # OmegaConf configurations
│   ├── baselines/               # Baseline models (LSTM classifier)
│   ├── validation/              # Cross-validation utilities
│   └── evaluation/              # Evaluation orchestration
├── tests/                       # Test suite (pytest)
├── data/                        # Raw M1 CSV data (gitignored)
├── configs/                     # YAML config files
```

## PO3 / FVG / IFVG Detection

Located in `quant_rl/features/po3_config.py` — the single source of truth for all PO3 signal rules.

### Detection Functions

| Function | Description |
|----------|-------------|
| `detect_fvg(bars, cfg)` | Detect Fair Value Gaps on a single timeframe |
| `detect_ifvg_confirmation(bars, fvg_signals, cfg)` | Detect IFVG (inversion) confirmations |
| `detect_entry_trigger(bars, fvg, ifvg, cfg)` | Generate entry signals (retest / close-through) |
| `detect_htf_fvg(m1, htf, cfg)` | Detect FVG on HTF, map back to M1 |
| `detect_ltf_ifvg(m1, ltf, cfg)` | Detect IFVG on LTF, map back to M1 |
| `detect_po3_entries(m1, htf, ltf, cfg)` | Unified multi-timeframe entry detection |
| `build_fvg_zones(signals, bars, cfg)` | Convert signals to full zone bounds for visualization |

### Session Tagging

```python
from quant_rl.features.structure import get_session

session = get_session(timestamp)  # Returns: "asia" | "london" | "ny"
```

Session boundaries (UTC+3): Asia 01:05–09:00, London 09:00–16:30, NY 16:30–23:50.

---

## RL Agent

Built on **Stable-Baselines3** with custom sequence encoders.

```python
from quant_rl.models.agent import build_agent

# PPO with TCN encoder (default)
model = build_agent(env, cfg)

# SAC with Transformer encoder
model = build_agent(env, cfg, arch="transformer", algo="sac")

model.learn(total_timesteps=1_000_000)
```

| Architecture | Description |
|--------------|-------------|
| **TCN** | Temporal Convolutional Network (default) |
| **Transformer** | Multi-head self-attention encoder |
| **GRU** | Gated Recurrent Unit encoder |
| **VAE** | Variational autoencoder latent features |

Observation: `Dict[seq: (60, F), account: (5,)]`
Action: Discrete {0=hold, 1-9=long, 10-18=short, 19=exit} or Continuous Box(-1,1)

---

## Backtesting

Event-driven backtest engine with realistic execution:

```bash
python -m quant_rl.train.run_backtest
```

Features structure-aware SL/TP, Differential Sharpe Ratio reward, configurable costs, and drawdown guardrails.

---

## Visualization

### Static Charts (matplotlib)
```python
from quant_rl.eval.po3_plots import plot_fvg_signals

fig = plot_fvg_signals(bars, signals, out_path="chart.png")
```

### Interactive Charts (Plotly)
```python
from quant_rl.eval.plots_interactive import plot_fvg_signals_interactive

plot_fvg_signals_interactive(bars, signals, out_path="chart.html")
```

Renders candlesticks with HTF FVG zones, LTF IFVG confirmations, and entry markers.

---

## Testing

```bash
pytest tests/                         # Full suite
pytest tests/test_features/           # PO3 + structure tests
pytest tests/test_eval/               # Evaluation + plot tests
pytest tests/test_integration/        # End-to-end smoke tests
```

---

## CI / Code Quality

| Check | Tool |
|-------|------|
| Linting | `ruff check .` |
| Formatting | `ruff format --check .` |
| Type checking | `mypy quant_rl` |
| Tests | `pytest tests/` |

---

## License

This project is part of a Master's thesis at Aalto University.
