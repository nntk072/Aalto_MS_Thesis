# Aalto MS Thesis — Quantitative RL Trading System

**Reinforcement learning trading agent** with multi-timeframe PO3/IFVG signal detection,
built on cleaned US100 (Nasdaq-100) M1 data and trained with Stable-Baselines3 PPO/SAC.

This repository implements the full pipeline: data ingestion → multi-timeframe feature
engineering → PO3 (Price Order Block 3) / FVG / IFVG signal detection → backtesting →
RL training → out-of-sample evaluation → chart visualization → optional live MT5 bridge.

---

## Table of Contents

- [Overview](#overview)
- [Documentation](#documentation)
- [Key Components](#key-components)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [PO3 / FVG / IFVG Detection](#po3--fvg--ifvg-detection)
- [RL Agent](#rl-agent)
- [Backtesting](#backtesting)
- [Engine Validation](#engine-validation)
- [Visualization](#visualization)
- [Testing](#testing)
- [CI / Code Quality](#ci--code-quality)
- [Known Limitations / Future Work](#known-limitations--future-work)

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

---

## Documentation

| Page | Purpose |
|------|---------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/architecture.md](docs/architecture.md) | Pipeline, contracts, invariants, adding a strategy |
| [docs/config.md](docs/config.md) | YAML config catalog and cache versioning |
| [docs/operations/RUNNING_COMMANDS.md](docs/operations/RUNNING_COMMANDS.md) | Command reference for research sessions |
| [docs/operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md) | Paper-to-live promotion protocol |
| [AGENTS.md](AGENTS.md) | Agent guide for coding assistants |
| [orchestra/README.md](orchestra/README.md) | Multi-model CLI pipeline |

---

## Key Components

| Module | Purpose |
|--------|---------|
| `quant_rl/data/` | Load and resample M1 → M5/M15/M30/H1/H4/D1 OHLCV bars |
| `quant_rl/features/` | Feature engineering: indicators, structure, PO3/FVG/IFVG, liquidity |
| `quant_rl/envs/` | Gymnasium trading environment with structure-aware SL/TP and strategy overlays |
| `quant_rl/models/` | SB3 PPO/SAC agent with TCN/Transformer/GRU encoders |
| `quant_rl/backtest/` | Event-driven backtest engine with realistic costs |
| `quant_rl/evaluation/` | Metrics, walk-forward splits, bootstrap CIs, episode runner |
| `quant_rl/eval/` | Rollout, plots, export, `eval_run` checkpoint re-evaluation |
| `quant_rl/train/` | Training scripts, baselines, callbacks |
| `quant_rl/live/` | `RLStrategyAdapter` — MT5 live bridge |
| `mt5_trading/` | Broker I/O: robots, risk manager, rule-based strategies |
| `orchestra/` | Multi-model task pipeline (triage → plan → implement → verify) |

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

## Quick Start

### One-time setup

```bash
cd Aalto_MS_Thesis
uv sync
source .venv/bin/activate
```

### Prepare data

```bash
python scripts/prepare_data.py
```

### Run tests

```bash
uv run pytest tests/ -v
```

### Train RL agent — MVP (~1 minute)

```bash
uv run python -m quant_rl.train.train_rl --mvp --seed=42

# Strategy overlay (Idea 1: PO3 + IFVG)
uv run python -m quant_rl.train.train_rl --mvp --strategy po3_ifvg --seed=42

# SAC + GRU encoder
uv run python -m quant_rl.train.train_rl --mvp --algo sac --arch gru --seed=42
```

### Evaluate a trained checkpoint

```bash
uv run python -m quant_rl.eval.eval_run --run outputs/<run_dir>
```

### MACD baseline

```bash
uv run python -m quant_rl.train.run_baselines --strategy macd --no-save
```

### Demo trading (rule-based crossover, not the RL pipeline)

```bash
# Paper mode (default) — logs signals only, no orders
DEMO_SYMBOL=EURUSD PAPER_TRADING=true uv run python demo_trading.py --once
```

### Live / paper trading (RL)

```bash
# Paper trade a trained checkpoint (PAPER_TRADING defaults to true)
PAPER_TRADING=true RL_MODEL_PATH=outputs/<run>/model/ppo_final \
    python live_trading_rl.py --once
```

See [docs/operations/RUNNING_COMMANDS.md](docs/operations/RUNNING_COMMANDS.md) for the full command catalog and
[docs/operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md) for the paper-to-live promotion protocol.

---

## Project Structure

```
Aalto_MS_Thesis/
├── quant_rl/                    # Core library
│   ├── data/                    # Data loading, resampling, split, ticks
│   ├── features/                # Feature engineering
│   │   ├── build.py             # Feature pipeline entry point
│   │   ├── indicators.py        # Technical indicators (RSI, MACD, etc.)
│   │   ├── structure.py         # Session/liquidity level detection
│   │   ├── swings.py            # Fractal pivots and ATR-filtered swings
│   │   ├── liquidity.py         # Liquidity sweeps and BOS
│   │   ├── po3_config.py        # PO3/FVG/IFVG detection + zone builder
│   │   ├── po3_state.py         # PO3 manipulation/distribution state
│   │   ├── smt.py               # SMT divergence detection
│   │   ├── session_ohlc.py      # CT-anchored session OHLC levels
│   │   └── normalize.py         # Feature normalization
│   ├── envs/                    # Gymnasium environments
│   │   ├── trading_env.py       # Main RL trading environment
│   │   ├── reward.py            # Differential Sharpe Ratio reward
│   │   ├── sweep_reward.py      # Sweep confirmation reward
│   │   ├── po3_reward.py        # PO3 alignment reward
│   │   ├── distribution_reward.py
│   │   └── strategies/          # Baseline, PO3IFVG, Distribution
│   ├── models/                  # RL model architectures
│   ├── backtest/                # Event-driven backtest engine
│   ├── evaluation/              # Metrics, walk-forward, bootstrap CIs
│   ├── eval/                    # Rollout, plots, export, eval_run
│   ├── train/                   # train_rl, run_backtest, run_baselines
│   ├── live/                    # RLStrategyAdapter (MT5 bridge)
│   ├── config/                  # OmegaConf default.yaml
│   └── utils/                   # Device helpers
├── mt5_trading/                 # MT5 broker I/O and rule-based robots
├── orchestra/                   # Multi-model CLI pipeline
├── config/                      # Strategy and feature variant YAMLs
├── scripts/                     # prepare_data, compare_encoders, report_g3, …
├── tests/                       # Test suite (pytest)
├── data/                        # Raw M1 CSV data (gitignored)
└── docs/                        # Documentation (architecture, config, operations)
    └── operations/              # Running commands and deployment
```

---

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
from quant_rl.data.session import get_session

session = get_session(timestamp)  # "asia" | "london" | "ny" | "closed"
```

Session boundaries (broker tz `Etc/GMT-3`, UTC+3): Asia 01:05–09:00, London 09:00–16:30,
NY 16:30–23:00 (inclusive).

---

## RL Agent

Built on **Stable-Baselines3** with custom sequence encoders.

```python
from quant_rl.models.agent import build_agent

model = build_agent(env, cfg)  # PPO + TCN (default)
model = build_agent(env, cfg, arch="transformer", algo="sac")

model.learn(total_timesteps=1_000_000)
```

| Architecture | Description |
|--------------|-------------|
| **TCN** | Temporal Convolutional Network (default) |
| **Transformer** | Multi-head self-attention encoder |
| **GRU** | Gated Recurrent Unit encoder |
| **VAE** | Variational autoencoder latent features (standalone training only) |

Observation: `Dict[seq: (60, F), account: (5,)]`
Action: Discrete `{0=hold, 1-9=long, 10-18=short, 19=exit}` or Continuous `Box(-1,1)`.
Strategy mode (`--strategy po3_ifvg|distribution`): 4-D `Box [direction, risk, rr, tp]`.

---

## Backtesting

Event-driven backtest engine with realistic execution:

```bash
uv run python -m quant_rl.train.run_backtest
```

Features structure-aware SL/TP, Differential Sharpe Ratio reward, configurable costs, and drawdown guardrails.

---

## Engine Validation

Cross-validate the custom event-driven backtest engine against `backtrader`:

```bash
uv run python -m quant_rl.backtest.cross_validation.run
```

See `tests/test_backtest/test_engine_cross_validation.py` for unit tests.

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

---

## Testing

```bash
uv run pytest tests/ -v                         # Full suite (CI merge bar)
uv run pytest tests/test_features/ -v           # PO3 + structure
uv run pytest tests/test_envs/ -v               # Environment + strategies
uv run pytest tests/test_integration/ -v        # End-to-end smoke tests
pytest -m "not slow"                            # Local shortcut only
```

---

## CI / Code Quality

Standard merge bar (mirrors `.github/workflows/ci.yml` and `orchestra/ci_gate.py`):

| Check | Command |
|-------|---------|
| Formatting | `uv run ruff format --check .` |
| Linting | `uv run ruff check .` |
| Type checking | `uv run mypy .` |
| Tests | `uv run pytest tests/ -v` |

---

## Known Limitations / Future Work

- **VAE feature extractor:** optional via `--use-vae --vae-path <checkpoint.pth>`
  (train the VAE first with `scripts/train_vae.py`). Default training path leaves VAE off.
- **Rule-based live baseline (`live_trading.py`):** uses simplified guardrail criteria compared to the full RL promotion protocol.
- **Multi-timeframe alignment:** some higher-timeframe feature alignment edge cases may benefit from additional validation.

## License

This project is part of a Master's thesis at Aalto University.
