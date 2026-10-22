# Aalto MS Thesis — Quantitative RL Trading System

**Reinforcement-learning trading system** with multi-timeframe market features and
PO3/IFVG strategy components. Configured runs can use US100 price bars, optional US500
context, native or derived timeframes, and bid/ask tick execution data.

This repository implements the full pipeline: data ingestion → multi-timeframe feature
engineering → PO3 (Power of Three) / FVG / IFVG signal detection → backtesting →
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
PO3 (Power of Three) signals** — a price-action methodology combining:

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
| [docs/reproducibility.md](docs/reproducibility.md) | Training/evaluation pipeline and evidence requirements |
| [docs/research_reference.md](docs/research_reference.md) | Implementation-verified action, data, reward and risk contracts |
| [docs/architecture.md](docs/architecture.md) | Pipeline, contracts, invariants, adding a strategy |
| [docs/config.md](docs/config.md) | YAML config catalog and cache versioning |
| [docs/operations/RUNNING_COMMANDS.md](docs/operations/RUNNING_COMMANDS.md) | Command reference for research sessions |
| [docs/operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md) | Paper-to-live promotion protocol |
| [AGENTS.md](AGENTS.md) | Agent guide for coding assistants |
| [orchestra/README.md](orchestra/README.md) | Multi-model CLI pipeline |

---

## Key Components

| Package | Purpose |
|--------|---------|
| `configuration/` | OmegaConf loading, merge, validation, hashing, and repository paths |
| `market_data/` | Dataset discovery, bar/tick loading, cleaning, resampling, sessions, splits, and cache identity |
| `features/` | Causal technical/context/PO3 feature construction and feature-cache semantics |
| `trading/` | Broker-neutral account, policy, execution, order, position, risk, and strategy contracts |
| `simulation/` | Gymnasium environment runtime plus event-driven backtesting |
| `models/` | Reusable encoders, policy networks, auxiliary models, and model factories |
| `learning/` | PPO/SAC backends, learner contracts, callbacks, and checkpoint loading |
| `decisioning/` | Versioned observation/action contracts and policy adapters |
| `training/` | Training configuration, setup services, callbacks, runners, and training artifacts |
| `evaluation/` | Rollout, metrics, reanalysis, reporting, provenance, and plotting |
| `experiments/` | Variant and campaign orchestration, checkpoint lineage, and campaign reporting |
| `live/` | Broker-neutral live/paper runtime |
| `integrations/mt5/` | MetaTrader 5 adapters, broker domain objects, robots, and risk integration |
| `compute/` | Slurm/Triton resource, continuation, resume, and worker-dispatch mechanics |
| `artifacts/` | Cross-cutting artifact identity, manifests, layout, registry, atomic writes, retention, and promotion |
| `orchestra/` | Multi-model task pipeline (triage → plan → implement → verify) |

---

## Requirements

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- MetaTrader 5 data (exported M1 CSV files in `data/`)

---

## Installation

The project uses **arch-matching virtualenvs** on Triton (shared NFS):

| Architecture | Host | Venv | Setup |
|--------------|------|------|-------|
| **aarch64** | GH200 GPU nodes | `.venv` | `uv sync` or `tools/setup-venv` |
| **x86_64** | H200, local, CI | `.venv-x86` | `tools/setup-venv-x86` |

Do not cross-activate (sourcing the wrong-arch venv gives "Exec format error").

```bash
git clone https://github.com/nntk072/Aalto_MS_Thesis.git
cd Aalto_MS_Thesis

# x86_64 (local, H200, CI)
tools/setup-venv-x86
source scripts/compute/slurm/activate_venv.sh

# aarch64 (GH200) — run on an aarch64 node
uv sync
source .venv/bin/activate
```

---

## Quick Start

### One-time setup

```bash
cd Aalto_MS_Thesis
tools/setup-venv          # auto-detects arch (.venv or .venv-x86)
source scripts/compute/slurm/activate_venv.sh
```

### Prepare data

```bash
uv run thesis-data prepare
```

### Run tests

```bash
uv run pytest tests/test_features/test_po3_state.py -q
uv run pytest tests/test_envs/test_entry_state_machine.py -q
```

Run the relevant test files for the change. GitHub CI and Orchestra coordinate
the broader verification gates.

### Train RL agent — MVP (~1 minute)

```bash
uv run thesis-train train --mvp --seed=42

# Strategy overlay (Idea 1: PO3 + IFVG)
uv run thesis-train train --mvp --strategy po3_ifvg --seed=42

# SAC + GRU encoder
uv run thesis-train train --mvp --algo sac --arch gru --seed=42
```

### Evaluate a trained checkpoint

```bash
uv run thesis-evaluate reanalyze --run outputs/<run_dir> --output outputs/reanalysis/<name>
```

### MACD baseline

```bash
uv run thesis-train baselines --strategy macd --no-save
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

```text
Aalto_MS_Thesis/
├── configuration/              # Config loading, merging, schemas, paths
├── artifacts/                  # Shared artifact lifecycle primitives
├── market_data/                # Bars/ticks, transforms, splits, cache identity
├── features/                   # Context, technical, PO3, causal pipeline
├── trading/                    # Account, execution, orders, policy, risk, strategies
├── simulation/                 # TradingEnv runtime and backtest engine
├── models/                     # Reusable encoders, policy networks, auxiliary models
├── learning/                   # PPO/SAC learning backends and checkpoint services
├── decisioning/                # Observation/action contracts and policy adapters
├── training/                   # Training setup, callbacks, runners, artifacts
├── evaluation/                 # Rollout, metrics, reanalysis, reporting, plotting
├── experiments/                # Variants and campaign orchestration
├── live/                       # Broker-neutral paper/live runtime
├── integrations/mt5/           # MT5-specific adapters and robots
├── compute/                    # Slurm/Triton compute mechanics
├── orchestra/                  # Multi-model engineering pipeline
├── config/                     # Declarative YAML assets only
├── scripts/
│   ├── compute/slurm/          # Thin platform wrappers
│   ├── research/               # Audits, benchmarks, thesis evidence, frozen launches
│   └── maintenance/            # Explicit rebuild/migration wrappers
├── tools/                      # Canonical engineering/check commands
├── tests/                      # Unit, integration, parity, architecture tests
├── data/                       # Raw/derived dataset storage
├── cache/                      # Disposable reproducible caches
├── outputs/                    # Run/campaign/evaluation artifacts
├── logs/                       # Non-artifact operational logs
├── baseline/                   # Explicitly promoted baseline evidence
└── docs/                       # Architecture, reproducibility, operations, thesis evidence
```

---

## PO3 / FVG / IFVG Detection

Located in `features/po3/config.py` — the single source of truth for all PO3 signal rules.

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
from market_data.transforms.sessions import get_session

session = get_session(timestamp)  # "asia" | "london" | "ny" | "closed"
```

Session boundaries (broker tz `Etc/GMT-3`, UTC+3): Asia 01:05–09:00, London 09:00–16:30,
NY 16:30–23:00 (inclusive).

These are fixed clock labels. The configured [date-level NY close schedule](config/ny_session_schedule.csv)
separately marks known shortened sessions from that date's first bar; shortened-session bars
remain in the chronological dataset and feature history.

---

## RL Agent

Built on **Stable-Baselines3** with custom sequence encoders.

```python
from models import build_agent

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
Strategy mode uses a versioned, flag-derived continuous action schema; with all current direction, multi-TP, SL/TP-mode, simplex-fraction, and exit controls enabled it is 12-D.

---

## Backtesting

Event-driven backtest engine with realistic execution:

```bash
uv run thesis-train backtest
```

Features structure-aware SL/TP, Differential Sharpe Ratio reward, configurable costs, and drawdown guardrails.

---

## Engine Validation

Cross-validate the custom event-driven backtest engine against `backtrader`:

```bash
uv run python -m simulation.backtest.cross_validation.run
```

See `tests/test_backtest/test_engine_cross_validation.py` for unit tests.

---

## Visualization

### Static Charts (matplotlib)

```python
from evaluation.plotting.po3_plots import plot_fvg_signals

fig = plot_fvg_signals(bars, signals, out_path="chart.png")
```

### Interactive Charts (Plotly)

```python
from evaluation.plotting.plots_interactive import plot_fvg_signals_interactive

plot_fvg_signals_interactive(bars, signals, out_path="chart.html")
```

---

## Testing

```bash
uv run pytest tests/test_features/test_po3_state.py -q
uv run pytest tests/test_envs/test_entry_state_machine.py -q
uv run pytest tests/test_integration/test_strategy_integration.py -q
```

Keep local runs scoped to the affected behavior. GitHub CI and Orchestra run
the broader verification gates; agent commit checks follow the scoped-test rule in
[`AGENTS.md`](AGENTS.md).

---

## CI / Code Quality

Standard merge bar (mirrors `.github/workflows/ci.yml` and `orchestra/ci_gate.py`):

| Check | Command |
|-------|---------|
| Formatting | `uv run ruff format --check .` |
| Linting | `uv run ruff check .` |
| Type checking | `uv run mypy .` |
| Tests | Broader verification gates in CI; scoped relevant test paths locally |

---

## Known Limitations / Future Work

- **VAE feature extractor:** optional via `--use-vae --vae-path <checkpoint.pth>`
  (train the VAE first with `training/runner/vae.py`). Default training path leaves VAE off.
- **Rule-based live baseline (`live_trading.py`):** uses simplified guardrail criteria compared to the full RL promotion protocol.
- **Multi-timeframe alignment:** some higher-timeframe feature alignment edge cases may benefit from additional validation.

## License

This project is part of a Master's thesis at Aalto University.
