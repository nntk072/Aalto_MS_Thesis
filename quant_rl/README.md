# quant_rl — RL Trading Infrastructure

Core library for the Aalto MS Thesis quantitative RL trading system. Covers
data ingestion, feature engineering, Gymnasium environment, backtest engine,
model wiring, training, evaluation, and the live MT5 bridge.

## Run order

```bash
# 1. Install deps
uv sync

# 2. Prepare data (raw CSV → parquet → feature cache)
python scripts/prepare_data.py

# 3. Verify baselines run on real data
uv run python -m quant_rl.train.run_baselines

# 4. Verify backtester (random policy)
uv run python -m quant_rl.train.run_backtest

# 5. Verify env wiring (stub config, no long training)
uv run python -m quant_rl.train.train_rl --mvp

# 6. Full run
uv run python -m quant_rl.train.train_rl

# 7. Run tests
uv run pytest tests/ -v
```

## Package layout

```
quant_rl/
  config/         OmegaConf default.yaml + loader
  data/           loader, resample (M1→all TFs), clean, session, align, pipeline, ticks
  features/       indicators, SMT, structure, swings, liquidity, PO3, normalize, build
  backtest/       account, costs, guardrails, broker, event-driven engine, cross_validation
  envs/           Gymnasium TradingEnv, DSR/sweep/alignment rewards, strategies/
  models/         TCN/GRU/Transformer encoders + build_agent (PPO/SAC), VAE, auxiliary
  baselines/      buy-and-hold, EMA/MACD/RSI rule-based, LSTM classifier
  evaluation/     metrics, walk-forward, bootstrap CIs, calibration, episode runner
  eval/           rollout, plots, export, eval_run checkpoint re-evaluation
  train/          train_rl, run_backtest, run_baselines, callbacks, auxiliary_training
  live/           RLStrategyAdapter (MT5 bridge)
  utils/          device helpers
scripts/
  prepare_data.py           raw → parquet → features
  compare_encoders.py       encoder architecture comparison
  verify_strategy_features.py  strategy column contract check
  report_g3.py              run-report gate
tests/            causal features, guardrails, envs, integration, live bridge
```

## Data notes

All higher-TF CSV files are mislabeled duplicates of finer TFs.
**Every timeframe is resampled from the true M1 source**
(`data/US100.cash_M1_*` and `data/US500.cash_M1_*`). Other CSV files are
ignored by the pipeline.

## Model contract

### Observation space

```
obs["seq"]     : float32  [batch, T, F]   (T = cfg.env.obs_window, F = n_features)
obs["account"] : float32  [batch, 5]      (normalised account state vector)
```

### What `build_agent` wires

1. **`quant_rl/models/encoder.py`** — `TCNEncoder` / `GRUEncoder` /
   `TransformerEncoder` (all subclass `BaseFeaturesExtractor`), chosen by the
   `arch` argument (`tcn` / `gru` / `transformer`).
2. **`quant_rl/models/agent.py`** — `build_agent(env, cfg, arch, algo,
   use_vae)` returns the SB3 `PPO` / `SAC` model wired to that encoder, used
   by `quant_rl/train/train_rl.py` and `scripts/compare_encoders.py`.

## Config overrides (key=value)

```bash
python scripts/prepare_data.py data.cache_dir=my_cache env.obs_window=30
uv run python -m quant_rl.train.train_rl --mvp --strategy po3_ifvg
```

See [docs/config.md](../docs/config.md) for the full YAML catalog.
