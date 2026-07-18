# quant_rl – Non-Model RL Infrastructure (Aalto MS Thesis)

> **What this is:** Everything except the neural model. Drop in your encoder + PPO agent and the full pipeline runs end-to-end.

## Run order

```bash
# 1. Install deps
source $HOME/.local/bin/env   # if uv was just installed
uv sync

# 2. Prepare data (raw CSV → parquet → feature cache)
python scripts/prepare_data.py

# 3. Verify baselines run on real data
python -m quant_rl.train.run_baselines

# 4. Verify backtester (random policy)
python -m quant_rl.train.run_backtest

# 5. Verify env wiring (stub MLP policy, no encoder needed)
python -m quant_rl.train.train_rl --stub

# 6. (After you implement quant_rl/models/encoder.py)
python -m quant_rl.train.train_rl

# 7. Run tests
pytest
```

## Package layout

```
quant_rl/
  config/         OmegaConf default.yaml + loader
  data/           loader, resample (M1→all TFs), clean, session, align, pipeline
  features/       indicators, SMT divergence, rolling z-score, build
  backtest/       account, costs, guardrails, broker, event-driven engine
  envs/           Gymnasium TradingEnv + DSR reward
  models/         ← YOU IMPLEMENT (typed stubs provided)
    base.py       Encoder / Policy abstract interfaces
    encoder.py    TCN/Transformer BaseFeaturesExtractor stub (NotImplementedError)
    agent.py      PPO wiring stub
  baselines/      buy-and-hold, EMA/MACD/RSI rule-based
  eval/           metrics (incl. breach_rate), purged walk-forward, multi-seed report
  train/          run_backtest.py, run_baselines.py, train_rl.py
scripts/
  prepare_data.py one-shot raw→parquet→features
tests/            causal features, guardrails, cost model, DSR reward, session filter
```

## Data notes

All higher-TF CSV files are mislabeled duplicates of finer TFs (see plan).
**Every timeframe is resampled from the true M1 source** (`data/US100.cash_M1_*` and `data/US500.cash_M1_*`). Other CSV files are ignored by the pipeline.

## Model stub contract

### Observation space
```
obs["seq"]     : float32  [batch, T, F]   (T = cfg.env.obs_window, F = n_features)
obs["account"] : float32  [batch, 5]      (normalised account state vector)
```

### What you implement
1. **`quant_rl/models/encoder.py`** – subclass `SequenceEncoder(BaseFeaturesExtractor)`, implement `forward(obs) → [batch, D + 5]`.
2. **`quant_rl/models/agent.py`** – fill in `build_agent(env, cfg)` to wire your encoder to `stable_baselines3.PPO`.

### Example encoder wiring
```python
policy_kwargs = dict(
    features_extractor_class=SequenceEncoder,
    features_extractor_kwargs=dict(
        seq_len=cfg.env.obs_window,
        n_features=<F>,
        latent_dim=128,
    ),
)
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs,
            n_steps=cfg.ppo.n_steps, ...)
```

## Config overrides (key=value)
```bash
python scripts/prepare_data.py data.cache_dir=my_cache env.obs_window=30
```
