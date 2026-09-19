# Configuration Reference

All runtime configuration flows through OmegaConf YAML files. The base is
`quant_rl/config/default.yaml`; variant files under `config/` are merged over
it at CLI time.

---

## Base config (`quant_rl/config/default.yaml`)

| Section | Key paths | Purpose |
|---------|-----------|---------|
| `data` | `m1_files`, `tick_files`, `split.train_end`, `split.test_start` | Data sources and locked OOS split |
| `session` | `tz`, `start`, `end` | Broker-tz NY eligibility window (`16:30`–`23:00`) |
| `account` | `initial_balance`, `leverage`, `margin_pct` | Simulated account |
| `costs` | `spread_us100`, `use_tick_execution`, `point_size` | Fill pricing |
| `ftmo` | `daily_loss_limit`, `max_loss_limit`, `risk_per_trade_limit` | Training/eval dollar kill-switches |
| `live_risk_overrides` | `risk_per_symbol`, `max_total_risk`, `stop_loss_pips` | Live MT5 percent-of-balance sizing |
| `features` | `htf_timeframes`, opt-in flags (below) | Feature matrix composition |
| `env` | `obs_window`, `action_type`, `strategy_actions` | Gymnasium environment |
| `training` | `total_timesteps`, `max_days_mvp` | Training loop |
| `ppo` / `sac` | algorithm hyperparameters | SB3 agent config |

### Locked OOS split

```yaml
data:
  split:
    train_end: "2025-12-31"   # in-sample ≤ this date
    test_start: "2026-01-01"  # out-of-sample ≥ this date
```

Variant configs must **not** override these dates.

---

## Feature opt-in flags (`features:`)

All off by default in `default.yaml`. Enable via variant YAML or CLI override.

| Flag | Adds |
|------|------|
| `include_po3` | PO3 time-of-day phase tag |
| `include_fvg_ifvg` | Per-TF FVG zone features |
| `include_po3_full` | Full HTF-FVG → LTF-IFVG → entry-trigger pipeline |
| `include_strategy_state` | Sweeps, BOS, PO3 state, IFVG zones (required for Idea 1/2 overlay) |
| `include_session_ohlc` | CT-anchored session OHLC, prior-period H/L, quadrants |
| `smt` | `{enabled, timeframes}` — MTF SMT divergence |
| `structure` | `{enabled, timeframes}` — MTF structure levels and BOS |
| `liquidity` | `{enabled, timeframes}` — MTF liquidity sweeps |
| `po3_state_mtf` | `{enabled, timeframes}` — MTF PO3 manipulation/distribution |
| `ifvg_mtf` | `{enabled, timeframes}` — MTF IFVG active zones |

---

## Strategy overlay configs

Selected via `--strategy` on `train_rl.py`. Merged over `default.yaml`.

| CLI flag | YAML file | Idea |
|----------|-----------|------|
| `--strategy baseline` (default) | none | Idea 3 — legacy discrete/continuous actions |
| `--strategy po3_ifvg` | `config/idea1_po3_ifvg.yaml` | Idea 1 — PO3 + IFVG overlay |
| `--strategy distribution` | `config/idea2_distribution.yaml` | Idea 2 — distribution overlay |

Strategy overlays require **both**:

```yaml
features:
  include_strategy_state: true
env:
  strategy_actions: true
strategy:
  name: "po3_ifvg"   # or "distribution"
  # ... gates, risk ranges, TP targets, reward weights
```

---

## Feature variant configs

Passed via `--config` on `train_rl.py`. Control MTF feature expansion without
changing the strategy overlay.

| File | Enables |
|------|---------|
| `config/features.yaml` | Base feature set reference |
| `config/features_technical_mtf.yaml` | Technical indicators on M5/M15/H1 |
| `config/features_po3_mtf.yaml` | PO3 phase tag on multiple TFs |
| `config/features_fvg_ifvg_mtf.yaml` | FVG/IFVG zones on multiple TFs |
| `config/features_full_po3_mtf.yaml` | All three Chain-F blocks (`include_po3`, `include_fvg_ifvg`, `include_po3_full`) |

Example:

```bash
uv run python -m quant_rl.train.train_rl --mvp \
    --config config/features_full_po3_mtf.yaml --seed=42
```

---

## Cache versioning

| Constant | File | Current value | Busts |
|----------|------|---------------|-------|
| `BAR_CACHE_VERSION` | `quant_rl/data/pipeline.py` | `v2-full-day` | Stale bar parquet caches |
| `FEATURE_CACHE_VERSION` | `quant_rl/features/build.py` | `v11-content-hash` | Stale feature parquet caches |
| Content hash | `feature_cache_content_hash()` | 16-hex digest of version + features cfg + tz keys + data identity + optional `train_mask` | Flag flip or data change must miss cache |

**Rule:** bump the relevant constant whenever the output schema changes.
All runs share one feature cache key; changing opt-in flags without bumping
the version requires `--force` or deleting the stale parquet.

Cache file pattern:

```
cache/{symbol}_{tf}_{BAR_CACHE_VERSION}.parquet          # bars
cache/{symbol}_features_{FEATURE_CACHE_VERSION}_{content_hash}.parquet
cache/{symbol}_features_{FEATURE_CACHE_VERSION}_{content_hash}.parquet.hash

```

---

## How configs compose

1. `quant_rl/config/default.yaml` is always the base.
2. `--config path/to/variant.yaml` merges over the base (OmegaConf).
3. `--strategy po3_ifvg|distribution` additionally merges the strategy YAML
   from `_STRATEGY_CONFIGS` in `train_rl.py`.
4. Positional `key=value` overrides on the CLI merge last.

```bash
# Base + feature variant + CLI override
uv run python -m quant_rl.train.train_rl --mvp \
    --config config/features_full_po3_mtf.yaml \
    --strategy po3_ifvg \
    env.obs_window=30
```

The run's merged config is saved to `outputs/<run>/config.yaml` for
train/live parity.

---

## Other config files

| File | Purpose |
|------|---------|
| `config/env.yaml` | Environment overrides reference |
| `config/reward.yaml` | Reward function overrides |
| `config/vae.yaml` | VAE standalone training |
| `config/wandb_sweep.yaml` | Weights & Biases sweep |
| `config/symbols_config.yaml` | Live multi-symbol selection (`live_trading.py`) |

---

## References

- [architecture.md](architecture.md) — pipeline contracts and overlay invariant
- [RUNNING_COMMANDS.md](operations/RUNNING_COMMANDS.md) — CLI examples
- [DEPLOYMENT.md](operations/DEPLOYMENT.md) — live risk alignment (`ftmo` vs `live_risk_overrides`)
