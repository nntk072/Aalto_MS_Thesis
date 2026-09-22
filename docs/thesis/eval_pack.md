# Thesis Eval Pack (T-04.1)

Frozen commands and artifact locations for chapter tables. Engine default:
**TradingEnv** (see [validation_protocol.md](validation_protocol.md)).

Locked OOS (`test_start: 2026-01-01`) is **final report only**.

## Frozen three-arm 20M campaign (seed 50)

Protocol: year episode; daily loss $5,000; max loss $10,000 from initial;
**trailing DD 7% from equity peak as a hard fail only** (`soft_trailing_dd_limit: 0`).
Do not plot a 7% line. Overlay PPO: `log_std` clamped to `[-0.7, 0.0]`
(`std` in about `[0.50, 1.0]`). Discrete Idea 3 keeps `ent_coef: 0.01`.

Launcher: `scripts/run_encoder_slot_tmux.sh` → `scripts/train_overlay_baseline_20m.sh`.
Slurm 20368731, log `outputs/idea123_20m_final_tmux.log`.

Identify Idea 3 by `training_log.json` `strategy: baseline` and
`strategy_actions: false` (saved `config.yaml` may still echo `strategy.name: po3_ifvg`).

| Role | Run dir | `strategy` / overlay |
|------|---------|----------------------|
| Idea 1 PO3/IFVG | `outputs/20260921_113043_rl_train_seed50` | `po3_ifvg`, `strategy_actions: true` |
| Idea 2 distribution | `outputs/20260921_135435_rl_train_seed50` | `distribution`, `strategy_actions: true` |
| Idea 3 unconstrained | `outputs/20260921_160121_rl_train_seed50` | `baseline`, `strategy_actions: false` |

### Main table (TradingEnv eval)

| Arm | Train Sharpe | Test Sharpe | Train / test trades | Test max DD | Test survived year | Test fail time |
|-----|--------------|-------------|---------------------|-------------|--------------------|----------------|
| Idea 1 | 0.00 | 0.00 | 0 / 0 | 0.00% | yes | — |
| Idea 2 | 0.00 | 0.00 | 0 / 0 | 0.00% | yes | — |
| Idea 3 | 0.06 | −2.28 | 181 / 82 | 3.87% | yes | — |

**Acceptance:** overlay arms must have **trades > 0** on train and test.
Idea 1 and 2 **fail**. Deterministic eval is entirely `hold_low_intensity`
(test 49,925 bars; train 99,504). `soft_brick` is 0. Overlay `std` stayed at
the floor (~0.51), so this is **not** the old `std` explosion and **not** the
4% soft trailing brick. The Box intensity mean sits below
`entry_intensity_threshold: 0.1` at every bar (hold if affine `u < 0.1`).
Do **not** treat these two folders as a successful overlay-vs-baseline
trading comparison. Idea 3 did trade (all shorts) and survived; test Sharpe is
poor.

Further overlay work (if any) should target intensity / hold-band, not
trailing DD. Do not retune on locked-OOS Sharpe.

### Negative controls (do not use as the main Idea 1 number)

Old seed-50 Idea 1 had **no** trailing-DD key and Gaussian `std` exploded
(~1 → ~3000). The first 7% trailing campaign used a 4% soft brick plus
`log_std_min: -2`, and Idea 1 learned a hold policy (`std` → 0.14).

| Run | What it is | Test Sharpe | Test trades | Notes |
|-----|------------|-------------|-------------|-------|
| `outputs/20260920_232943_rl_train_seed50` | Old Idea 1, no trailing DD | −1.18 | 215 | `std` ~3000; 61 test breaches; fail 2026-04-10 |
| `outputs/20260921_042429_rl_train_seed50` | Idea 1, 4% soft DD + `std` floor | 0.00 | 0 | hold-only |
| `outputs/20260921_064902_rl_train_seed50` | Idea 2, same soft-DD protocol | −1.50 | 18 | 53 train trades; survived |
| `outputs/20260921_085819_rl_train_seed50` | Idea 3, same protocol | +0.95 | 107 | 227 train trades; survived |

## Configs & seeds

| Role | Config / flags | Seed |
|------|----------------|------|
| Idea 3 baseline | `quant_rl/config/default.yaml` | **50** |
| Idea 1 PO3/IFVG + PD | `--config config/features_full_po3_mtf.yaml --strategy po3_ifvg` + session/liquidity/PO3/IFVG flags | **50** |
| Idea 2 distribution | `--config config/idea2_distribution.yaml --strategy distribution` | **50** |
| Ablation matrix | `config/experiments.yaml` | `defaults.seeds` |

## Commands

```bash
# Full three-arm 20M (login tmux + one GH200 srun)
bash scripts/run_encoder_slot_tmux.sh

# MVP smoke (not used for chapter tables)
uv run python -m quant_rl.train.train_rl --mvp --seed=50
```

## Artifacts

| Output | Path |
|--------|------|
| Frozen Idea 1 / 2 / 3 | `outputs/20260921_{113043,135435,160121}_rl_train_seed50/` |
| Campaign log | `outputs/idea123_20m_final_tmux.log` |
| Ablation JSON | `results/ablations/<variant>.json` |
| OOS cost grid | `results/oos_*.json` |
| Leak report | `results/leak_detectors.json` |

## Methods language

- **Trailing DD:** 7% from peak is a kill-switch like max loss from initial, not
  a “don’t trade” objective. Survival with zero trades is not a strategy result.
- **Distributional stats:** [distributional_methodology.md](../distributional_methodology.md)
- **Multiple testing:** report the number of variants / seeds tried. A full
  Deflated Sharpe / PBO / SPA package is deferred to Integrated_startup_plan
  plan 09; chapter prose should haircut claims qualitatively when many trials
  share the locked OOS.
- **Cost:** do not claim superiority on default costs alone; cite the ×2 / ×3
  stress cells.

## Validity docs

- [threats_to_validity.md](threats_to_validity.md)
- [../architecture/thesis_audit_2026.md](../architecture/thesis_audit_2026.md)
- [validation_protocol.md](validation_protocol.md)
