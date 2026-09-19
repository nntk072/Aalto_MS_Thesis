# Thesis Eval Pack (T-04.1)

Frozen commands and artifact locations for chapter tables. Engine default:
**TradingEnv** (see [validation_protocol.md](validation_protocol.md)).

## Configs & seeds

| Role | Config / flags | Seed |
|------|----------------|------|
| Baseline | `quant_rl/config/default.yaml` | 42 |
| Idea 1 PO3/IFVG + PD | `config/idea1_po3_ifvg.yaml` (or `--strategy po3_ifvg`) | 42 |
| Idea 2 distribution | `config/idea2_distribution.yaml` | 42 |
| Ablation matrix | `config/experiments.yaml` | `defaults.seeds` |

## Commands

```bash
# MVP smoke (keeps train_rl runnable)
uv run python -m quant_rl.train.train_rl --mvp --seed=42

# Idea 1 MVP
uv run python -m quant_rl.train.train_rl --mvp --strategy po3_ifvg --seed=42

# Walk-forward on train calendar (selection)
uv run python -m quant_rl.train.train_rl --walk-forward --wf-splits 5 --seed=42

# Ablations — OOS is final_report_only unless flagged
uv run python scripts/ablation_runner.py \
  --bars-csv <bars> --features-csv <feat> --steps 8192 --seeds 42

# Cost stress (TI-7)
uv run python scripts/test_oos.py \
  --model-path outputs/<run>/model/ppo_final.zip \
  --bars-csv <bars> --features-csv <feat> \
  --cost-multipliers 0.5 1.0 2.0 3.0 \
  --out results/oos_cost_stress.json

# Leak detectors
uv run python scripts/leak_detectors.py \
  --features-csv <feat> --target-col ret_1 --feature-col rsi \
  --out results/leak_detectors.json
```

## Artifacts

| Output | Path |
|--------|------|
| Training runs | `outputs/<timestamp>_rl_train_seed*/` |
| Ablation JSON | `results/ablations/<variant>.json` |
| OOS cost grid | `results/oos_*.json` |
| Leak report | `results/leak_detectors.json` |

## Methods language

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
