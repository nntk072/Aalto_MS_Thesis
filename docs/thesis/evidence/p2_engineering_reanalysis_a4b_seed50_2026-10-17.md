# P2 Engineering Reanalysis — A4b Distribution Seed 50

**Date:** 2026-10-17

**Role:** legacy deterministic-evaluator engineering evidence only

**Status:** deterministic repeat and train/test frame/shared-bundle parity passed; evidence remains engineering-only

## Source and execution

- Source run: `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/runs/attempts/ladder_a4b_distribution__seed50/attempt1/ladder_a4b_distribution__seed50/20261016_173920_057388_rl_train_seed50_tcn`
- Checkpoint: `model/ppo_final.zip`; SHA-256 `7d81739de7e1bc54d818149a0bacd6525e58d1e06b4d7e88c2b060dc55f21ba3`
- Source cache: `cache/US100.cash_features_v20-mtf-swing-candidates_f7b463019fc2fde4.parquet`; identity `f7b463019fc2fde4`; file SHA-256 `87e65a8d4b775d10674f6b59193ef2662c8cab1d02daeebd938c30935b676fa9`
- Replay ran on existing MI210 allocation `21014127`, using the deterministic frame evaluator, exact source feature cache, and full train/test splits.
- Explicit legacy flags were required because the historical source manifest contains neither ordered sequence-feature schema nor stop-candidate schema. The evaluator reports `legacy_unverified` sequence disposition and `legacy_opt_in` stop disposition.
- The source lineage is from the contaminated 18-variant campaign with known NY-session lookahead. The test split was already inspected; this report is not pristine OOS evidence.

## One replay result

| Split | Bars | Trades | Total PnL | Return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Training, through 2025-12-31 | 351,550 | 6 | +$0.458 | +0.000458% | 0.4644 | 0.000638% |
| Historical test, from 2026-01-01 | 177,210 | 2 | -$0.658 | -0.000658% | -1.3261 | 0.000969% |

These metrics describe only this exploratory replay and do not indicate a viable or selected strategy. The checkpoint has minimal activity and nearly flat equity.

## Artifacts and integrity

Output directory: `outputs/thesis_engineering_reanalysis/a4b_distribution_seed50_20261017_run1/`

- Reanalysis manifest SHA-256: `c126838de50060da55b9a42df0fb8aea3db8cfb43106afed702c7d111e0ae888`
- Training metrics SHA-256: `99603058c0c4c342605faa1da7801ed4a19e58cef465e383b100a7b405463092`
- Test metrics SHA-256: `0e77cb68df6b47217646aa0171747ff689ee53819d7650d5d60c71f705570af5`
- Manifest marks checkpoint-manifest hash and dataset parity `verified`, cache disposition `source_cache_replay_verified`, historical holdout inspected `true`, artifact role `historical_reanalysis`, and the validity note “Historical inspected holdout; not pristine OOS evidence.”

## Repeatability and backend parity follow-up

- Independent exact replay outputs: `outputs/thesis_engineering_reanalysis/a4b_distribution_seed50_20261017_run2_gh200/` and `.../run3_gh200/`.
- Comparator: `docs/thesis/evidence/p2_repeatability_a4b_seed50_20261017_gh200_stop_legacy_fix.json`; status `passed`, `errors=[]`, with required train/test numerical artifacts byte-identical.
- Training frame/shared-bundle parity: `docs/thesis/evidence/p2_frame_bundle_a4b_seed50_train_20261017_gh200_retry1.json`; status `passed`, all 25 compared behavior/metric fields agree, 6 trades.
- Historical test frame/shared-bundle parity: `docs/thesis/evidence/p2_frame_bundle_a4b_seed50_test_20261017_gh200_retry1.json`; status `passed`, all 25 compared behavior/metric fields agree, 2 trades.
- Comparator and parity regression tests were added. Slurm accounting shows validation wrappers `21018094` and `21018146` ended `FAILED` with exit code 1; their stdout/stderr were not retained in the repository, so these jobs do not count as a successful current-tree CI gate. The individual comparator/parity JSON artifacts pass, but full current-tree verification must be rerun and logged on an eligible GPU allocation.

This closes only the bounded repeatability and backend-parity checks for this exact
legacy checkpoint/config/cache. It does not accept the checkpoint as a thesis
result or close P2 globally: its source campaign is lookahead-contaminated, the
historical holdout was inspected, the schema required explicit legacy opt-ins,
and no admissible completed checkpoint has passed the plan's broader acceptance.

## Remaining P2 acceptance work

1. Apply completed-checkpoint acceptance to a checkpoint from an admissible, uncontaminated, provenance-complete campaign after P1 is cleared.
2. Preserve all legacy and contamination limitations. This checkpoint cannot close P1/P3 validity, select a variant, or support confirmatory thesis claims.
