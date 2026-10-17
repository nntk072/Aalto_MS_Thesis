# Finding P1-SCREEN-2026-10-16 — 18-variant seed-50 recovery disposition

**Status:** exploratory / invalid for variant selection
**Evidence role:** one-seed screen reconciliation and failure disposition
**Campaign:** `ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016`
**Slurm job:** `20990395`
**Fingerprint:** `91d2f05c4c13e67753a47b88a1c900f4887ced66dad51980a018eb09465dd10e`

## Finding

The campaign summary records 18 variant identities: 11 controller-succeeded,
4 failed the train-equity survival gate, and 3 were interrupted before the
requested 20M-step budget. The allocation later ended with exit `124:0` at its
six-hour limit. The gate failures were `ladder_a0_flat`, `ladder_a2_tp`,
`ladder_a3_multi_tp`, and `ladder_a0_flat_dist`.

Per-run telemetry records one completed 99,504-step train-split episode for each
of 64 environments in 17 variants (1,088 environments total). The current
recovery attempt has no completed-split episode telemetry for
`ladder_a0_flat`. This is separate from the earlier source-lineage budget audit,
which verified a full-split traversal in all 18 source lineages before the
recovery attempt. PPO checkpoints do not restore vector-environment cursors;
the earlier traversal cannot be attributed to the current attempt.

Eight variants are `INACTIVE` with zero trades in both training and
report-only test evaluation: `ladder_a1_sl`, `ladder_a4b_entry_fsm`,
`ladder_a4a_entry_fsm_hidden`, `ladder_a0_flat_po3`, `ladder_a2_tp_po3`,
`ladder_a3_multi_tp_po3`, `ladder_a1_sl_dist`, and `ladder_a2_tp_dist`.
The remaining nonzero-activity rows do not establish a valid strategy result.

The real-market fold-poison audit found the model-visible
`prev_ny_high`/`prev_ny_low` lookahead in 17 of 18 saved run configurations.
The sole profile that passed that bounded poison check, `ladder_a0_flat`,
failed its train-survival gate. Therefore this campaign provides no run that is
both causally admissible and passes the declared train-survival gate. Retain
all attempts as exploratory failure/inactivity evidence; do not select a
variant or make a confirmatory claim from them.

## Numerical and resource boundaries

The machine summary reports 32 verified PnL reconciliations, zero mismatches,
and 4 unavailable reconciliations. Its sampled GPU utilization and VRAM fields
have zero samples, so this report cannot support GPU-utilization claims. The
evaluation protocol records previously inspected, selection-acknowledged data;
it is not pristine confirmatory OOS evidence.

### Seed-summary correction — 2026-10-17

The campaign manifest declares seed `50` for all 18 jobs, and every per-run
summary row retains seed 50. The original schema-v1 report incorrectly has a
null campaign-level `seed` because its generator hard-coded scalar reporting to
seed 42. A schema-v2 source repair now reports the actual seed set and only
emits a scalar seed for complete homogeneous metadata. Its regressions and a
new descriptive report are still pending verification; retain the original
report unchanged until that gate passes. This metadata defect does not alter
the campaign results, run identities, or fingerprint.

## Evidence

- Campaign summary JSON: `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/campaign_results_summary.json`
- Campaign summary Markdown: `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/campaign_results_summary.md`
- Controller/allocation logs: `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/logs/`
- Real-market feature causality audit: `docs/thesis/g1_real_market_fold_fit_scope_2026-10-16.json`
- Separate earlier-lineage budget audit: `docs/thesis/p1_checkpoint_full_split_budget_2026-10-16.json`
- Live scheduler snapshot: `docs/thesis/live_status_2026-10-16_2350.md`

## Allowed interpretation

This is a reconciled, single-seed exploratory screen with full-split exposure
evidence for 17 current-attempt variants, four observed train-gate failures,
three budget interruptions, widespread zero-trade inactivity, and a
model-visible causality defect in 17 configurations. It is not evidence that
any variant improves trading performance or survives correctly after a full
year of training.
