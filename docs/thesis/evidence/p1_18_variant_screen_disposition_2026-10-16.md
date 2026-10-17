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

### Cross-source reconciliation — 2026-10-17

The preserved campaign manifest contains 18 seed-50 job records
(`campaign.json`, SHA-256
`76a4ea35169386ebac57d07f46595b8aeaf5defe5392e50578b529aaf463b688`). The
controller status file has 11 succeeded, 4 failed, 3 interrupted, 0 pending,
and 0 running (SHA-256
`ab606086aa944b00215f2a12ae1fd435a224b5ceefc1b78f69b344d04013b89c`), matching
the 18 identities in the preserved summary. Its `telemetry_available_runs: 18`
means all 18 telemetry artifacts were present; it does **not** mean all
variants completed a train-split episode. The same report records 1,088
full-split episodes across 1,088 environments, exactly 17 × 64, and records
zero episodes for the failed `ladder_a0_flat` attempt. The original summary
JSON SHA-256 is
`6af7f6117b1f177f83c9bc5271df9dc9bd06d9352bfba99d56ff32c7ca948d7e`; its
schema-v1 campaign-level `seed: null` is a reporting defect because the
manifest and all 18 run records declare seed 50. Preserve that report. A
schema-v2 correction was generated under a new stem and its source plus
artifact provenance passed the then-current software gate. That gate predates
the current dirty-tree tick-ingestion and observation-schema changes and does
not validate them. Its SHA-256 is
`aeeb5eabab0fd568b619f368d4bb937265dbfd6f19bc32f1f85849af5cadbda7`; it
reports seed 50. No original report was overwritten.

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
seed 42. The schema-v2 source repair reports the actual seed set and only emits
a scalar seed for complete homogeneous metadata. Its source tests passed in
the earlier 1,664-test source snapshot, not the current dirty tree. The recovery
schema-v2 JSON SHA-256 is
`aeeb5eabab0fd568b619f368d4bb937265dbfd6f19bc32f1f85849af5cadbda7`; the
separate v3 schema-v2 JSON SHA-256 is
`edbd0312b473d118982a68b6c38ebff2e1c5d0b254092e8a727e63b4751355c1`.
Both embed the current summary-script SHA-256. This metadata repair does not
alter the campaign results, run identities, or fingerprint. The final gate
wrapper declined to regenerate either immutable report because they already
exist; no report was overwritten.

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
