# Independent Research Review Checklist — 2026-10-14

**Status: OPEN — no independent reviewer has completed or signed this review.**

This checklist prepares the remaining independent review of the Aalto thesis
trading-system evidence. It is not itself independent sign-off, a strategy
selection, or evidence that a strategy works. The internal source review is in
[`research_path_review_2026-10-13.md`](research_path_review_2026-10-13.md); the
campaign decision and evidence boundary are in
[`18_variant_training_readiness_plan.md`](../training/18_variant_training_readiness_plan.md).

## Evidence to review

- The 18-variant 100k exploratory results:
  [`campaign_results_summary.md`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md)
  and its JSON report. All 18 training equity gates failed. Twelve test
  replays failed the 130-day survival requirement; the remaining six made at
  most two trades. All 36 saved train/test PnL values reconcile with their
  equity curves. The 2026 test interval was previously inspected, so these
  results are exploratory and report-only.
- The 100k resource observation:
  [`resource summary`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md).
  It records 95 samples, 94 active, PSS peak 251.2 GiB, sampled GPU
  utilization peaks 97%/96%, and lacks allocation-wide cgroup memory and
  process-tree CPU-time measurements. It does not establish a sustained 20M
  resource profile.
- The current-source real-market feature-prefix audit:
  [`feature_prefix_causality.json`](../../outputs/gpu_campaigns/readiness_reaudit_20261014/feature_prefix_causality.json).
  It passed 9/9 comparisons across the three registered profiles and three
  cutoffs; recorded source, script and raw-data hashes were rechecked against
  the current tree. Its scope is prefix invariance at those cutoffs only.
- The current-source full-calendar frames/shared-bundle audit:
  [`frames_bundle_parity.json`](../../outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.json).
  It completed all 18 variants, comparing a 49,925-step exposed test-calendar
  trace and a 10,000-step active-action trace per variant, with zero failures
  and maximum scalar delta 0.0. Its 166 source/config hashes, four raw-data
  hashes and dirty-path manifest were rechecked against the current worktree.
  The result is bounded to those fixed traces; review it alongside the
  execution/reward source contract, not as proof of exhaustive policy-state
  equivalence or strategy performance. See the
  [human-readable summary](../../outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.md)
  and [Slurm accounting](../../outputs/gpu_campaigns/readiness_reaudit_20261014/reports/slurm_accounting_20951231.psv).
- The [validation protocol](validation_protocol.md),
  [threats to validity](threats_to_validity.md), and current source review.

## Reviewer decisions requested

| Topic | Question for the independent reviewer | Current boundary |
| --- | --- | --- |
| Experiment design | Do the registered contrasts support the thesis claims? Which results, if any, are valid within each family? | Baseline, PO3/IFVG, and distribution families differ in reward and action/strategy semantics; do not interpret cross-family differences as isolated causal effects. |
| Execution and accounting | Are order timing, tick/quote fills, stops, take profits, costs, equity, and drawdown calculations correct for the stated market assumptions? | Code review and focused regressions exist; no independent broker integration or exhaustive reviewer sign-off exists. |
| Reward and risk | Are the training/evaluation breach penalties, shaped rewards, risk limits, and hard train-equity/survival gates appropriate and consistently interpreted? | Reward is not a profitability metric; all 18 pilot training gates failed. Do not relax gates to obtain a pass. |
| Feature causality | Are the MTF close-time alignments, swing/structure availability, train-only fitting, split boundaries, and cache identities causal? | On 2026-10-14, seven MTF/HTF/variant/swing/sweep/PO3/IFVG files passed 126/126 fixture tests in 5.76s; six causal-feature/split/fit-scope/cache/assembled-causality/leakage files passed 29/29 in 7.01s with two expected Transformer warnings; four entry-gate/eligibility/golden-trace files passed 30/30 in 3.50s (Python 3.12.14, pytest 9.1.1). Current real-market prefix invariance passed 3 profiles × 3 cutoffs with matching hashes. The current-source frames/shared-bundle audit passed 18/18 variants over the full 49,925-step test-calendar trace and 10,000-step active trace per variant; all 36 traces had maximum scalar delta 0.0, with matching source and raw-data hashes. Together these checks strengthen bounded feature/backend causality and parity evidence; they do not establish every execution/label/model-selection path or pristine confirmatory OOS. |
| Model selection | Is the selection protocol protected from exposed-period and human checkpoint-selection leakage? | The 2026 interval is permanently exposed. The one-seed campaign has no selected candidate and no independent selection record. |
| Performance claims | Do any results support viability, generalization, or confirmatory conclusions? | No. The 18-run pilot failed every training gate; six surviving test replays had at most two trades. No pristine holdout result exists. |
| Resource/reproducibility | Are the observed pilot resources sufficient for the proposed 20M campaign, and can the results be reproduced from retained inputs/source? | Not established. The 100k n_envs=64 shared-bundle probe used 0.35 average CPU cores per worker-step against 6 assigned CPUs (Slurm `TotalCPU`, 18 worker steps) and PSS peaked at 251.2 GiB. This is not the frozen frames/n_envs=4/20M shape; allocation-wide cgroup memory peak is unavailable, and the probe's launch-time source contents were not archived. New campaigns snapshot hashed source/config before launch but do not repair the historical probe. See `slurm_cpu_accounting_20944844.md` and raw `.psv` in the campaign reports. |

## Reviewer record

- Reviewer name and affiliation: ______________________________
- Relevant expertise / conflicts: ______________________________
- Materials and source revision reviewed: ______________________________
- Date: ______________________________
- Disposition: `ACCEPT` / `ACCEPT WITH CONDITIONS` / `REJECT` / `MORE EVIDENCE REQUIRED`
- Required changes or limitations to state in the thesis: ______________________________
- Signature or durable approval reference: ______________________________

Until a qualified reviewer fills this record and resolves the open items, the
independent-review gate remains **OPEN** and overall campaign readiness remains
**NOT READY**.
