# Codex pause handoff — Aalto thesis trading system

**Captured:** 2026-10-13 (Europe/Helsinki)
**Repository:** `/scratch/work/nguyenl37/Aalto_MS_Thesis`
**Branch / HEAD:** `main` / `b00b5e85622ec1410c8a11cc063cd52c87d3976d`

## Resume point

This note preserves the current state of the Aalto thesis trading-system work
while a separate plan in `/scratch/work/nguyenl37/Idea 1_2.md` is reviewed and
shaped. Finish that separate work first. Then resume this system from this note
and the linked plans; do not assume the thesis campaign became ready in the
meantime.

## Checkout state and preservation

At capture, `git status --short` showed **73 tracked modified paths and 43
untracked paths**. These changes predate this handoff and must be preserved.
This handoff itself is one additional untracked file. No commit or clean-tree
reset was made. The external plan and the two in-repository plans below are
available in this checkout:

- [Agent implementation plan](/scratch/work/nguyenl37/Agent_Implementation_Plan.md)
- [18-variant readiness plan](/scratch/work/nguyenl37/Aalto_MS_Thesis/docs/training/18_variant_training_readiness_plan.md)
- [Implementation continuation ledger](/scratch/work/nguyenl37/Aalto_MS_Thesis/docs/thesis/implementation_ledger_2026-10-13.md)

The external implementation plan's objective is to repair and verify execution,
realised-reward accounting, action-schema compatibility, and evaluation
validity before a new thesis campaign. The follow-on campaign specification is
one seed (42), 18 registered variants, 20,000,000 requested steps per variant,
frames backend, and report-only final evaluation. The production campaign has
not been launched.

## Work verified complete

- Full local verification passed: `ruff check .`, `ruff format --check .`,
  `mypy .` across 389 source files, `git diff --check`, and full pytest:
  **1,351 passed, 38 warnings**, exit code 0. Evidence:
  [pytest log](/scratch/work/nguyenl37/Aalto_MS_Thesis/outputs/gpu_campaigns/readiness_verification_20261013/logs/full_pytest_final.log)
  and `outputs/gpu_campaigns/readiness_verification_20261013/logs/full_pytest_final.exit`.
- All 18 variants passed the recorded real-market frames/bundle parity traces
  (49,925-step calendar trace plus 10,000-step active-action trace per
  variant); maximum scalar delta was 0.0. This is bounded tested-trace
  evidence, not exhaustive policy-state proof:
  `outputs/gpu_campaigns/real_market_frames_bundle_parity_20261013/reports/frames_bundle_parity.json`.
- Real-market feature prefix invariance passed six profile/cutoff checks, and
  the train/test seam audit passed four checks with zero changed features and
  max delta 0.0. This covers sampled feature-prefix behavior only; it does not
  certify execution, reward, termination, or model selection:
  `outputs/gpu_campaigns/real_market_feature_causality_20261013/reports/prefix_invariance_profiles.json`
  and `.../reports/train_test_seam.json`.
- Controller attempt provenance and retry/failure artifact checks are recorded
  in the readiness plan and ledger. The 42 unrecoverable historical A2/A3
  checkpoint archives were explicitly excluded from new candidate ranking and
  confirmatory claims; originals were preserved:
  [disposition record](/scratch/work/nguyenl37/Aalto_MS_Thesis/docs/thesis/legacy_a2_a3_checkpoint_exclusion_2026-10-13.json).
- The historical 2026 test period was already inspected and is exposed,
  selection-acknowledged development evidence, not an untouched holdout. The
  available data ends on **2026-07-03**:
  [exposure audit](/scratch/work/nguyenl37/Aalto_MS_Thesis/docs/thesis/historical_oos_exposure_audit_2026-10-13.json).

## Current campaign result and readiness decision

**Status: NOT READY for confirmatory thesis claims or the 18-run production
campaign.**

- The bounded A0/A1 seed-42 100,000-step exposed comparison failed its required
  training-equity acceptance for both variants. Its 2026 outcomes are
  development evidence and cannot serve as final OOS results.
- The single-worker A0 resource profile requested 2,000,000 steps and completed
  2,007,040. The checkpoint loads on CPU, but the controller marked the run
  failed: training end equity was `$89,773.27` (−10.23%), equity slope was
  negative, peak trailing drawdown was 10.82%, and the max-loss stop triggered.
  Its test evaluation was inactive with zero trades over 49,925 steps. This is
  not a passing strategy or candidate result.
- Resource evidence is incomplete for production sizing. The profile sampled
  45.4 GiB process-tree PSS, 5,439 MiB GPU memory, and 28% peak sampled GPU
  utilization. Cgroup telemetry was unavailable; process I/O samples were
  unusable because that sampler process had loaded the pre-fix parser. Slurm
  job-step I/O deltas are allocation-wide, not worker-only. The saved summary
  is [here](/scratch/work/nguyenl37/Aalto_MS_Thesis/outputs/gpu_campaigns/resource_profile_a0_seed42_2m_exposed_20261013/reports/resource_profile_summary.json).
- Slurm job `20928644` is recorded as completed; its wrapper ended after the
  controller returned. Do not infer a current allocation from this record.
  No replacement GPU run was launched during this handoff capture.
- Fresh post-2026-07-03 holdout data is required. Remaining research review
  includes execution, reward/termination, checkpoint/model selection, and
  confirming the validity of any chosen candidate. Risk/equity acceptance must
  not be weakened to obtain a pass.

## Exact steps to resume the old plan

After the separate `Idea 1_2.md` work is complete:

1. Re-read this handoff, the [Agent implementation plan](</scratch/work/nguyenl37/Agent_Implementation_Plan.md>), the [18-variant readiness plan](</scratch/work/nguyenl37/Aalto_MS_Thesis/docs/training/18_variant_training_readiness_plan.md>), and the [implementation ledger](</scratch/work/nguyenl37/Aalto_MS_Thesis/docs/thesis/implementation_ledger_2026-10-13.md>). Recheck `git status`, branch, and HEAD first; preserve all dirty and untracked work.
2. Obtain fresh post-2026-07-03 US100.cash/US500.cash market data with the required M1/tick coverage, then validate provenance and split boundaries before using it as a final holdout.
3. Close the remaining execution/reward/termination and checkpoint/model-selection reviews. Decide on a candidate using development/selection data only, and require it to pass the existing equity and risk-survival gates.
4. Repeat a production-representative resource profile with valid process I/O telemetry and usable memory limits/peaks. Size the actual concurrency and allocation from measured sustained behavior.
5. Prepare and verify the immutable 18-key campaign dry run, manifests, restart plan, and final evaluation protocol. Launch the 18-variant campaign only after its open scientific/data/resource gates are closed and the author separately authorizes production launch with a valid allocation.

Do not treat successful lint/tests, backend parity on bounded traces, feature
prefix invariance, a loadable checkpoint, or resource utilization as proof of a
profitable strategy. Do not treat the exposed 2026 period as confirmatory OOS.
