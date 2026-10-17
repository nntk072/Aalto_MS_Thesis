# Independent thesis-evidence review brief — 2026-10-16

**Status: preparation only. No independent review or sign-off has occurred.**
**Evidence refresh (2026-10-17):** the original pre-repair defect remains
documented as historical evidence. Bounded post-repair checks passed, but P1 is
still open; see the scope limits below and the master plan.

**Replacement-data refresh (2026-10-18):** the source/cache schedule now covers
training through 2025-12-31 and one continuous report-only test interval from
2026-01-01 through 2026-10-16. January–July outcomes were previously inspected,
so the whole interval is exposed exploratory evidence; no later subperiod is
fresh or confirmatory. The full-range M1/MTF/tick caches, four
resolved feature profiles (the three requested caches plus a baseline
strategy-action profile), and exact resolver schedule passed the stated bounded
parity/coverage checks. A 615-archive text screen found date references in 15
archives; triage found source/config/manifests and two resource-telemetry
snapshots, but no direct market-data or strategy-result member paths. This
does not prove external or human non-exposure. Raw tick headers identify an
MT5-style export format. The user identifies FTMO as the prop firm; FTMO's
current [public terms](https://ftmo.com/en/terms-and-conditions/) state its
platform/data feed is powered by liquidity providers, but do not identify the
underlying feed for this account or establish data-use/publication rights for
these files. The user has explicitly authorized these FTMO Demo exports for
internal model training and considers them adequate for that purpose; the
underlying liquidity providers are unknown, and no account-specific export or
publication record is available. Record this as user-attested training use,
not as independently verified source or publication permission. These gaps are
disclosure and public-artifact questions, not a withdrawal of training
authorization. See the
[replacement-data report](evidence/replacement_data_20261017.md), including the
successful full-period data and 18-variant feature-cache rerun; the user report
and remaining provenance gaps are machine-recorded in
[`provider_provenance_status_20261018.json`](evidence/replacement_data_20261017/provider_provenance_status_20261018.json).
See also the
[archive triage report](evidence/replacement_data_20261017/candidate_holdout_archive_exposure_20261017_triaged.json).

This brief is a short entry point for an independent reviewer of the thesis
evidence. The full evidence inventory and review questions are in the
[independent research review checklist](independent_research_review_checklist_2026-10-14.md),
and the authoritative execution queue is the
[master project execution plan](master_project_execution_plan.md).

## Review purpose

Assess whether the planned thesis claims and evidence gates are scientifically
defensible. This review is not a strategy endorsement, a substitute for
reproducing the analyses, or permission to treat incomplete evidence as final.
The P4 validity gate remains open.

## What can be reviewed now

1. **Causal-feature failure and repair scope.** The immutable pre-repair
   [report](g1_real_market_fold_fit_scope_2026-10-16.json) and
   [same-session future-poison regression](../../tests/test_features/test_session_ohlc.py)
   recorded model-visible `prev_ny_high`/`prev_ny_low` leakage in three feature
   profiles mapped to 17/18 saved worker configurations; that regression
   failed on the pre-repair source. The causal feature repair and regression
   were subsequently applied. A bounded post-repair
   [purged-fold poison report](g1_real_market_fold_fit_scope_post_array_opt_2026-10-17.json)
   passed for four representative feature profiles on one early fold, and the
   [prefix-invariance report](g1_feature_prefix_post_repair_2026-10-17.json)
   passed 12 sampled comparisons across four profiles. These results support
   only the tested source/config/data slices. They do not prove exhaustive
   causality, establish a valid market calendar, or clear G1/P1. The old
   campaign remains contaminated and cannot support selection or confirmation.
2. **Bounded evaluator evidence.** The plan's P2 section and linked audit
   reports document deterministic repeats and frame/shared-bundle parity for
   the legacy `ladder_a4b_distribution__seed50` checkpoint across its historical
   train and already-inspected test splits. The exact source cache and explicit
   legacy-schema opt-ins were used. Review the checkpoint, split, and backend
   scope in each report; this evidence does not close acceptance on a clean,
   admissible completed checkpoint or all schema/compatibility requirements.
3. **Data provenance limitation.** The user identifies the source as FTMO Demo,
   authorizes internal model training, and reports the underlying liquidity
   providers are unknown. No account-specific export or publication terms are
   documented in the available project records. The data hashes and integrity
   checks establish file identity and observed properties, not independent
   source identity or publication rights. Assess what can be reported publicly
   and whether raw-data redistribution must be excluded; do not treat the
   missing provider record as absence of the user's training authorization.
4. **RQ1 design boundary.** The current 18-variant configs do not provide a
   ready-made matched control/treatment pair. P5 must freeze one after valid
   single-seed findings, aligning reward, observation/action semantics, risk,
   costs, data, training budget, runtime, and checkpoint selection except for
   the predeclared structural intervention.

## Decisions requested from the reviewer

- Does the fold-fit perturbation audit correctly detect information leakage,
  and is the current invalidation boundary (17/18 configurations) justified?
- After the repair, what additional causal, cache, or split checks are needed
  before G1 can pass?
- Are the evaluator's deterministic and backend-parity claims appropriately
  limited to the checkpoints, data splits, and fields actually compared?
- Is the proposed matched RQ1 contrast sufficient for the central thesis
  claim, given the current config differences and exposed 2026 interval?
- Given user-attested training authorization but unknown liquidity providers
  and no account-specific export/publication terms, what data-provenance
  disclosure is required for public reporting, and should raw source data be
  excluded from the submission package?
- Which result claims should be rejected, narrowed, or labelled exploratory,
  incomplete, negative, or inconclusive?

## Explicitly deferred from sign-off

Do not sign off the final selected pair, final RQ1 effect, G1/P1 closure, or
P7 robustness results from the current campaign. Remaining gates include
independent review of the bounded post-repair evidence, broker timezone/DST and
authoritative session-calendar validation, quote-aging/impact validity,
broader causal/cache checks, a valid fresh screen, frozen matched design/runtime,
and final paired-seed evidence. The user-authorized training-use statement is
recorded; unknown provider identity and unverified publication terms remain
limitations for disclosure and packaging.
The versioned cache rebuild is recorded in
[`prebuilt_cache_rebuild_2026-10-17_mi210.json`](prebuilt_cache_rebuild_2026-10-17_mi210.json).
The old 100k campaign's underlying reports and per-run artifacts are absent
from this checkout; its aggregate results cannot be independently recomputed
here.

## Reviewer record

Complete only after reviewing the cited artifacts and raising any required
corrections:

- Reviewer name and relevant expertise:
- Review date:
- Artifacts and source hashes reviewed:
- Findings by gate (G0/G1/G2/G3):
- Required corrections or additional evidence:
- Disposition: `pass` / `conditional` / `fail` / `not yet reviewable`:
- Signature or durable review-record reference:
