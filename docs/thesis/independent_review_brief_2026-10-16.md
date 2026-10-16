# Independent thesis-evidence review brief — 2026-10-16

**Status: preparation only. No independent review or sign-off has occurred.**

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

1. **Causal-feature failure and its scope.** The real-market fold-fit audit
   [report](g1_real_market_fold_fit_scope_2026-10-16.json) and
   [reproduction script](audit_real_market_fold_fit_scope_current_source_2026-10-16.py)
   show that validation-window OHLC perturbations alter model-visible
   `prev_ny_high`/`prev_ny_low` in three feature profiles mapped to 17/18
   saved worker configurations. The
   [same-session future-poison regression](../../tests/test_features/test_session_ohlc.py)
   was run on `gpuarm1` with CUDA disabled and failed on the current source as
   expected (`1 failed, 17 deselected`). The active campaign source remains
   frozen. Current screen outcomes are not valid for variant selection or
   confirmation.
2. **Bounded evaluator evidence.** The plan's P2 section and linked audit
   reports document deterministic replays and frame/shared-bundle parity for
   interrupted checkpoints. Review the exact checkpoint, split, and backend
   scope in each report; this evidence does not close completed-selected-
   checkpoint acceptance or all schema/compatibility requirements.
3. **Data provenance limitation.** The feed/export provider and data-use terms
   are not documented in the available project records. The data hashes and
   integrity checks establish file identity and observed properties, not
   source or publication rights. This limitation needs an explicit disposition
   before the data are used in a public thesis artifact.
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
- What data-provenance evidence is required for public reporting, and what
  should be excluded if the provider or terms cannot be recovered?
- Which result claims should be rejected, narrowed, or labelled exploratory,
  incomplete, negative, or inconclusive?

## Explicitly deferred from sign-off

Do not sign off the final selected pair, final RQ1 effect, G1 repair, or P7
robustness results from the current campaign. These require the post-boundary
source repair, passing regression, versioned cache rebuild, repeated real-market
causal audits, valid fresh screen, frozen matched design/runtime, and final
paired-seed evidence. The old 100k campaign's underlying reports and per-run
artifacts are absent from this checkout; its aggregate results cannot be
independently recomputed here.

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
