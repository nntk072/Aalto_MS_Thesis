# Threats to Validity

Examiner-facing note for the MSc. This records current limitations against the
single operational queue in
[`master_project_execution_plan.md`](master_project_execution_plan.md). Do not
describe a mitigation as closed until its current evidence gate passes.

| Threat | Finding | Mitigation | Residual risk |
|--------|---------|------------|---------------|
| Short sample / few regimes | F1 | Accepted limitation; do not claim decade-scale robustness | High for regime generalisation |
| Feed identity, data-use rights, and broker clock/DST | File headers and repository metadata identify MT5-format exports but do not establish the provider, broker, server-clock/DST schedule, or permission to use and publish the data | Obtain authoritative feed and license records; until then report the configured fixed `Etc/GMT-3` interpretation as an assumption and make no provenance or publication-rights claim | High; G1 remains open |
| Locked OOS reused for selection | F4 | The researcher explicitly authorized exploratory development use of the configured 2026 period on 2026-10-13; all such use is non-confirmatory, and a new future holdout is required | High for any claim treating that period as pristine or confirmatory |
| Chronological training exposure versus active trading | The checkpoint-lineage audit verifies that all 18 current model lineages had 64/64 environments complete a 99,504-step train-split episode in attempt 2. Across 1,152 training episodes, 621 recorded `trailing_dd`; two variants had zero trades in all 64 episodes, and all 16 variants with trades had negative mean episode-equity deltas. These are training diagnostics, not final evaluation PnL. The current 14M-resume baseline also has budget for only 93,750 additional steps/environment, too few for a second complete split ([budget audit](p1_checkpoint_full_split_budget_2026-10-16.json)) | Preserve this lineage evidence as bounded exploratory coverage. For thesis-final policies, require the same per-environment telemetry under the repaired causal feature path and pair it with trade/activity and risk records | High: the verified exposure and behavior metrics came from a campaign with model-visible feature leakage; the current continuation cannot add a second full split for A0. Its first terminal A0 attempt now has a deterministic report. The in-sample replay covered all 99,504 train-split steps but failed the survival gate at `2025-09-11`; it traded in 91 of 259 sessions. Its report-only test interval is previously inspected; no admissible final campaign evaluation exists |
| Overlapping WF test folds | F5 | T-03.1 non-overlapping expanding WF + `label_horizon` purge | Low after fix; legacy API deprecated |
| Stale or causally invalid feature cache | Existing cache hashes establish content identity, not causality. The three named caches used by the active recovery run were generated before the discovered feature repair; current-source cache mapping resolves affected profiles to 17/18 configs ([mapping audit](g1_campaign_feature_cache_resolution_2026-10-16.json)) | After the feature fix, increment the cache version, rebuild baseline/Idea 1/Idea 2 caches, and verify fresh raw-to-cache parity plus all 18 config mappings | Critical for the current lineage; its results are exploratory and invalid for model selection. G1 remains open until the repaired caches pass |
| Fold-local fitting and feature causality | Fold-aware normalization and bounded unit tests exist, but the 2026-10-16 real-market fold-fit poison audit found that validation perturbations change model-visible `prev_ny_high`/`prev_ny_low` in three feature profiles mapped to 17/18 configs ([audit report](g1_real_market_fold_fit_scope_2026-10-16.json)) | Repair `prev_ny_high_low`, add/run a regression, rebuild caches, and rerun the real-market prefix/fold poison audit on the frozen source | Critical; the current source path fails G1 despite prior unit, sampled-prefix, and raw/cache parity passes |
| Incomplete causality coverage | The earlier assembled-matrix truncation and sampled-prefix checks did not exercise the NY-session opening interval that exposes this defect | Require both broad prefix checks and real-market fold-local poison checks after the repair; record profiles, cutoff times, and source/cache hashes | High until the repaired audits pass; do not infer causality from the earlier bounded passes |
| Optimistic costs | F10 | T-03.3 cost grid; TI-7 — no superiority on default costs alone | Medium for live expectancy |
| Multi-simulator divergence | F9 | Eval pack names engine per table | Medium for live parity claims |

## Declared assumptions

1. The configured interpretation for session/PD features is **`Etc/GMT-3`** (UTC+3 fixed); the actual broker clock and DST behavior are unverified.
2. Primary thesis tables use **`TradingEnv`** episode evaluation unless a table caption says `run_backtest`.
3. Dates in `config/default.yaml` / `experiments.yaml` were originally intended as a **final holdout**. The configured 2026 period has since been inspected and, by explicit researcher instruction on 2026-10-13, is now exposed development evidence for exploratory tuning only. It cannot support confirmatory or untouched-holdout claims; see [validation_protocol.md](validation_protocol.md).
4. Cost-stressed results (see `scripts/eval/test_oos.py` and eval pack) bound default-cost claims.
5. The source feed and data-use/publication terms are not recorded in the repository. Data quality scans and matching file hashes do not establish provenance or usage rights.

## Related docs

- [thesis_audit_2026.md](../architecture/thesis_audit_2026.md)
- [validation_protocol.md](validation_protocol.md)
- [eval_pack.md](eval_pack.md)
- [distributional_methodology.md](../distributional_methodology.md)
