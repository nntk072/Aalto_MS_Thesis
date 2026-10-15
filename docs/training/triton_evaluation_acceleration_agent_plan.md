# Agent plan: accelerate Triton evaluation and report generation

> **Plan routing:** Current order and next action are governed by the [master project execution plan](../thesis/master_project_execution_plan.md). This document is the technical evaluation/reporting reference; do not execute it as a separate queue.

Date: 2026-10-11
Repository: /scratch/work/nguyenl37/Aalto_MS_Thesis
Campaign: outputs/gpu_campaigns/pilot_18_seed42_100k/
Planning baseline HEAD: b00b5e8
Status: PLAN ONLY — do not modify or restart the running campaign.

## Objective

Shorten the elapsed time from final PPO checkpoint to a complete, scientifically valid campaign result. First target expensive chart/report generation, then profile the simulation evaluation itself. Keep all model decisions, reward calculations, entry gates, SL/TP behavior, risk checks, and acceptance criteria invariant. Never equate reduced plot production with faster evaluation simulation.

## Verified evidence

Existing ladder_a0_flat__seed42 attempt 2 log (2026-10-11, EEST):
- 03:57:59 model saved.
- 03:58:00 OOS test starts; 04:01:24 OOS result (~3m24s).
- 04:01:24 in-sample test starts; 04:08:04 result (~6m40s).
- 04:08:04 saving run artifacts; 04:08:13 drawing training per-trade PNGs (sample 200/228).
- 05:05:39 starts testing per-trade PNGs (sample 200/375).
- 05:29:17 run saved (~81m13s after export starts).
- 05:29:20 exit 1 because TRAIN_EQUITY_FAILED; this is an acceptance failure, not an export exception.

These figures are differences between log timestamps, not per-function profiler measurements. Report export dominates this observed attempt. Training has already finished before export; slowing artifact generation stalls a --workers 1 campaign. Prior logs also showed attempt retries: inspect retry semantics without changing failure treatment.

Relevant implementation seen during inspection:
- quant_rl/train/train_rl.py evaluates test with deterministic=True, train with deterministic=False, then calls save_run().
- quant_rl/eval/export.py saves statistical artifacts and optional Matplotlib/Plotly charts, including per-trade charts.
- cfg.output includes save_plots, save_html, save_csv, dpi, and max_order_charts.
- quant_rl/eval/plots.py implements plot_per_trade_orders().
- Other chart modules include trade_plots, distribution_plots, calibration and training_plots.

## Operating constraints

1. Inspect Git status, live Slurm jobs, controller state, scripts and artifacts before making a change. Preserve all pre-existing dirty files and untracked files. Work only in the relevant evaluation/export and tested orchestration paths.
2. Do not stop, restart or duplicate the currently running 18-variant campaign; changes affecting a running process require a separately approved transition after it has finished. Do not run new jobs outside an approved allocation.
3. Never edit reward/risk logic, trading config, train/eval market splits, agent policy, PPO hyperparameters, seed, entry gating, SL/TP, fill prices, costs, episode termination, OOS survival thresholds, or scientific acceptance conditions.
4. Do not suppress TRAIN_EQUITY_FAILED or convert it into success. Do not silently skip in-sample or OOS evaluation; if a future fast mode omits anything required, explicitly label the partial result.
5. Record workload, CPU architecture, active env, allocation, git SHA, source data, number of trades, chart count, initial process state, and wall-clock timings.

## Phase 0 — Inspect and establish baseline (read only)

- Read docs/operations/triton_environment.md, docs/operations/triton.md, .agents/rules/triton-slurm.md and the project agent rules.
- Check running campaign status and worker logs, scheduler.jsonl, campaign_status.json and per-variant PHASE_TIMING lines. Collect successful and failed examples; avoid basing speed claims on one run.
- Read quant_rl/config/default.yaml and any overlay settings; determine actual effective output flags, max_order_charts and DPI for the current campaign. Determine whether training and testing each export as many as 200 PNGs and whether HTML is enabled.
- Inspect save_run(), plot_per_trade_orders(), order_window preparations, trade_plots, distribution_plots and the controller's attempt/retry/accounting behavior.
- Determine whether plots execute serially and whether there are repeated feature calculations, datetime index construction, matplotlib initialization, disk writes, or network filesystem metadata operations.

Baseline table (fill measured values per run):
variant | attempt | PPO time | OOS eval | train eval | metrics/CSV | summary plots | per-trade PNG train | per-trade PNG test | HTML | total export | end-to-end | output MB | status

## Phase 1 — Instrument timing and count work

- Add monotonic perf_counter timing around each *report-export* subphase and per-trade chart batch, plus per-trade count, output bytes, and failures.
- Write a structured JSON timing report into each run directory (and summarize it in campaign reports); log an event before/after chart batches so controller progress stays observable even if stdout is buffered.
- Ensure timing and instrumentation are passive: no additional RNG consumption, no modification of observations/actions/metrics.
- Consider per-chart sample timing (first 5, median 20, p95 if enough charts) and log every 10-25 charts or 30 seconds; avoid printing sensitive data.
- Capture user CPU/system time, wall time, process-tree PSS, and filesystem write throughput if accessible. Measure on Lustre and node-local scratch with comparable configurations.

## Phase 2 — Separate scientific evaluation and artifact rendering

Implement an explicit, documented, backward-compatible output profile:
- full: current artifacts and charts, same defaults for existing scripts and final thesis/report runs.
- campaign_fast: complete mandatory train/OOS inference, risk/equity metrics, gate JSONs, model checkpoint, provenance/config snapshots and machine-readable report/CSV; omit or limit optional per-trade PNG and HTML *only*, based on explicit profile; produce a manifest listing deferred artifacts.
- plots_only/postprocess: deterministic render from saved trades/equity/bars/config, without training or reevaluating a policy; idempotent, atomic outputs, resumable per run, supports sampling and bounded concurrency. In case source data is insufficient for exact plot recreation, identify the missing values and add lossless export; do not recompute scientific metrics with new assumptions.

Clarify whether the existing max_order_charts=0 means unlimited; do not guess. Explicitly set bounded values for pilot use, e.g. zero per-trade PNGs in campaign_fast while retaining required summary charts if acceptable; choose representative cases for full chart generation. Separate lightweight run-ready report from graphics-ready report.

Important: The final report's scientific status MUST depend on the existing numerical evaluations and gates. Plot failures can mark visualization incomplete but should not silently hide numerical failures, erase checkpoints, or encourage unnecessary retraining; check existing consumer expectations and tests before making status changes.

## Phase 3 — Profile and optimize expensive rendering

After measuring actual hot spots, prioritize:
1. Reuse prepared session bounds, aligned indexes and indicator arrays shared across chart windows.
2. Build reusable plotting context once per split; avoid repeated O(N) data scans per trade and dataframe copies. Verify chart contents on representative long/short/SL/TP/overnight/PO3/IFVG trades.
3. Close Matplotlib figures promptly; use Agg; avoid costly high-DPI output by default for pilot-only visualizations.
4. Render optional chart batches from a bounded, separate CPU process pool only if benchmarks show benefit and memory/I/O permit. Use spawn-safe workers and avoid shipping full market DataFrames for each chart; account for Lustre contention.
5. Benchmark rendering to $SLURM_TMPDIR or safe node-local /tmp, followed by verified atomic publication to persistent campaign storage. Do not store unique required results exclusively on ephemeral disks.
6. Avoid duplicate PNG + HTML work unless explicitly requested; log artifact counts and total size.

Do not parallelize core evaluation first. It is currently ~10 minutes versus ~81 minutes of export in the observed attempt. If optimizing core evaluation later, check inference sequence, vectorized/deterministic behavior, state transitions and RNG consumption.

## Phase 4 — Controller and campaign throughput

- Determine why a failed TRAIN_EQUITY_FAILED attempt is retried; distinguish transient execution failure from deterministic model/risk rejection. Avoid redoing PPO training only to rediscover a fixed gate failure. Such a classification change must preserve an explicit failed status and the original acceptance policy.
- Add progress/ETA derived from completed variant phases and actual durations, reporting training, OOS, in-sample, export separately.
- Assess throughput as completed scientifically evaluated variants per GPU-hour and wall-hour, including I/O contention, not only PPO steps/s.
- Consider asynchronous deferred visualization after the campaign controller records numerical completion; bound CPU/RAM and ensure ownership, idempotency and failure reporting.

## Phase 5 — Validation gates

A. Snapshot one representative completed run's serialized mandatory results. Repeat the identical fixed evaluation (no training) under old/full and campaign_fast with same model/data/seed/config.
B. Assert exact or explicitly justified tolerance equality for actions, trades, SL/TP prices, order timing, account/equity/balance series, reward and risk outcomes, train/OOS Sharpe, DD, return, trade count and gate outcomes.
C. Confirm config/provenance/checkpoint/metrics/CSV files survive both modes. Full-mode artifact filenames and visual contents should remain backward compatible.
D. Check chart sample equivalence for long, short, partial TP, risk-terminated and entry-state variants; compare output structure and representative image render.
E. Fault-inject one chart generation failure. It must be visible, recoverable via postprocess, with scientific status and source numerical artifacts intact.
F. Verify resumability, idempotency, concurrent write safety, failure propagation, open file handles and no runaway memory.
G. Confirm pre-existing unrelated changes remain untouched, run targeted tests, mypy/ruff if available under the correct compute architecture, and git diff --check.

## Benchmarks and acceptance criteria

Matched experiments: one representative run with 200+ trades, same node, data, plots, DPI, output configuration, existing model, number of charts and file storage. Compare baseline full versus optimized full, then campaign_fast + subsequent deferred full; separate intrinsic plot acceleration from skipped/deferred work.

Report median over >=3 runs where practical, variability, process CPU time, wall time, peak PSS, Slurm cgroup MaxRSS, file count/size and failures. Quantify both checkpoint-to-metrics and checkpoint-to-all-charts time. Aim for >=80% reduction in time from completed evaluation to campaign-ready *numeric* report; a target, not a measured result. Do not claim 80% improvement until measured. Use no new training jobs just to benchmark graphics if existing artifacts can support reproduction.

## Agent deliverables

1. Baseline and optimized report with actual timestamps, measured phase durations, chart counts, resource use, experimental configuration and caveats.
2. Implementation diff limited to approved evaluation/report/export components, unit/integration coverage, and exact CLI/config examples.
3. Campaign-fast, full, and plots-only documentation plus reproducible commands for Triton allocated nodes and tmux logs.
4. Parity comparison showing what was equal and which artifacts differ intentionally.
5. Progress and unresolved issues list. Do not commit/push or touch the active campaign unless separately authorized.

## Immediate next actions for the agent

1. Check Git status and squeue, then inspect config.output and the running controller's worker logs.
2. Extract PHASE_TIMING values from *existing* attempts and measure per-trade chart time without rerunning PPO.
3. Prepare a code diff with isolated report profiles and plotting instrumentation.
4. Validate saved numerical artifacts on fixed model/data; benchmark export-only workload with preserved output.
5. Report recommendations before proposing application to the running/production campaign.


## Implementation progress (2026-10-11)

Applied a first incremental implementation on the remote working tree:
- 'campaign_fast' and 'full' report profiles at the save_run boundary; the former forces optional per-trade image count to zero and disables interactive HTML, while retaining numerical exports and other existing summary artifacts.
- Per-split timing for numerical exports, static PNG, optional per-trade PNG, HTML, and other diagnostics in report_export_timing.json; periodic log phase markers at the beginning/end of expensive per-trade export.
- Command-line variant launcher supports QUANT_RL_FAST_REPORTS=1 and QUANT_RL_FULL_REPORTS=1; the existing full default is preserved unless explicitly overridden.
- Focused parity test for numerical output and metadata.

Remaining phases: export-only plotting postprocessor, CPU/memory profiling and chart rendering micro-optimizations, concurrent rendering evaluation, controller progress reporting, real-data matched scientific parity and quantitative timing benchmarks. These require follow-up validation and must not be declared complete prematurely.

Suggested usage inside an authorized allocation for future runs:
    QUANT_RL_FAST_REPORTS=1 VARIANT=ladder_a0_flat SEED=42 scripts/train/train_one_variant.sh
To request the full profile instead:
    QUANT_RL_FULL_REPORTS=1 VARIANT=ladder_a0_flat SEED=42 scripts/train/train_one_variant.sh

Avoid changing the currently running or historical campaign's configuration while it is active.


### Verified follow-up (2026-10-11)

- Existing tests/test_plots_export.py suite: 27 passed in 57.27 seconds under .venv-x86 Python 3.12, pytest 9.1.1 on login5.
- Ruff check: passed for touched Python modules; initial format check requested reformat of export.py and test_plots_export.py, and both were formatted successfully.
- Python compilation, Bash syntax and git diff whitespace check passed.
- No Slurm allocations were listed for nguyenl37 during this verification. No remote GPU benchmarks or real-data evaluation parity tests were run. The originally measured 81-minute export remains the baseline and speedup remains unmeasured.
- Added a regression test ensuring campaign_fast still emits summary PNG and metrics while suppressing per-trade order rendering; execute in the next focused test batch.

Do not report the full optimization plan complete until deferred export/postprocess, matched benchmarking, and scientific parity gates have been verified.
