# Implementation continuation ledger — 2026-10-13

## MTF swing and IFVG integration continuation

The M1 PO3 state can now consume causally aligned IFVG confirmations from all
configured source timeframes. Revised Idea 1/2 overlays enable M5/M15/H1;
`M30`, `H4`, and `D1` are also accepted when explicitly selected. Local swing
and structure features continue to be computed from each M5/M15/H1 source's
own candles, then become visible on M1 after the HTF confirmation candle
closes. Confirmed HTF sweep references remain M5/M15/H1 as specified in the
Idea decision record.

The aligned IFVG confirmation includes source-zone edges, a mapped M1 origin
index, and direction-specific source-timeframe codes. M1 has precedence for a
same-side confirmation on the same row; otherwise the configured timeframe
order is used. Feature cache schema is `v19-causal-mtf-ifvg`. New overlay IDs
are `po3_confirmed_ifvg_htf_mtf_v1` and
`distribution_prev_session_pd_mtf_v1`.

Verification: 92 focused tests passed across the PO3/IFVG state, variants,
feature build, HTF alignment, structure, pivot/swing, and sweep tests. Scoped
Ruff check, Ruff format check, and Mypy passed. This was feature-contract verification only. No training,
Slurm/campaign job, result generation, commit, or readiness-status change was
made; the frozen 18-run campaign remains unlaunched.

Repository: `/scratch/work/nguyenl37/Aalto_MS_Thesis`<br>
Branch: `main`<br>
HEAD: `b00b5e85622ec1410c8a11cc063cd52c87d3976d`<br>
Authoritative plan: `/scratch/work/nguyenl37/Agent_Implementation_Plan.md` (unchanged)

This ledger records the checkout inspected on 2026-10-13. The working tree was
already dirty at the start: 41 tracked paths modified and 28 untracked paths,
with no staged changes. Those existing changes were preserved. The current
checkout matches the historical HEAD from the plan; implementation candidate
changes are present as local modifications rather than commits. The v7 ZIP and
standalone delivery handoff files were not present in this workspace, so no
candidate-to-ZIP hash comparison was possible.

## 2026-10-14 continuation: output reset and 18-worker launch

At the researcher's explicit request, every generated item in `outputs/` was
deleted to start from a clean output directory. This removed the prior
campaign checkpoints, logs, reports, resource profiles, GPU/RAM probes, and
figures; no backup archive was retained. The cancelled Slurm job
`20938752` had been verified terminal before cleanup. It stopped during A0 at
approximately 4.03M/20M steps; the 18-run campaign did not complete, and its
partial checkpoint is no longer present. Earlier report values retained in
this ledger are historical notes whose source artifacts were removed; they
cannot now be re-opened from this checkout.

A fresh one-seed, 20M-step-per-variant exploratory campaign is configured in
`config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_20261014.yaml`.
It covers all 18 registered variants (360M requested total steps) and marks
the already-exposed 2026 interval as `selection_acknowledged`. The new
allocation launcher is `scripts/matrix/run_campaign_allocation.sh`; the
controller now records each worker's assigned node in attempt provenance,
starts pinned Slurm steps with `--overlap --exact` (sharing the allocated GPU
while limiting each worker to its requested CPU slice), and launches a resource sampler on
each node. Workers publish their local process IDs so each sampler can report
worker-tree PSS/I/O plus that node's GPU use. Controller output is tee'd to a
durable campaign log. The selected shape is two GH200
nodes, one allocated GPU per node, 64 CPUs and 900G RAM per node, with nine
workers per GPU and seven CPUs per worker. The 18-run dry-run, Python compile,
and Bash syntax checks pass. Slurm allocation request `20942600` was cancelled
after the `aalto_users` GH200 group run-minute limit rejected the five-day
request. A six-hour `srun --test-only` for two nodes, one GH200 per node, 64
CPUs and 900G RAM per node forecasts `gpuarm1` and `gpuarm2` at
2026-10-16 13:32:18; this is not a reservation. No trainer has started yet.
The new campaign remains exploratory and cannot
repair the failed A0 pilot gate or provide confirmatory evidence.

### Train/evaluation hard-breach reward alignment

The prior source-path review found episodic training used a zero terminal
breach penalty while evaluation used `-10.0`. Updated `TradingEnv` so both
paths use `-10.0`; the flight-recorder wiring check now reports mismatches
instead of treating any nonzero training penalty as an incident. The strict
train equity/survival gate is unchanged. Verification: campaign artifact,
sampler, year-survival, golden-trace and deterministic-checkpoint suites pass
(34 tests), and scoped
Ruff checks pass. The queued
campaign has not started, so no full-training impact has been measured; the
exposed A0 failure remains failed and independent research sign-off remains
open.

The train-calendar `ppo_best` diagnostic now uses deterministic policy actions.
With the campaign's final-timestep evaluation schedule, it is a reproducible
copy of the final policy plus an in-sample score, not selection among interim
checkpoints. A dedicated regression asserts deterministic action prediction.

### Resume-path and launch-supervisor correction — 2026-10-14

The first focused verification exposed that `hash_run_artifacts()` had lost its
return expression during a prior edit. Restored the checksum mapping and added
a regression showing failed attempts retain artifact hashes and remain
terminal on `--resume`. PPO checkpoint continuation now preserves the loaded
value head; a fresh run retains the deliberate zero initialization. The
checkpoint selector chooses the latest CRC-valid numbered checkpoint only
when attempt provenance matches the campaign fingerprint, key and attempt.
Campaign identity now uses hashes for runtime/config inputs rather than a hash
of the entire working-tree diff, so thesis ledger/status edits do not prevent
continuation while runtime source changes still fail closed.

Added `scripts/matrix/run_campaign_18way_supervisor.sh` to request repeated
six-hour allocations until the campaign is terminal, with a bounded retry
limit when Slurm cannot create a campaign manifest. The parent request uses
`--gpus-per-node=gh200:1`; the earlier form `--gpus=gh200:1` was rejected by
Slurm because it requested fewer GPUs than tasks. The current `--test-only`
request accepts the shape and forecasts 2026-10-16 13:32:18 on `gpuarm1` and
`gpuarm2`. Verification after these corrections: 33 focused tests pass;
scoped Ruff, shell syntax, Python compilation and `git diff --check` pass. The
campaign is not yet submitted. Its nine-workers-per-GPU/eight-env concurrency
remains an experiment to measure, not a validated capacity claim. The
requested output purge left `outputs/` empty, so prior reports cited above are
historical and their source artifacts cannot be reopened. Readiness remains
**NOT READY**: A0 failed equity/survival acceptance, independent research
review remains open, and the exposed data is exploratory only.

## Work-package status

| WP | Status | Evidence and remaining acceptance gap |
|---|---|---|
| WP00 | PASS — historical baseline | The 2026-10-12 audit records the baseline and pre-fix evidence. This continuation confirmed current HEAD and dirty state. Dataset/checkpoint hashes and a packaged handoff were unavailable; this is not a new immutable artifact freeze. |
| WP01 | PASS — declared schema, full archive census, explicit legacy migration | Action meanings come from declared flags/schema keys; new checkpoints persist schema version/key, and unversioned checkpoints require both opt-in and an exact layout selection. The read-only inventory covers all 301 output archives (all unversioned; widths 4/5/6/7/11), with SHA-256 hashes. One representative from each of 16 run directories loads on CPU; the 4D archive uses a named legacy TCN/policy migration and receives an appended structural-exit coordinate. Forty-two archives have no saved run config and remain unevaluable until that historical config is recovered; no layout or config was inferred for them. |
| WP02 | PASS — bounded accounting contract | Tick and bar paths reconcile partial TP, TP→SL, multiple fills, session closure, commissions, stable IDs, position totals, reward PnL and incremental R. Incomplete positions remain open; cooldown begins only on full completion. The backtest's single-target rows also carry stable IDs and net completed-position summaries. Full engine parity is evaluated separately under WP12. |
| WP03 | PASS — bounded causal trailing contract | Confirmed structure swings are unavailable until their right-side confirmation bar; a confirmed update is logged with its reference, decision timestamp and next-bar effective timestamp. Long/short reference sides, invalid references, finite/geometry checks, monotonic tightening, immutable initial risk, partial-position closure and gap-through fill are covered. The guard evaluates the existing stop before applying the current bar's update. |
| WP04 | PASS — environment tick/bar slot contract | Tick and bar paths preserve slot quantities and IDs, reconcile equivalent event sequences, process targets chronologically, close only the residual at SL, handle rounding/disabled/already-hit slots, and use stop-first for ambiguous OHLC. The simplified benchmark backtest still exposes one target; it is not the strategy multi-slot execution path. |
| WP05 | PASS — bounded causal timing/cost contract | Both environment and backtest equity points carry bar-close valuation timestamps, and the backtest curve records session/terminal liquidation at liquidation value. Entry-bar extrema cannot trigger exits before a bar-close fill; tick scans start no earlier than entry; new fills on final NY bars are suppressed. OHLC exits expose interval bounds and unknown exact fill time. Protective fills use price-unit spread and adverse slippage consistently. Nonzero `fill_latency_bars` now creates a pending entry, prevents early position/equity impact, and stamps decision, submission and scheduled fill times; blocked/terminal pending orders are cancelled. Rollout preserves `equity_times` as the Series index and CSV export writes that index. Focused timing/cost and latency regressions pass. Complete cross-engine parity remains under WP12. |
| WP06 | PASS | Completed-position PnL aggregation is shared by `calculate_metrics` and `run_episode`. It sums net fill PnL once for completed positions, excludes incomplete positions, preserves legacy ungrouped trade rows, and reports fill/incomplete counts separately. The original failing regression and a net-PnL/commission regression now pass; metric, runner, confidence-interval and deflated-Sharpe selections pass 41 tests. The full non-slow suite passes after this correction. |
| WP07 | PASS — production fold feature path verified | Fold normalization, feature builder, cache scope, HTF alignment and causality regressions pass. In this continuation the production fold builder ran on the complete configured training calendar and real primary/secondary M1 inputs: fold 0 returned 281,160 train rows, 14,062 validation rows and 312 aligned feature columns; its temporary fold cache was 466,789,239 bytes. The `train_rl` caller was traced to this builder before `model.learn`. This verifies preprocessing assembly and train-only fitting without running policy training. |
| WP08 | PASS — Idea 2 gate and inactive-run reporting verified | Distribution readiness requires the post-inversion latch, matching distribution direction and matching HTF bias. The inversion bar is ineligible and no later retest is required. Required gate columns fail closed in environment construction; strict eligibility propagates through training/evaluation constructors and no-NY/missing-entry-input checks pass. A real 2,880-row primary/secondary sample built 312 features, included all 12 required strategy fields, and constructed a strict environment with 782 NY bars. Zero completed trades now serialize and display `INACTIVE`. A learned policy funnel was not run. |
| WP09 | PASS — causal reference and approved mode contract | Causal target confirmation, front-running buffers, tick precision, de-duplication, one/two/three-slot preservation and reference provenance are covered. Strategy opens export candidate reference identity, timeframe, original/adjusted prices, confirmation time, selection reason and RR-window fallback reason. Author approval in this session resolves mode semantics: `fixed` is immutable; legacy `breakeven` makes a one-time nearest-target pullback to `entry ± alpha × initial risk` without moving SL; `trailing` extends only the nearest remaining unhit slot to a confirmed directional swing. Updates are logged and effective next bar; already-hit slots and their quantities remain stable; crossed levels and slot-overtaking updates are rejected. Regression evidence is in the 191-test scoped run. |
| WP10 | PASS — matched-comparison and reward-attribution implementation | The comparison contract matches starting capital, model settings, dataset, split, costs, execution, budget, seed and timestamps; exactly one explicitly declared feature/reward/environment dimension may differ. Timestamp alignment and shared contiguous-block resampling are enforced. Evaluation rollouts aggregate reported reward parts and explicitly expose unattributed and unreported amounts so they reconcile to the reward scalar. Matched-comparison, reward-attribution, dashboard and report-path regressions pass. No provenance-complete treatment/control policy pair exists in the checkout; no treatment effect or thesis result is claimed. |
| WP11 | PASS — fail-closed per-trial provenance path | Manifests require exact train/test bars and feature SHA-256 identities, split boundaries, an existing feature cache and checkpoint, and checkpoint/training-log/trade-log paths. Dirty untracked directories are expanded to hashed file records. The training caller writes config/source/runtime/schema/cost/execution/metric identity per saved seed checkpoint and fails if Git identity is unavailable. Helper and constructor/call-path evidence passes. No `run_manifest.json` exists under `outputs/`, so prior runs remain uncertified; no provenance is backfilled. Previously inspected OOS data remain historical, not pristine holdout. |
| WP12 | PASS — bounded deterministic parity contract | The prior safe non-slow selection passed 1,280 tests (48 deselected, 19 warnings). Both cross-validation engines consume the same generated SMA crossover actions with matched zero-cost close fills, completed-position counts and drawdown conventions. Real custom-engine/Backtrader long and short action traces reconcile completed-trade count and realized PnL. Environment tick/bar and broker fill traces are covered in the same scoped suite. The combined WP09–WP12 verification passes 191 tests (1 warning). This establishes the bounded deterministic cases only; exhaustive production TradingEnv/broker/evaluation/backtest parity and live execution parity are not claimed. |
| WP13 | PASS — source-truth handoff updated | This ledger, checkpoint inventory, evaluation/training domain maps, reproducibility baseline, distributional methodology, eval pack, validation protocol and threats-to-validity record state implementation evidence, completed-position metrics, inactive-run labeling, matched criteria, strict eligibility, historical OOS exposure, the approved TP-mode contract and exact limits. No thesis results were invented and the authoritative plan was not changed. |

## Finding reconciliation

| Finding | Current disposition |
|---|---|
| F01 | Fixed in the current implementation: swing availability is tested through the confirmation bar; an accepted trailing update records its reference and next-bar effective time. Mirrored, invalid-reference, initial-risk, partial-position and gap-stop regressions pass. |
| F02 | Invalid/duplicate active target slots are disabled rather than assigned fabricated prices; bounded regressions pass. Multi-mode/slot lifecycle is not fully proven. |
| F03–F04 | Tick-slot replay and fill/reward reconciliation regressions pass for bounded traces. Full skipped-bar and production reward reconciliation remain open. |
| F05 | Fixed: the DataFrame adapter and episode runner now share completed-position aggregation; realized fill PnL and completed-position trade statistics are kept separate. |
| F06 | Flag-derived action schema, explicit legacy layout selection, checkpoint validation, training/dashboard/export/reanalysis schema metadata and SB3 round trips pass. All 301 archived output checkpoints are inventoried and hashed; 16 run-family representatives load, including the named 4D migration. Forty-two checkpoints lack saved run configuration and cannot be reanalysed safely until it is recovered. |
| F07 | EMA exit uses executable bid/ask quotes. This continuation also fixed entry-bar lookback exits, final-NY-bar entries, spread-unit mismatch, omitted backtest stop/target slippage, and equity timestamps that predated later fills/liquidations. Tick fills have quote timestamps; OHLC fills identify the triggering bar interval without claiming an exact intrabar instant. Delayed entry orders remain pending until their scheduled NY bar and are cancelled if execution becomes blocked. |
| F08–F09 | Rising-equity bootstrap MDD and selected metric/CI regressions pass. The earlier completed-position adapter failure is fixed. Production-wide frequency protocol and paired-trial evidence remain unverified. |
| F10–F11 | Frozen-fold normalization and fold builder pass helper/cache/causality tests; the full configured training calendar was built into a real fold with 312 aligned features. The policy training/evaluation phase was not run. |
| F12 | Strict research eligibility and production Idea 2 gate inputs are verified on a bounded real-data feature build; zero completed positions are reported as `INACTIVE`. Default legacy permissiveness remains explicit. |
| F13 | Resolved with author approval on 2026-10-13. The legacy TP mode meanings are explicitly documented and regression-tested; no stop movement is implied by the `breakeven` TP mode. |
| F14 | Matched-comparison, rollout reward attribution, and historical-reanalysis provenance helpers are integrated and tested. No matched policy pair or pristine holdout claim is established. |
| F15 | Corrected model/live/features/evaluation/training maps and the relevant reproducibility, distributional, eval-pack and validity records. Remaining source-truth review concerns docs outside these implementation claims. |

Continuation findings (2026-10-13): the WP10 comparison helper previously
omitted initial capital, model/environment configuration, and the declared
evaluation window from its matching key. Those mismatches now fail closed and
the return timestamps are checked against the declared endpoints. The WP11
manifest helper previously accepted a partial dataset list and absent saved
artifacts; it now requires exact train/test bars and feature identities plus
the cache, checkpoint, and output-path contract. Target references also carry
the RR-window fallback reason, and metric exports label zero-trade runs
`INACTIVE`. A further WP10 defect was that the helper could only compare
identical configurations; it now permits one explicitly declared feature,
reward, or environment intervention while requiring every control dimension to
match. A new two-direction deterministic trace compares custom-engine and
Backtrader fills on the same precomputed actions. The prior cross-validation
CLI's custom engine always returned hold, so its comparison was invalid; the
CLI now shares a generated SMA crossover action stream and bounded zero-cost
close-fill protocol across engines. An artifact audit found no saved per-run
manifests in `outputs/` and no matched policy-pair manifest.

## Validation log

Execution environment for all commands below: x86_64 Linux, Python 3.12.14,
`.venv-x86`, repository root `/scratch/work/nguyenl37/Aalto_MS_Thesis`.
Every pytest command below was run with the exact `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest` prefix.

| Command / selection | Result | Limitation |
|---|---|---|
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_evaluation/test_metric_input_contract_unittest.py tests/test_evaluation/test_metrics.py tests/test_evaluation/test_runner.py -q --disable-warnings --maxfail=5` | **25 passed, 1 failed**. Failing assertion: expected `n_trades == 1`, actual `3`; confirms WP06. | Direct DataFrame-adapter regression. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_envs/test_tick_fill_reconciliation.py tests/test_wp02_accounting.py tests/test_envs/test_tp_mode.py tests/test_backtest/test_broker_fill_accounting_unittest.py tests/test_backtest/test_broker_quote_parity_unittest.py -q --disable-warnings --maxfail=8` | **53 passed**. | Bounded deterministic traces; not full production parity. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_action_schema_contract.py tests/test_models/test_action_schema_checkpoint.py tests/test_models/test_action_schema_contract_unittest.py tests/test_models/test_action_schema_sb3_roundtrip.py tests/test_wp_correctness.py tests/test_envs/test_trading_env_golden_traces.py tests/test_backtest/test_target_reference_contract_unittest.py -q --disable-warnings --maxfail=8` | **34 passed, 1 warning**. | Isolated schema/method/trace checks; not production run parity. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_features/test_fold_normalize.py tests/test_features/test_train_walkforward_contract.py tests/test_features/test_feature_cache_and_fit_scope.py tests/test_envs/test_eligibility_wiring.py tests/test_envs/test_entry_rejection_counters.py tests/test_eval/test_eval_run_provenance.py tests/test_eval/test_reward_attribution_unittest.py tests/test_evaluation/test_matched_comparison_unittest.py -q --disable-warnings --maxfail=8` | **46 passed**. | Helpers and selected wiring tests; no campaign training or matched trials. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_evaluation/test_metrics.py tests/test_evaluation/test_calibration_ci.py tests/test_evaluation/test_runner.py tests/test_evaluation/test_deflated_sharpe.py tests/test_evaluation/test_metric_input_contract_unittest.py -k 'not completed_position_stats_include_partial_fills_once' -q --disable-warnings --maxfail=8` | **38 passed, 1 deselected**. | The known failing WP06 regression was excluded only from this neighboring metric/CI run; it was run separately and failed above. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_features/test_build_pipeline.py tests/test_features/test_htf_alignment.py tests/test_causal_features.py -q --disable-warnings --maxfail=5` | **21 passed, 2 warnings**. | Does not execute the training pipeline. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_backtest/test_engine_cross_validation.py -q --disable-warnings --maxfail=5` | **7 passed**. | Backtrader bounded cross-validation tests only. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_envs/test_ny_steps_eod.py tests/test_envs/test_entry_state_machine.py tests/test_envs/test_trader_actions.py tests/test_envs/test_distribution_reward.py tests/test_envs/test_strategy_env.py -q --disable-warnings --maxfail=8` | **75 passed**. | Deterministic environment fixtures; not a production assembled-data campaign. Old test actions were updated to match declared schema coordinates before this passing run. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_envs/test_bundle_env_equivalence.py tests/test_envs/test_bundle_worker.py -q --disable-warnings --maxfail=5` | **14 passed**. | Confirms frame/bundle behavior for bounded fixtures, not complete engine parity. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_eval/test_initial_balance_contract_unittest.py tests/test_eval/test_evaluate_metrics_contract.py tests/test_campaign_artifacts.py tests/test_train/test_variant_resolver.py tests/test_plots_export.py -q --disable-warnings --maxfail=5` | **49 passed**. | Artifact, config, export and metric contracts; no model training or real campaign manifest. |
| `.venv-x86/bin/ruff check quant_rl/action_schema.py quant_rl/envs/trading_env.py quant_rl/evaluation/metrics.py quant_rl/evaluation/runner.py quant_rl/evaluation/bootstrap_ci.py quant_rl/features/build.py quant_rl/features/normalize.py quant_rl/models/agent.py quant_rl/models/ppo_policy.py quant_rl/train/train_rl.py quant_rl/backtest/broker.py quant_rl/backtest/entry_levels.py tests/test_wp_correctness.py tests/test_wp02_accounting.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_evaluation/test_metric_input_contract_unittest.py` | **Passed**: all checks passed. | Scoped to selected paths, not full tree. |
| `.venv-x86/bin/ruff format --check quant_rl/action_schema.py quant_rl/envs/trading_env.py quant_rl/evaluation/metrics.py quant_rl/evaluation/runner.py quant_rl/evaluation/bootstrap_ci.py quant_rl/features/build.py quant_rl/features/normalize.py quant_rl/models/agent.py quant_rl/models/ppo_policy.py quant_rl/train/train_rl.py quant_rl/backtest/broker.py quant_rl/backtest/entry_levels.py tests/test_wp_correctness.py tests/test_wp02_accounting.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_evaluation/test_metric_input_contract_unittest.py` | **Failed check**: six files would be reformatted (`broker.py`, `entry_levels.py`, `trading_env.py`, `features/build.py`, `train/train_rl.py`, and the metric contract test); ten were already formatted. | No formatting was applied because those files were already dirty and a prior platform refusal covered certain formatter operations. Full-tree format/mypy gates were not run; no commit/push was requested. |
| `.venv-x86/bin/ruff check tests/test_envs/test_trader_actions.py tests/test_envs/test_bundle_env_equivalence.py` | **Passed**. | Scoped lint only. |
| `git -c core.whitespace=cr-at-eol diff --check` | **Passed**. | Preserves the existing CRLF convention in `test_trader_actions.py` while checking whitespace. |
| `.venv-x86/bin/mypy quant_rl/models/agent.py` | **Passed** after switching dynamic schema metadata writes to `setattr`. | Scoped to the changed PPO builder; not a full-tree mypy run. |
| `.venv-x86/bin/ruff check quant_rl/models/agent.py tests/test_envs/test_fixed_risk_sizing.py tests/test_envs/test_risk_alignment.py tests/test_envs/test_trading_env_decomposition.py tests/test_rollout.py tests/test_orchestra/test_parse.py` | **Passed**. | Scoped lint only. |
| `.venv-x86/bin/ruff format --check quant_rl/models/agent.py tests/test_envs/test_fixed_risk_sizing.py tests/test_envs/test_risk_alignment.py tests/test_envs/test_trading_env_decomposition.py tests/test_rollout.py tests/test_orchestra/test_parse.py` | **Failed check**: Ruff says the existing dirty `test_risk_alignment.py` and `test_trading_env_decomposition.py` need formatting; four others are formatted. | No formatter was applied because earlier explicit platform refusal covered certain Ruff formatting operations on existing files. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_evaluation/test_metric_input_contract_unittest.py -q` | **2 passed, 1 failed** at the earlier checkpoint. `test_completed_position_stats_include_partial_fills_once` expected one completed position but received 3 fills. | Direct defect reproduction; production caller path was verified in `quant_rl/eval/rollout.py` and `quant_rl/eval/eval_run.py`. The earlier platform refusal was followed by the author's explicit authorization, after which the correction was applied. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_envs/test_fixed_risk_sizing.py tests/test_envs/test_risk_alignment.py tests/test_envs/test_trading_env_decomposition.py -q --disable-warnings` | **51 passed** after aligning fixtures to v2 coordinates and selecting structural exits. | Deterministic environment fixtures. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_models/test_ppo_log_std.py -q --disable-warnings` | **15 passed** after policy construction was changed to use the environment's declared schema. | CPU policy construction and prior tests; no model learning. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_models/test_ppo_log_std.py tests/test_models/test_action_schema_checkpoint.py tests/test_models/test_action_schema_contract_unittest.py tests/test_models/test_action_schema_sb3_roundtrip.py -q --disable-warnings` | **22 passed**. | Rechecks policy construction and checkpoint schema behavior on CPU; no model learning. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_rollout.py tests/test_orchestra/test_parse.py tests/test_models/test_ppo_log_std.py -q --disable-warnings` | **34 passed, 2 warnings**. | Rollout regression now records both sessions' deliberate tiny-loss breaches; date parser fixture is relative to current date. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -m 'not slow' -k 'not test_callback_co_trains_with_ppo_on_trading_env' -q --disable-warnings` | **1,255 passed, 1 failed, 28 deselected, 20 warnings**. The only failure is `test_completed_position_stats_include_partial_fills_once` (`n_trades=3`, expected `1`). | Full non-slow suite, excluding marked slow tests and the non-slow auxiliary test that calls `model.learn()`. Confirms remaining WP06 regression. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -m 'not slow' -k 'not test_callback_co_trains_with_ppo_on_trading_env' -q --disable-warnings` | **1,257 passed, 28 deselected, 20 warnings** after the position aggregation correction. | Full non-slow suite; excludes marked slow tests and the non-slow auxiliary test that calls `model.learn()`. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_evaluation/test_metric_input_contract_unittest.py tests/test_evaluation/test_runner.py tests/test_evaluation/test_metrics.py tests/test_evaluation/test_calibration_ci.py tests/test_evaluation/test_deflated_sharpe.py -q --disable-warnings` | **41 passed** after the correction, including the initial completed-position regression, net PnL/commission accounting, and runner-level fill-count coverage. | Focused WP06 and neighboring statistical metric suites. |
| `.venv-x86/bin/ruff check quant_rl/evaluation/metrics.py quant_rl/evaluation/runner.py tests/test_evaluation/test_metric_input_contract_unittest.py tests/test_evaluation/test_runner.py` | **Passed**. | Scoped lint only. |
| `.venv-x86/bin/mypy quant_rl/evaluation/metrics.py quant_rl/evaluation/runner.py` | **Passed**. | Scoped type check only. |
| `.venv-x86/bin/ruff format --check quant_rl/evaluation/metrics.py quant_rl/evaluation/runner.py tests/test_evaluation/test_metric_input_contract_unittest.py tests/test_evaluation/test_runner.py` | **Failed check**: `metrics.py` and the metric input contract test would be reformatted; `runner.py` and the runner test are formatted. | Formatter was not run; the earlier platform restriction concerned Ruff formatting operations on existing files. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_backtest/test_engine_execution_contract_unittest.py -q --disable-warnings --maxfail=1` | **1 failed** before the fix: structural SL produced no `stop_close` when `max_loss_per_trade_usd=None`. | RED reproduction of the backtest stop-gating defect. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_backtest/test_engine_execution_contract_unittest.py::test_structural_stop_executes_without_optional_max_loss_cap -q --disable-warnings --maxfail=1` | **1 failed** before adding backtest fill identity stamping: exit row had no `position_id`. | RED reproduction of missing backtest position/fill lifecycle metadata. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_backtest/test_engine_execution_contract_unittest.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_wp02_accounting.py tests/test_wp_correctness.py tests/test_envs/test_tp_mode.py tests/test_backtest/test_broker_fill_accounting_unittest.py tests/test_backtest/test_broker_quote_parity_unittest.py tests/test_envs/test_ny_steps_eod.py tests/test_macd_baseline.py -q --disable-warnings --maxfail=8` | **101 passed** after the fixes. | CPU deterministic regressions for structural stops without the safety cap, commission-aware completed backtest fills, delayed tick timestamps, terminal-bar entry prevention, skipped-bar stop replay, ambiguous OHLC stop-first behavior, multi-slot environment fills, and neighboring contracts. No training. |
| `.venv-x86/bin/ruff check quant_rl/data/ticks.py quant_rl/envs/trading_env.py quant_rl/backtest/engine.py tests/test_envs/test_tp_mode.py tests/test_backtest/test_engine_execution_contract_unittest.py` | **Passed**. | Scoped lint. |
| `.venv-x86/bin/mypy quant_rl/data/ticks.py quant_rl/envs/trading_env.py quant_rl/backtest/engine.py` | **Passed** after using the existing `Any` cast convention for the stored entry timestamp. | Scoped type check, not full-tree Mypy. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_backtest/test_engine_execution_contract_unittest.py::test_structural_stop_executes_without_optional_max_loss_cap -q` | **1 failed before the timing fix**: the OHLC stop interval began at 14:02 UTC while the entry filled at 14:03 UTC. | RED reproduction of an exit recorded before its position existed. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_backtest/test_engine_execution_contract_unittest.py tests/test_envs/test_tp_mode.py tests/test_envs/test_entry_state_machine.py tests/test_envs/test_ny_steps_eod.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_envs/test_entry_rejection_counters.py tests/test_live/test_rl_bridge.py -q --disable-warnings --maxfail=5` | **111 passed, 1 warning** after WP05 changes. | Covers entry-bar causality, NY/data boundary entry suppression, delayed tick floors, price-unit spread, adverse stop slippage, equity/fill time alignment, session close, EMA bid/ask, and latency environment compatibility. No model learning. |
| `.venv-x86/bin/ruff check quant_rl/data/ticks.py quant_rl/envs/trading_env.py quant_rl/backtest/engine.py tests/test_envs/test_tp_mode.py tests/test_envs/test_entry_state_machine.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_envs/test_entry_rejection_counters.py tests/test_backtest/test_engine_execution_contract_unittest.py` | **Passed**. | Scoped lint after the timing/cost changes. |
| `.venv-x86/bin/mypy quant_rl/data/ticks.py quant_rl/envs/trading_env.py quant_rl/backtest/engine.py` | **Passed**. | Scoped type check after the timing/cost changes. |
| `.venv-x86/bin/ruff check quant_rl/action_schema.py quant_rl/models/agent.py quant_rl/eval/export.py quant_rl/eval/eval_run.py quant_rl/train/dashboard.py quant_rl/train/train_rl.py tests/test_models/test_action_schema_checkpoint.py tests/test_train/test_dashboard.py tests/test_train/test_train_rl_decomposition.py tests/test_plots_export.py` | **Passed**. | Scoped action-schema consumer lint. |
| `.venv-x86/bin/mypy quant_rl/action_schema.py quant_rl/models/agent.py quant_rl/eval/export.py quant_rl/eval/eval_run.py quant_rl/train/dashboard.py quant_rl/train/train_rl.py` | **Passed** after annotating the walk-forward summary rows. | Scoped type check for action-schema consumers; not full-tree Mypy. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_action_schema_contract.py tests/test_models/test_action_schema_checkpoint.py tests/test_models/test_action_schema_contract_unittest.py tests/test_models/test_action_schema_sb3_roundtrip.py tests/test_train/test_train_rl_decomposition.py tests/test_train/test_dashboard.py tests/test_plots_export.py::test_export_records_declared_action_coordinate_schema tests/test_eval/test_eval_run_provenance.py -q --disable-warnings --maxfail=5` | **51 passed**. | Verifies action coordinates and explicit legacy/versioned schema metadata through policy/checkpoint, training logs, dashboard CSV, report export and reanalysis provenance. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -m 'not slow' -k 'not test_callback_co_trains_with_ppo_on_trading_env and not test_integration' -q --disable-warnings` | **1,263 passed, 1 failed, 47 deselected, 18 warnings in 113.39s**. The failure is `test_codex_yaml_extra_args_match_codex_exec_cli`: local `codex exec --approve-for-me --ephemeral` exceeded the test's 5-second subprocess timeout. | Safe non-slow selection; excluded training callback and integration tests. The lone failure is unrelated to execution code but remains a real failed test result. |
| `git -c core.whitespace=cr-at-eol diff --check` | **Passed**. | Preserves pre-existing CRLF handling in dirty files. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_backtest/test_engine_execution_contract_unittest.py tests/test_envs/test_entry_state_machine.py tests/test_envs/test_tp_mode.py tests/test_envs/test_ny_steps_eod.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_envs/test_trailing_stop_causality.py -q --disable-warnings --maxfail=5` | **90 passed** before the final session-partial regression was added; a separate follow-up run of that regression plus the engine contract passed **6 passed**. | Deterministic WP02–WP05 execution and lifecycle checks. |
| `.venv-x86/bin/ruff check quant_rl/data/ticks.py quant_rl/envs/trading_env.py quant_rl/backtest/engine.py tests/test_backtest/test_engine_execution_contract_unittest.py tests/test_envs/test_entry_state_machine.py tests/test_envs/test_tp_mode.py tests/test_envs/test_trailing_stop_causality.py` | **Passed**. | Scoped lint. |
| `.venv-x86/bin/mypy quant_rl/data/ticks.py quant_rl/envs/trading_env.py quant_rl/backtest/engine.py` | **Passed** after changing bar-duration inference to typed NumPy timestamps. | Scoped type check; not full-tree Mypy. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_eval/test_run_manifest.py tests/test_eval/test_eval_run_provenance.py tests/test_campaign_artifacts.py tests/test_envs/test_distribution_reward.py tests/test_envs/test_eligibility_wiring.py tests/test_features/test_po3_state.py tests/test_features/test_fold_normalize.py tests/test_features/test_train_walkforward_contract.py tests/test_features/test_feature_cache_and_fit_scope.py tests/test_evaluation/test_matched_comparison_unittest.py tests/test_eval/test_reward_attribution_unittest.py -q --disable-warnings --maxfail=5` | **73 passed**. | Provenance helper and split tests, eligibility gate and Idea 2 causal IFVG/distribution state, fold/cache contracts, matched-comparison and reward-attribution contracts. Does not launch training or actual matched trials. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -m 'not slow' -k 'not test_callback_co_trains_with_ppo_on_trading_env and not test_integration and not test_codex_yaml_extra_args_match_codex_exec_cli' -q --disable-warnings` | **1,272 passed, 48 deselected, 18 warnings in 107.64s**. | Safe non-slow regression selection. Excludes slow-marked, explicit PPO co-training/integration tests and the local CLI test listed next. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_orchestra/test_sessions.py::test_codex_yaml_extra_args_match_codex_exec_cli -q --disable-warnings` | **1 passed in 0.21s**. | Retried the previously timed-out test independently; it passed. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_rollout.py tests/test_live/test_rl_bridge.py tests/test_envs/test_entry_state_machine.py tests/test_envs/test_tp_mode.py -q --disable-warnings --maxfail=3` | **69 passed, 3 warnings**. | Rechecked delayed entry lifecycle, timestamps and prior decision-bar report convention against rollout and environment regressions. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_live/test_rl_bridge.py tests/test_rollout.py tests/test_envs/test_entry_state_machine.py tests/test_envs/test_tp_mode.py tests/test_backtest/test_engine_execution_contract_unittest.py -q --disable-warnings --maxfail=5` | **78 passed, 4 warnings** after adding cancellation on episode end and soft-risk block before delayed fill. | Bounded WP02/WP04/WP05 rollout and execution group. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -m 'not slow' -k 'not test_callback_co_trains_with_ppo_on_trading_env and not test_integration and not test_codex_yaml_extra_args_match_codex_exec_cli' -q --disable-warnings` | **1,272 passed, 48 deselected, 18 warnings in 109.94s** after the delayed-order/report-field correction. | Safe non-slow selection, excluding explicit PPO co-training/integration tests and the flaky local CLI subprocess. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -m 'not slow' -k 'not test_callback_co_trains_with_ppo_on_trading_env and not test_integration' -q --disable-warnings` | **1,271 passed, 2 failed, 47 deselected, 18 warnings in 118.15s** at the preceding checkpoint. The failures were the local Codex CLI 5-second timeout and a report `bar` semantic regression; the latter was fixed and the full safe selection above now passes. | The CLI test also passed in a standalone run (0.21s) but timed out in the broad run. The broad result is retained; no blanket full-suite pass is claimed. |
| `.venv-x86/bin/ruff check quant_rl/eval/run_manifest.py quant_rl/train/train_rl.py quant_rl/envs/strategies/distribution.py tests/test_eval/test_run_manifest.py tests/test_envs/test_distribution_reward.py tests/test_envs/test_eligibility_wiring.py` | **Passed**. | Scoped lint after WP08/WP11 continuation. |
| `.venv-x86/bin/mypy quant_rl/eval/run_manifest.py quant_rl/train/train_rl.py quant_rl/envs/strategies/distribution.py` | **Passed**. | Scoped type check; not full-tree Mypy. |
| `git -c core.whitespace=cr-at-eol diff --check` | **Passed**. | Includes current changes without normalizing existing CRLF files. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -m 'not slow' -k 'not test_callback_co_trains_with_ppo_on_trading_env and not test_integration' -q --disable-warnings` | **1,258 passed, 47 deselected, 18 warnings in 108.62s** before the WP05 clock/cost continuation. | Excludes slow-marked tests, the explicit PPO co-training callback test, and integration-named tests. No campaign training. |
| `.venv-x86/bin/python` read-only SB3 checkpoint inspection for `outputs/gpu_campaigns/pilot_18_seed42_100k/.../20261011_035441_430571_rl_train_seed42_tcn/model/ppo_final.zip`, then `validate_model_action_schema(..., allow_legacy=True)` using the adjacent `config.yaml` | **Passed** on CPU; saved action width `(6,)` matched declared `6_dir` schema. The archive has no embedded action-schema version marker. | One actual archive only; no checkpoint was modified and no inference/training was run. |
| `.venv-x86/bin/python` production `build_walkforward_fold_features` invocation on tracked M1 sample with temporary cache | **Passed**: 2,880 input rows, 2,200 train, 680 validation, 189 feature columns; cache and hash sidecar created under a temporary directory. | Two-day sample; does not certify full-data training assembly or campaign run. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -q --disable-warnings` | **Interrupted** after 762 passed and 25 failures; `tests/test_integration/test_scripts_smoke.py` is documented to launch tiny-budget LSTM/RL training. | The broad command inadvertently reached the training-smoke module. It was interrupted after discovery; it is not a full-suite result. No thesis campaign was started. |

### Continuation validation (after WP07–WP13 follow-up)

| Command / exercise | Result | Limitation |
|---|---|---|
| Focused WP07–WP11 selection: fold normalization/walk-forward/cache tests, Idea 2 eligibility/reward, target-reference tests, matched comparison, manifest and reanalysis provenance | **70 passed** in 3.91s. | Contract/helper and production-constructor coverage; no learned policy. |
| Safe non-slow selection: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests -m 'not slow' -k 'not test_callback_co_trains_with_ppo_on_trading_env and not test_integration and not test_codex_yaml_extra_args_match_codex_exec_cli' -q --disable-warnings` | **1,277 passed, 48 deselected, 19 warnings** in 123.34s. | No training smoke, named integration tests, or flaky CLI subprocess. This is not the full suite. |
| Production fold feature build using `cache/US100.cash_M1_v2-full-day.parquet`, matching US500 M1 data, `config/3_distribution.yaml`, `purged_walk_forward`, and `build_walkforward_fold_features` | **Passed**: fold 0 produced 281,160 training rows, 14,062 validation rows, 312 aligned columns and a temporary 466,789,239-byte cache. | CPU preprocessing only. Policy training and evaluation were not run. |
| Production Idea 2 feature build on the first 2,880 tracked M1 bars with matching US500 bars; strict environment construction | **Passed**: 312 columns, all 12 required gate fields present, 782 NY bars, strict environment constructed. | Bounded sample; no policy actions or outcome claims. |
| `.venv-x86/bin/ruff check` and `.venv-x86/bin/mypy` on changed target-reference, matched-comparison and manifest modules | **Passed** after fixing one test import-order finding. | Scoped checks; no full-tree static check. |
| Final safe non-slow selection after WP11 dirty-source hashing change | **1,280 passed, 48 deselected, 19 warnings** in 134.79s. | Exact selection excludes slow-marked tests, explicit PPO co-training/integration tests, and one flaky CLI subprocess. No training campaign. |
| `.venv-x86/bin/ruff check` for changed WP07–WP11 modules/tests; `.venv-x86/bin/mypy` for six changed source modules; `git -c core.whitespace=cr-at-eol diff --check` | **All passed.** | Scoped checks only. |
| `tests/test_eval/test_run_manifest.py` after dirty-directory hashing regression | **6 passed** in 27.70s. A read-only check against the actual dirty checkout found one untracked campaign file and confirmed it was SHA-256 recorded. | No run manifest or training artifact was created in the repository. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_evaluation/test_report.py tests/test_evaluation/test_matched_comparison_unittest.py tests/test_eval/test_run_manifest.py tests/test_backtest/test_target_reference_contract_unittest.py tests/test_envs/test_entry_state_machine.py -q --disable-warnings --maxfail=3` | **46 passed** in 28.98s. | Focused final reporting/provenance/reference tests. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_evaluation/test_matched_comparison_unittest.py tests/test_backtest/test_target_reference_contract_unittest.py tests/test_envs/test_entry_state_machine.py tests/test_eval/test_run_manifest.py tests/test_backtest/test_engine_execution_contract_unittest.py tests/test_backtest/test_engine_cross_validation.py tests/test_backtest/test_engine_backtrader_signal_parity.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_envs/test_trading_env_golden_traces.py tests/test_envs/test_bundle_env_equivalence.py tests/test_envs/test_bundle_eval.py tests/test_plots_export.py -q --disable-warnings` | **101 passed, 1 warning** in 36.74s. | Bounded deterministic reference, manifest, report/export, environment, broker, backtest, bundle evaluation and custom-engine/Backtrader traces; not full production parity. |
| `.venv-x86/bin/ruff check quant_rl/evaluation/matched_comparison.py tests/test_evaluation/test_matched_comparison_unittest.py tests/test_envs/test_entry_state_machine.py tests/test_backtest/test_engine_backtrader_signal_parity.py`; `.venv-x86/bin/mypy quant_rl/evaluation/matched_comparison.py`; `git -c core.whitespace=cr-at-eol diff --check` | **All passed.** | Scoped static/style and whitespace checks only. |
| Latest WP09–WP12 selection after replacing the cross-validation no-op: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_evaluation/test_matched_comparison_unittest.py tests/test_backtest/test_target_reference_contract_unittest.py tests/test_envs/test_entry_state_machine.py tests/test_eval/test_run_manifest.py tests/test_backtest/test_engine_execution_contract_unittest.py tests/test_backtest/test_engine_cross_validation.py tests/test_backtest/test_engine_backtrader_signal_parity.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_envs/test_trading_env_golden_traces.py tests/test_envs/test_bundle_env_equivalence.py tests/test_envs/test_bundle_eval.py tests/test_plots_export.py -q --disable-warnings` | **102 passed, 1 warning** in 36.25s. | Bounded deterministic behavior only; no model evaluation or training. |
| `.venv-x86/bin/ruff check` for the changed comparison, cross-validation and test modules; `.venv-x86/bin/mypy quant_rl/evaluation/matched_comparison.py quant_rl/backtest/cross_validation/run.py`; `git -c core.whitespace=cr-at-eol diff --check` | **All passed.** | Scoped style, type and whitespace checks only. |
| Read-only `find outputs -type f -name run_manifest.json` and comparison artifact inspection | **0 run manifests found.** The only `comparison.json` inspected compares report-export performance, not policy arms. | Historical outputs are preserved; no provenance was synthesized and no historical OOS results were used as fresh holdout evidence. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_evaluation/test_matched_comparison_unittest.py tests/test_eval/test_reward_attribution_unittest.py tests/test_train/test_dashboard.py tests/test_backtest/test_target_reference_contract_unittest.py tests/test_envs/test_entry_state_machine.py tests/test_eval/test_run_manifest.py tests/test_eval/test_eval_run_provenance.py tests/test_backtest/test_engine_execution_contract_unittest.py tests/test_backtest/test_engine_cross_validation.py tests/test_backtest/test_engine_backtrader_signal_parity.py tests/test_envs/test_tick_fill_reconciliation.py tests/test_envs/test_tp_mode.py tests/test_envs/test_trailing_stop_causality.py tests/test_envs/test_trading_env_golden_traces.py tests/test_envs/test_bundle_env_equivalence.py tests/test_envs/test_bundle_eval.py tests/test_plots_export.py -q` | **191 passed, 1 warning** in 37.72s. | Warning: golden trace fixture lacks London/Asian levels and `volume_spike`; legacy entry gate falls back to allowing entries. Bounded deterministic checks, no learned policy or campaign. |
| Scoped `.venv-x86/bin/ruff check` on the touched WP09–WP12 source/test modules | **Passed.** | Scoped lint only. |
| Scoped `.venv-x86/bin/mypy quant_rl/envs/trading_env.py quant_rl/evaluation/matched_comparison.py quant_rl/eval/reward_attribution.py quant_rl/eval/run_manifest.py quant_rl/backtest/cross_validation/run.py` | **Passed: no issues in 5 source files.** | Scoped typing only. |
| Scoped `ruff format --check` on 11 relevant source/test files | **Failed: seven files would be reformatted** (`trading_env.py`, `run_manifest.py`, `matched_comparison.py`, cross-validation, TP-mode, run-manifest and matched-comparison tests). | Check only; no formatting write was applied to avoid rewriting existing dirty files. |
| `git -c core.whitespace=cr-at-eol diff --check` | **Passed.** | Whitespace validation only. |
| Read-only SB3 archive inventory: 301 `.zip` files under `outputs/`, 6,144,691,658 bytes; parse each `data` record, record width/schema markers/config association, and SHA-256 every archive in `docs/thesis/checkpoint_schema_inventory_2026-10-13.json` | **301 parsed, 0 archive errors; all unversioned; 259 have a saved run config, 42 do not; widths: 4D=23, 5D=138, 6D=96, 7D=23, 11D=21.** | Archive metadata and hashes only; no historical metrics changed and no output archive modified. |
| CPU representative checkpoint loadability audit, one representative per each of 16 run directories | **16/16 loadable**, with explicit migration for the 4D legacy family. The legacy `20260921.../ppo_best.zip` failed ordinary loading due its pre-account-embedding TCN and 134-wide policy input; the compatibility loader restored the archived TCN/policy and structural-exit adapter inference returned 5D. | Representative files only; all 301 were not individually instantiated. Synthetic zero observation only; no OOS rollout. |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_action_schema_contract.py tests/test_models/test_action_schema_checkpoint.py tests/test_models/test_action_schema_contract_unittest.py tests/test_models/test_action_schema_sb3_roundtrip.py tests/test_models/test_legacy_checkpoint_migration.py tests/test_models/test_encoders.py tests/test_models/test_ppo_log_std.py tests/test_eval/test_eval_run_provenance.py -q` | **64 passed, 8 warnings** in 7.84s. | Warnings are PyTorch Transformer nested-tensor notices. Includes migration test; does not load every saved archive. |
| `.venv-x86/bin/ruff check` on action schema, legacy policy/encoder, eval loader, and tests; `.venv-x86/bin/mypy quant_rl/models/agent.py quant_rl/models/encoder.py quant_rl/models/ppo_policy.py quant_rl/eval/eval_run.py`; `git -c core.whitespace=cr-at-eol diff --check` | **All passed.** | Scoped style, typing and whitespace checks. |

Counts across groups overlap; do not add them as a count of unique tests.
No thesis training campaign, Triton allocation/cancellation, commit, push, or
historical artifact modification occurred. One unfiltered full-suite attempt
reached the module documented to run tiny-budget LSTM/RL training before it
was recognized. The progress output suggests the LSTM smoke completed and the
PPO smoke began; it was interrupted during a Torch autograd step, and the
completed step count was not captured. That accidental test invocation is
recorded above. Windows validation and GPU validation were not performed in
this continuation.

## Changed in this continuation

- `.agents/domain_maps/models.md` — account observation width and flag-derived action schema.
- `.agents/domain_maps/live.md` — remove stale fixed 4-D strategy-action description.
- `.agents/domain_maps/features.md` — document frozen-fold and legacy normalization modes.
- `docs/thesis/threats_to_validity.md` — replace stale claim that walk-forward folds reuse IS-scoped features with current implementation status and remaining validation gap.
- `docs/thesis/validation_protocol.md` — explicitly preserve the historical status of any previously inspected OOS period.
- `docs/distributional_methodology.md` — define completed-position observations and label zero-trade runs `INACTIVE`.
- `docs/thesis/eval_pack.md` — clarify that configured 2026 OOS data are historical, not pristine.
- `.agents/domain_maps/evaluation.md` and `.agents/domain_maps/training.md` — document matched-provenance criteria, fold-scoped fit and per-trial manifests.
- `quant_rl/evaluation/matched_comparison.py` and `tests/test_evaluation/test_matched_comparison_unittest.py` — require matched capital/model/environment/evaluation-window evidence, validate window endpoints, and allow only one explicitly declared intervention dimension.
- `quant_rl/backtest/cross_validation/run.py` and `tests/test_backtest/test_engine_cross_validation.py` — replace the no-op custom policy with shared SMA crossover actions and align terminal fill/count conventions.
- `tests/test_backtest/test_engine_backtrader_signal_parity.py` — compare deterministic long/short fills and realized PnL through the custom engine and Backtrader using identical actions.
- `tests/test_envs/test_entry_state_machine.py` — verify target-reference metadata survives JSON serialization.
- `quant_rl/eval/run_manifest.py` and `tests/test_eval/test_run_manifest.py` — require exact split datasets and saved artifacts; hash nested untracked source files.
- `quant_rl/backtest/entry_levels.py` and `tests/test_backtest/test_target_reference_contract_unittest.py` — preserve RR-window fallback reason with target-reference provenance.
- `quant_rl/envs/trading_env.py` and `tests/test_envs/test_entry_state_machine.py` — export target-reference provenance on strategy open events.
- `quant_rl/evaluation/report.py`, `quant_rl/eval/eval_run.py`, and `tests/test_evaluation/test_report.py` — serialize and display `INACTIVE` for zero-completed-position reports.
- `tests/test_envs/test_trader_actions.py` — align fixture action arrays with the declared schema, including configured SL-mode/exit coordinates.
- `tests/test_envs/test_bundle_env_equivalence.py` — generate schema-width continuous actions for strategy-mode bundle parity fixtures.
- `tests/test_envs/test_fixed_risk_sizing.py` — explicitly select the intended schema for the EMA-exit fixture.
- `tests/test_envs/test_risk_alignment.py` — use schema-width vectors and declared structural-exit coordinates.
- `tests/test_envs/test_trading_env_decomposition.py` — construct overlay environments with a declared schema and schema-width vectors.
- `quant_rl/models/agent.py` — bind PPO policy layout to the environment `ActionSchema`, rejecting ambiguous strategy Box spaces.
- `tests/test_rollout.py` — expect the second session's intentional tiny-loss breach at terminal close.
- `tests/test_orchestra/test_parse.py` — use a future-relative quota date instead of an expired literal.
- `quant_rl/evaluation/metrics.py` — aggregate completed positions from fill records; expose separate fill/incomplete diagnostics.
- `quant_rl/evaluation/runner.py` — reuse the same position/fill aggregation as the DataFrame adapter.
- `tests/test_evaluation/test_metric_input_contract_unittest.py` — cover completed-only trade statistics, commission-aware net PnL, fill counts, and incomplete positions.
- `tests/test_evaluation/test_runner.py` — cover shared aggregation in `run_episode`.
- `quant_rl/backtest/engine.py` — execute structural stops independently of the optional USD safety cap, stamp backtest position/fill summaries, log delayed tick exits at execution time, and prevent terminal-bar entries without a future fill bar.
- `quant_rl/data/ticks.py` — expose the selected tick quote timestamp alongside delayed exit events while retaining the legacy quote-only API.
- `quant_rl/envs/trading_env.py` — log delayed single and multi-slot tick exits at their execution timestamps; retain explicit timestamp typing.
- `tests/test_backtest/test_engine_execution_contract_unittest.py` — add RED/GREEN structural-stop and delayed-fill-time engine regressions.
- `tests/test_envs/test_tp_mode.py` — cover stop-first ambiguous OHLC, delayed single-target tick timestamps, and completed stops during skipped-bar replay.
- `tests/test_envs/test_trailing_stop_causality.py` — verify confirmed swing visibility, auditable activation time, immutable initial risk, invalid references, and partial-position gap closure.
- `tests/test_envs/test_entry_state_machine.py` — verify decision and actual fill timestamps are both recorded for an entry.
- `tests/test_backtest/test_engine_execution_contract_unittest.py` — verify entry fill source/time and NY session closure at the permitted bar close rather than an after-hours quote.
- `quant_rl/backtest/engine.py` — timestamp tick fills at observed quote time and bar fallbacks at the selected bar close; use current NY close for session-end liquidation and final-bar close for end-of-data liquidation.
- `quant_rl/backtest/engine.py` — align equity observations to left-labeled bar closes, include session/terminal liquidation values at their close timestamps, suppress entries whose fill would be on the final NY bar, enforce entry-before-exit causality, and apply price-unit spread plus adverse stop/target slippage.
- `quant_rl/data/ticks.py` — allow protective tick scans to begin no earlier than the position's entry time.
- `quant_rl/envs/trading_env.py` — timestamp entry decisions and fills separately, align equity observation times to actual fills/valuations, use fill time for position age and guard-triggered close records, skip entry-bar OHLC exits, suppress entries that fill at the final NY boundary, and use consistent protective spread units.
- `tests/test_backtest/test_engine_execution_contract_unittest.py` — cover pre-entry stop timing, final-NY fill rejection, terminal/session valuation times, and spread/slippage parity.
- `tests/test_envs/test_tp_mode.py` — cover environment entry-bar causality and final-NY fill rejection.
- `tests/test_envs/test_entry_state_machine.py` — assert equity valuation time follows the actual entry fill and that entry-bar extrema are not replayed.
- `quant_rl/envs/trading_env.py` — execute nonzero bar-latency entries as pending orders at the scheduled next eligible NY bar; keep early marks/exits on next-market quotes, record submission/fill times, and cancel orders blocked before fill.
- `tests/test_live/test_rl_bridge.py` — replace the prior immediate-future-price assertion with checks that delayed orders remain flat until scheduled fill, have no future-dated equity point, and retain submission/fill timestamps.
- `tests/test_envs/test_tick_fill_reconciliation.py` — cover earliest allowed tick trigger time.
- `tests/test_envs/test_entry_rejection_counters.py` — assert price-unit protective spread semantics.
- `quant_rl/action_schema.py` — serialize schema version, key, width, coordinate names and flags.
- `quant_rl/models/agent.py` — expose validated schema metadata on constructed/loaded models, marking legacy checkpoints explicitly.
- `quant_rl/train/train_rl.py` — include the declared schema in per-seed and aggregate training logs; make walk-forward summary typing explicit.
- `quant_rl/train/dashboard.py` — persist schema version/key/coordinates in dashboard rows.
- `quant_rl/eval/export.py` — include configured coordinate semantics in export reports.
- `quant_rl/eval/eval_run.py` — include validated checkpoint schema semantics in the historical reanalysis manifest.
- `tests/test_models/test_action_schema_checkpoint.py`, `tests/test_train/test_dashboard.py`, `tests/test_train/test_train_rl_decomposition.py`, and `tests/test_plots_export.py` — cover schema metadata through checkpoint validation, training logs, dashboard rows and export reports.
- `quant_rl/eval/run_manifest.py` — add content fingerprints for config, dataframe splits, feature cache and checkpoints; record Git HEAD plus hashes of dirty source files, split boundaries, schema, execution/metric protocols and runtime/CUDA identity; write manifests atomically and reject missing split or dataset identity.
- `quant_rl/train/train_rl.py` — write resolved root/seed configs; fingerprint the exact train/test bars and features plus tick arrays; emit one provenance manifest per saved seed checkpoint.
- `tests/test_eval/test_run_manifest.py` — verify manifest contents, hashes, atomic writes, split fail-closed behavior and array identity.
- `quant_rl/envs/strategies/distribution.py` — require Idea 2 distribution phase/direction, HTF bias and post-IFVG inputs before enabling the strategy environment.
- `tests/test_envs/test_distribution_reward.py` and `tests/test_envs/test_eligibility_wiring.py` — verify direction/bias alignment, post-confirmation entry without a later retest, and fail-closed environment construction when Idea 2 gate inputs are missing.
- `docs/thesis/implementation_ledger_2026-10-13.md` — this status, evidence and handoff record.

Additional WP07–WP13 continuation files (this final follow-up):

- `.agents/domain_maps/evaluation.md`
- `.agents/domain_maps/training.md`
- `docs/distributional_methodology.md`
- `docs/reproducibility.md`
- `docs/thesis/eval_pack.md`
- `docs/thesis/implementation_ledger_2026-10-13.md`
- `docs/thesis/threats_to_validity.md`
- `docs/thesis/validation_protocol.md`
- `quant_rl/backtest/entry_levels.py`
- `quant_rl/envs/trading_env.py`
- `quant_rl/eval/eval_run.py`
- `quant_rl/eval/run_manifest.py`
- `quant_rl/evaluation/matched_comparison.py`
- `quant_rl/evaluation/report.py`
- `tests/test_backtest/test_target_reference_contract_unittest.py`
- `tests/test_envs/test_entry_state_machine.py`
- `tests/test_eval/test_run_manifest.py`
- `tests/test_evaluation/test_matched_comparison_unittest.py`
- `tests/test_evaluation/test_report.py`
- `docs/thesis/checkpoint_schema_inventory_2026-10-13.json`
- `quant_rl/models/encoder.py`
- `quant_rl/models/ppo_policy.py`
- `quant_rl/eval/eval_run.py`
- `tests/test_models/test_legacy_checkpoint_migration.py`
- `.agents/domain_maps/models.md` — document the narrow explicit 4D legacy
  checkpoint migration contract.

Final WP09–WP12 continuation changes:

- `quant_rl/envs/trading_env.py` — reject trailing target updates that would
  overtake a later active TP slot and discard its allocated quantity.
- `tests/test_envs/test_tp_mode.py` — verify mode invariants, hit-slot
  immutability, directional trailing references, next-bar activation,
  marketability guard and slot ordering.
- `quant_rl/eval/reward_attribution.py` — use the standard `collections.abc`
  mapping import and Ruff import order; behavior is unchanged.
- `docs/distributional_methodology.md` — record the approved versioned TP
  mode/reference contract and lifecycle timing.
- `docs/thesis/implementation_ledger_2026-10-13.md` — close WP09–WP12 status
  with final regression, lint, type, format-check and known parity evidence.

The other dirty and untracked code paths listed by `git status` existed before
this continuation. Their exact pre-session content was not compared with the
Windows v7 candidate because that package was unavailable here.

## Remaining blockers and readiness

- WP02–WP04 now meet their bounded package contracts with regression evidence.
  An additional full `TradingEnv.step()` trace now verifies a TP1 partial fill,
  a later stop on the residual size, stable fill IDs, completed-position PnL,
  commission allocation, and PnL reward attribution. Full production engine
  parity remains under WP12.
- WP05's deterministic clock/cost defects identified in this continuation are
  fixed and bounded tests pass, including a pending-order regression for
  nonzero `fill_latency_bars`. Rollout and CSV exports preserve the timestamped
  equity index. OHLC trigger times are represented as intervals, which is the
  available precision without ticks. Full environment/backtest parity remains
  under WP12.
- WP06's completed-position defect is fixed and its acceptance tests pass.
- WP09–WP12 meet their bounded implementation/test acceptance and are marked
  PASS in the matrix. No treatment/control policy experiment was run, no new
  run manifest was emitted, and exhaustive production or live-path parity was
  not attempted. These are evidence gaps for campaign claims, not failures of
  the helpers or deterministic cases covered here.
- WP01's implementation and archive inventory are complete. Forty-two legacy
  checkpoints have no saved run config, so they remain unavailable for safe
  reanalysis until that original config is recovered; the new eval path fails
  closed rather than inventing it. Learned policy behavior and fresh matched
  outcomes remain untested. Full simulator/live parity is a remaining
  correctness risk even though the bounded WP12 traces pass.
- Dataset, checkpoint and full campaign provenance were not available for
  verification. Previously inspected OOS results cannot be represented as an
  untouched holdout.
- Campaign readiness: **NOT READY**. Do not initiate training from this state.

## Continuation verification — 2026-10-13

After recovering the prior Codex thread, the bounded WP09–WP12 regression
selection was rerun against this checkout: **191 passed, 1 warning** in 75.70s.
The warning is the already documented golden-trace fixture missing entry-gate
columns and using the compatibility fallback.

Added `tests/test_envs/test_tick_fill_reconciliation.py::test_trading_env_step_reconciles_partial_target_then_stop_reward` to exercise
the production environment step path across a partial TP1 and a later stop. It
asserts quantity conservation, exactly-once position/fill identity, net PnL
against account balance after entry and exit commissions, and that each step's
PnL reward equals its newly realized fill PnL.

Validation commands (x86 `.venv-x86`, CPU):

- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_envs/test_tick_fill_reconciliation.py::test_trading_env_step_reconciles_partial_target_then_stop_reward -q` — **1 passed**.
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_envs/test_tick_fill_reconciliation.py tests/test_evaluation/test_metric_input_contract_unittest.py tests/test_evaluation/test_metrics.py tests/test_evaluation/test_runner.py -q` — **34 passed**.
- `.venv-x86/bin/ruff check tests/test_envs/test_tick_fill_reconciliation.py` — passed.
- `.venv-x86/bin/ruff format --check tests/test_envs/test_tick_fill_reconciliation.py` — passed; file already formatted.
- `git diff --check` — reported trailing whitespace in the pre-existing modified `tests/test_envs/test_trader_actions.py`; no edits were made to that file.

This strengthens bounded environment integration evidence. It does not establish
exhaustive environment/backtest/live parity, create historical run provenance,
recover missing checkpoint configurations, or produce a matched policy pair.
Campaign readiness remains **NOT READY**.

## Final handoff check — 2026-10-13

The locally available campaign dry-run was executed with the frozen campaign
spec and produced exactly 18 unique variant/seed keys, seed 42, 20,000,000
steps per run, and 360,000,000 total requested steps. It did not create the
campaign output directory or start a worker.

The two 2026-10-08 A2/A3 run directories still contain 21 checkpoint archives
each but no `config.yaml` or run manifest. Their available debug artifacts do
not fully identify the effective training configuration; the logs also contain
inconsistent strategy labels. Treat all 42 archives as unavailable for
provenance-complete reanalysis unless the original run configs or an equivalent
authoritative record are recovered. Do not reconstruct them from current
configuration defaults.

The current primary and secondary M1 parquet caches each cover
2024-12-30 01:05 through 2026-07-03 19:59 (Etc/GMT-3; 528,760 and 527,139
rows respectively). The configured 2026 test period is therefore present in
the local data and was previously inspected. It remains historical evidence,
not an untouched final holdout. A future thesis claim requiring a pristine
holdout needs a newly sourced and independently frozen period, with its
availability and non-exposure documented before evaluation.

Read-only Triton status/session discovery found no running job and no reusable
GPU tmux session. No allocation or training was started. GPU-side real-data
parity, sustained resource profiling, and end-to-end campaign execution remain
unverified. The exact local dry-run and dataset boundary check do not close
those remote gates.

**Handoff decision: implementation and bounded tests are ready for review;
the thesis campaign is not launch-ready.** To reach a defensible launch state,
recover the two historical run configs (or exclude those checkpoints from
reanalysis), freeze a valid final-holdout source and split, complete the listed
compute-node parity/resource gates, and obtain the project-required explicit
authorization for a new GPU allocation. The subsequent explicit instruction
permits exploratory tuning on the 2026 period, which is permanently exposed
development evidence and cannot serve as a final holdout.

## Continued readiness work — 2026-10-13

The campaign fingerprint previously omitted untracked runtime source modules.
`campaign_source_files()` now hashes both tracked and nonignored untracked
sources under the training/config/controller paths, and worker count plus the
evaluation exposure record are part of the campaign identity. This prevents a
resume from silently accepting a different local implementation. The new
regression test specifically places a runtime module in the untracked state.

The read-only historical OOS audit is recorded in
`docs/thesis/historical_oos_exposure_audit_2026-10-13.json`, with hashes for
each training log and saved config. The meaningful trained evaluations in
those logs breached the account loss limit; the 8,192-step zero-trade run is
not viability evidence. These runs are heterogeneous and do not form a
matched treatment/control comparison, so the audit supports a no-go on current
candidate claims, not a specific risk or architecture adjustment.

Per the explicit instruction to use this period for tuning, the validation
protocol and threats-to-validity record now designate 2026 data as exposed
development evidence. Added
`config/campaigns/dev_a0_a1_seed42_100k.yaml`: an exploratory A0/A1 one-factor
comparison, seed 42, 100,000 steps per variant, with selection explicitly
acknowledged. A fresh post-2026-07-03 final holdout remains required. Its dry-run
resolves to exactly two keys / 200,000 requested steps. The original 18-run
dry-run still resolves to 18 keys / 360,000,000 requested steps.

After these edits, the affected regression selection passed **200 tests, 1
warning** in 64.52s. The warning is the existing golden-trace fixture fallback
for missing entry-gate fields. Ruff and Python compilation passed for the
campaign controller and its tests.

Per the explicit bounded-training instruction in the active task, one H100
12-hour allocation was requested in tmux as job **20928644**. At this ledger
update it is pending scheduler resources; the tmux pane has append-only output
capture configured. No trainer has started. The request is reserved for the
two-run exploratory comparison, not the 18-variant campaign. The campaign
remains **NOT READY** until a fresh final-holdout source is frozen and the
legacy checkpoint provenance is recovered or those archives are formally
excluded from reanalysis.

### Bounded GPU campaign completion — 2026-10-13

The requested H100 allocation became job **20928644** on `gpu47` with 48 CPUs,
512 GiB requested memory, and the existing tmux session. The original
two-variant run exposed two report-export timezone defects: mixed aware/naive
trade times in per-trade equity plotting, then differing UTC offsets across a
DST boundary in MAE/MFE chart slicing. Both were corrected by normalizing
timestamps to UTC or to the source bars' timezone. Ruff and Python compilation
passed for the two changed modules.

The initial A0/A1 attempts are retained under
`outputs/gpu_campaigns/dev_a0_a1_seed42_100k_exposed_2026/`; both completed
training and evaluation but failed report export, so they are incomplete
diagnostic evidence only. The corrected, fresh-fingerprint campaign is
`outputs/gpu_campaigns/dev_a0_a1_seed42_100k_exportfix_exposed_2026/`,
fingerprint `fade673eaafd44fdf5374f3b75abb4c334ba496528bc71642c84372cd4afb824`.
Both runs completed the requested 100,000 steps (106,496 vectorized steps
recorded) and exported reports and checkpoints. The controller correctly
recorded both as failed because each failed the mandatory in-sample
training-equity acceptance gate; the failures were not waived.

| Variant | Exposed 2026 return | Sharpe | Max DD | Trades | Survived | Train equity gate |
|---|---:|---:|---:|---:|---|---|
| A0 flat | -10.01% | -1.383 | 12.12% | 546 | No | Failed |
| A1 SL | -11.02% | -2.643 | 12.35% | 60 | No | Failed |

The hash-indexed metrics, effective flags and dimensions, model checksums,
training-log checksums, and prior incomplete-attempt checksums are in
`outputs/gpu_campaigns/dev_a0_a1_seed42_100k_exportfix_exposed_2026/reports/exploratory_comparison.json`.
These results use the previously inspected 2026 period and are explicitly
selection-acknowledged development evidence. They establish no viable
candidate and support no final-performance claim. Repeating A0 with the same
nominal seed produced different metrics, so seed-only bitwise repeatability is
not established. No risk or architecture change was selected from these runs.

The supplementary process-tree sampler observed roughly 42–48 GiB PSS and
about 4.3–4.7 GiB GPU memory, with cgroup fields unavailable in this
interactive allocation. This is a bounded one-worker observation, not a
concurrency sizing result. The allocation and tmux shell remain open as
requested; no training process remains active.

**Readiness remains NOT READY for the 18-variant thesis campaign.** A fresh
post-2026-07-03 holdout and recovered provenance for the missing historical
A2/A3 checkpoints (or a documented exclusion of those checkpoints) remain
necessary, along with the listed parity and failure-path evidence. The two-run
exploratory campaign is complete and failed its acceptance gates.

### Real-market parity and controller recovery closure — 2026-10-13

The bundle worker constructor had not forwarded `strict_research_eligibility`
and `allow_simplex` from the resolved environment configuration. It now forwards
both flags, and `tests/test_envs/test_bundle_env_equivalence.py` adds a regression
for this constructor contract. The focused suite passed **8 tests** after the
change; Ruff passed for the worker and test.

Ran `scripts/matrix/audit_real_market_bundle_parity.py` against the real
2026-01-02 through 2026-07-03 market calendar on the existing H100 allocation.
All **18/18** variants passed. Each paired frames/bundle run compared 49,925
calendar steps with an all-zero policy action vector and 10,000 steps with an
active deterministic action sequence. Comparisons included reset and per-step
observations, rewards, termination and info, trade logs, equity/time, PnL,
breach events, pending orders, account/position state and counters. Every
reported maximum scalar delta was **0.0**. The all-zero action-vector trace is
not a literal hold: context-driven strategies can still open trades. Exact
inputs and provenance are in
`outputs/gpu_campaigns/real_market_frames_bundle_parity_20261013/reports/frames_bundle_parity.json`;
it records raw M1 hashes, supplemental loaded tick-cache hashes, feature-cache
hashes, source hashes and executed/current audit-script hashes. The report's
trace name was clarified after execution, with the wording-only change and both
script hashes disclosed in its provenance note.

The controller now hashes artifacts for failed as well as successful attempts
and includes failure reason, verification error and checksums in the failure
summary. Resume clears stale failure metadata, starts a new attempt-specific
directory, and preserves previous attempt files. Added an integration test
covering a failed worker, retained hashes, and a successful second attempt.
`tests/test_campaign_artifacts.py` passed **11 tests**; Ruff passed for the
controller and artifact tests.

The real-market parity and failed-attempt recovery gates are closed for these
specific inputs, variants and traces. They do not close feature causality or
split-leakage audits, sustained resource sizing, broader CI, fresh final-holdout
evaluation, or checkpoint provenance/exclusion decisions. The 18-run production
campaign remains **NOT READY**; the completed A0/A1 exploratory campaign remains
failed at its required training-equity gates. No production worker was started.

As an additional local science regression check, ran the causal-feature, data
split, assembled-causality, cache/fit-scope, walk-forward, fold-normalization,
variant-resolver, shared-bundle and eligibility-wiring suites: **49 passed,
2 warnings** in 13.00s. These checks close the targeted fixture/regression gate
only; a timestamp-level causality and purge/embargo audit on the complete real
M1/M5/M15/H1 series is still open, as is a fresh locked final holdout.

Finally ran the complete Linux pytest suite on Python 3.12: **1,348 passed,
38 warnings** in 476.96s. This includes training, risk, action, bundle,
evaluation, script-smoke and campaign-resume tests. It closes the full local
regression gate, but does not replace real-data causality/leakage certification
or sustained compute-resource measurements. The H100 allocation and tmux shell
are still open by prior instruction, with no active trainer.

### Legacy checkpoint disposition — 2026-10-13

Searched the two incomplete historical A2/A3 run directories, adjacent campaign
logs and repository output metadata for the missing effective configs. No
run-specific config, training log, run manifest or versioned action schema was
recoverable. The adjacent logs provide partial details only: each run resumed
from 19M steps toward 20M; the launch line reports 64 environments while the
effective PPO line reports one; and the archive action widths (A2=7, A3=11)
conflict with current real-data-instantiated widths (A2=8, A3=12).

To prevent unverifiable archives from influencing the next campaign, the
decision record
`docs/thesis/legacy_a2_a3_checkpoint_exclusion_2026-10-13.json` excludes all 42
archive files (40 unique SHA-256 contents) from candidate ranking, checkpoint
selection, comparative reanalysis and confirmatory thesis claims. The files and
partial logs remain untouched. The decision is reversible only if the original
effective configs, source/data/feature identity, vectorization and action-layout
provenance are recovered and checked against the recorded hashes. This closes
the archive-disposition gate for new work; it does not certify those historical
models.

Completed real-market feature prefix-invariance audits on the existing H100
allocation. The audit resolves effective feature configs through the
authoritative variant resolver and confirms the A0 raw-data/config hash matches
the actual exploratory training cache (`a95b9d0a4c82f06e`). The registry has two
feature profiles (baseline; PO3/distribution with PD context), covering all 18
variants. All six profile/date checks passed at train, DST and later OOS
cutoffs; all four additional checks passed at the exact train/test seam (last
train bar 2025-12-31 23:49; first test bar 2026-01-02 01:05). The reports show
zero changed features and max absolute numeric delta 0.0 for 418 baseline and
486 PO3/distribution features. Inputs, source files, script hash, versions,
variant groups and cutoff timestamps are recorded in
`outputs/gpu_campaigns/real_market_feature_causality_20261013/reports/`.
This closes sampled real-M1 feature-prefix invariance only; it does not certify
execution, reward, termination or checkpoint-selection paths, nor provide a
fresh holdout.

The campaign artifact verifier now requires `attempt_provenance.json` to match
the campaign fingerprint, run key, variant, seed, requested step budget, attempt
number and serialized identity hash before accepting a checkpoint. This closes
a prior gap where a valid config/checkpoint could be admitted without evidence
tying it to the current campaign attempt. Focused artifact tests now pass
**13/13** after adding rejection cases for both campaign and identity hash
mismatches; Ruff passes. The final full-suite rerun including the verifier and
resource-I/O telemetry changes completed with **1,351 passed, 38 warnings in
487.03 seconds** (exit code 0).

The resource sampler now records process-tree cumulative `rchar`, `wchar`,
storage read and storage write bytes alongside PSS, cgroup memory and GPU
samples. The one-worker 2M A0 profile completed 2,007,040 steps at ~889 PPO
steps/s over a 2,257-second learning phase. Its final checkpoint loads on CPU,
but the mandatory equity/survival gate failed: end equity $89,773.27, return
−10.23%, slope −0.0217, peak trailing drawdown 10.82%, and a max-loss stop. The
2026 test evaluation had zero trades and was inactive. This is resource evidence
only and not a candidate pass. The 93 samples reached 45.4 GiB process-tree PSS,
5,439 MiB GPU memory and 28% sampled GPU utilization; cgroup telemetry was
unavailable. The process-I/O fields were all zero because this sampler process
had imported the pre-fix `/proc/PID/io` parser. The later parser fix passed a
current-process smoke check but was not exercised in a sustained campaign. The
Slurm job-step counters showed approximate 36.6-minute deltas of 146.8 GB read
and 137.9 GB written; these are allocation-step totals, not worker-only I/O.
Full evidence and artifact paths are recorded in
`outputs/gpu_campaigns/resource_profile_a0_seed42_2m_exposed_20261013/reports/resource_profile_summary.json`.

Full-tree `.venv-x86/bin/ruff check .`, `ruff format --check .`, `mypy .` (389
source files) and `git diff --check` pass. The final full suite, run as
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest -q`,
completed with **1,351 passed, 38 warnings in 487.03 seconds** (exit code 0).
The initial unbounded-thread attempt was interrupted after 64 passing tests;
the successful bounded-thread run is the recorded full result. Resource report
JSON also parses successfully.

### Resume after Idea 1_2 MTF integration — 2026-10-13

Resumed from `docs/thesis/codex_pause_handoff.md` after the separate Idea 1_2
work. The checkout remains on `main` at
`b00b5e85622ec1410c8a11cc063cd52c87d3976d`; existing dirty and untracked work
was preserved. Idea 1_2's M1 PO3 setup clock can consume causally completed
IFVG confirmations and active zones from configured M1/M5/M15/H1 sources; the
builder also supports the standard M30/H4/D1 resampling options. Local swings,
structure and selected sweep references are computed on their own M5/M15/H1
bars and aligned to M1 only after confirmation. Implementation details and
known event priority are recorded in `docs/decisions/po3_ifvg_decisions.md`.

After the MTF changes, the full CPU suite completed with **1,394 passed, 38
warnings**. It exposed a stale delayed-fill timestamp assertion in
`tests/test_live/test_rl_bridge.py`; the test now follows the established
bar-open timestamp convention and checks the pending order's scheduled fill.
That full-suite run preceded a type-only test-indexing cleanup in
`tests/test_features/test_po3_state.py`; the affected feature, variant and live
bridge group then passed **44 tests, 2 warnings**. Full-tree Ruff check,
format-check and Mypy passed after that cleanup (Mypy covered 395 source
files). No training or Slurm job was launched.

These results close local regression and static-check gates for the current
implementation. Campaign readiness remains **NOT READY**: A0/A1 exploratory
pilots fail the required equity/survival acceptance; the 2026 period is
exposed development evidence; no fresh post-2026-07-03 holdout is available;
the 2M A0 profile's process-I/O samples are invalid and cgroup limits/peaks
were unavailable, so resource sizing is open; execution/reward/termination and
checkpoint/model-selection reviews still need a complete research-level
acceptance record; and the frozen 18-run campaign has not launched. Do not
interpret local tests, feature alignment, a loadable checkpoint or resource
samples as evidence of a viable strategy or confirmatory performance.

### Available-data authorization and checkpoint-selection review — 2026-10-13

The researcher confirmed that work may proceed with the data currently
available and that post-July data is not required for the requested
continuation. Inspected all files under `data/`, including hidden and nested
files. The existing US100.cash and US500.cash M1 files start on 2024-12-30 and
end at 2026-07-03 19:59; the available US100 tick file starts on 2025-12-29
and ends at 2026-07-03 19:59:44.709, while the US500 tick file starts on
2024-12-30 and ends at 2026-07-03 19:59:44.388. Header and final records were
read from each file. There is no newer export in this repository data
directory. Use this period for the authorized exploratory continuation and
retain `oos_role: selection_acknowledged`; its prior exposure remains
permanent, so it cannot support a pristine or confirmatory final-holdout claim.
`data/examples/README.md` identifies the raw bars/ticks as MT5 exports and the
bar timestamps as broker-server time (`Etc/GMT-3`). The available tick files do
not provide full coverage for US100 before 2025-12-29. SHA-256 identities for
the current primary inputs are:

| Input | SHA-256 |
| --- | --- |
| `data/US100.cash_M1_202412300105_202607031959.csv` | `59c14312a9d42a53adbcdd24925f991e488d3c7346a583810ead57fb897ce897` |
| `data/US500.cash_M1_202412300105_202607031959.csv` | `d4563f58c5bd367dfa11a92a9cb22e9d3fb10a241575c2572046365a8422e6d9` |
| `data/US100.cash_202512290105_202607031959.csv` (ticks) | `87716e3d6ee8d0f5031f83641f1e438931e29d7299a3c80114f34b9e51865d91` |
| `data/US500.cash_202412300105_202607031959.csv` (ticks) | `f0ee746eab46bada61b39a8877a4c40745eef8941e50ae9d01e149c4b127ad91` |

Read-only review of the current training and evaluation flow found that
`BestCheckpointEvalCallback` evaluates the first 20,000 training bars (or the
shorter available train set), stochastically, for one episode, once at the
configured final training timestep, and saves `ppo_best`. The ordinary
post-training report evaluates the final in-memory model deterministically on
the configured test split; it does not load `ppo_best`. `eval_run` defaults to
`ppo_final.zip`, but can fall back to the newest archive if that file is
missing, and it permits an explicit checkpoint override. Walk-forward folds
are run after the ordinary report and use only the training split. No observed
branch uses locked-test metrics to select a checkpoint automatically. This is
source-path evidence, not a completed model-selection review: human selection
after viewing exposed-period reports remains possible, and the one-episode
`ppo_best` criterion is too noisy to describe as robust checkpoint selection.
For exploratory comparisons, preserve the existing exposure label and record
any human choice; for any report-only comparison, pin the intended checkpoint
explicitly and retain its hash. No evaluation or training was run during this
review.

### Resumed CPU audit — 2026-10-13

The full real-market prefix-invariance audit was rerun on CPU-only Slurm job
20937646 (`batch-csl`, one node, three single-core workers, 32 GiB RAM request).
The three effective feature profiles (baseline, PO3/IFVG, distribution) each
compared five timestamps through 2026-06-15 against the full-data reference.
All 15 comparisons passed with zero changed features; profile reports and the
merged, source/input-hashed report are in
`outputs/gpu_campaigns/real_market_feature_causality_20261013_resume/`.
Slurm exit code was 0. This strengthens the tested real-market feature-prefix
invariance evidence, but does not certify execution, reward, termination,
checkpoint-selection or human-selection paths, and does not create fresh or
confirmatory evidence from the already exposed 2026 period.

The final bounded-thread full pytest rerun completed with **1,397 passed, 38
warnings** in 680.78 seconds (exit code 0), logged at
`outputs/gpu_campaigns/readiness_verification_20261013_resume/logs/full_pytest.log`.
The Codex CLI args test was adjusted to invoke `codex exec ... --help`, which
checks parser compatibility without starting a timed interactive execution.
The focused test passed; full Ruff check, format check, Mypy (396 source files),
`git diff --check`, and edited shell-script syntax checks passed.

### Exploratory 18-run campaign launch — 2026-10-13

The researcher authorized an exploratory run using the previously inspected
market period through 2026-07-03. A separate
`selection_acknowledged` campaign specification was dry-run checked at exactly
18 variants, seed 42, 20,000,000 requested steps per variant (360,000,000 total),
frames backend, one worker and eight environments. It is distinct from the
frozen report-only/production specification and cannot create confirmatory
claims. The controller is running under persistent tmux `quant-rl-train`
on GH200 Slurm job 20938752. The account caps a GH200 allocation at 720 GPU
minutes, so this allocation is 12 hours; the campaign may need resumable
allocations. At the first progress check, A0 had completed about 700,000
steps at approximately 755 sampled steps/sec; controller cgroup sampling saw
47.5 GB peak memory and zero OOM events (the cgroup reports `memory.max=max`).
These are early execution/resource observations, not strategy acceptance or
sustained final resource sizing. Full progress and attempts are under
`outputs/gpu_campaigns/ladder18_seed42_20m_exploratory_selection_acknowledged_20261013/`.

### Live campaign checkpoint — 2026-10-14 00:34 EEST

Revalidated Slurm job 20938752 as RUNNING on `gpuarm1` (GH200, 16 CPUs,
256 GiB requested, 12-hour allocation limit) and tmux session `quant-rl-train`.
The A0 seed-42 attempt had reached 3,145,728 of 20,000,000 PPO steps at about
733 sampled steps/sec. Controller cgroup memory peak was 47.8 GB, with zero
OOM events; `memory.max` reports `max`. One worker remained running and 17
runs pending. Interim equity values varied across checkpoints; no final A0
acceptance result exists yet. Do not infer strategy quality from those interim
metrics. The live logs, progress and campaign manifest remain the execution
source of truth.

### Live resource checkpoint — 2026-10-14 00:42 EEST

Slurm job 20938752 remains RUNNING on `gpuarm1` in partition
`gpu-grace-h200-141g`, with one GH200, 16 CPUs, 256 GiB requested RAM, and a
12-hour limit ending at 11:15:46 EEST. The exploratory A0 seed-42 attempt had
reached 3,506,176 of 20,000,000 requested steps; 17 variants remained pending
and no run had been admitted as successful. At 00:41:48 EEST the live sampler
recorded 42,648,461,312 bytes current cgroup memory, a 47,847,305,216-byte
observed peak, `memory.max=max`, zero OOM events, 8% sampled GPU utilization,
and 5,966 MiB GPU memory used. This is an interim single-trainer observation,
not full-run resource sizing. The sampler does not report per-process CPU
utilization, so CPU saturation and the benefit of a larger CPU request remain
unverified. The 256 GiB request has substantial observed RAM headroom; current
data do not support requesting 600–900 GiB for one trainer. Do not extrapolate
the one-trainer peak to two trainers without a sustained concurrent profile.

At 00:43:53 EEST, live Slurm step accounting (`sstat -j 20938752.0`) reported
`AveCPU=01:45:31` against `Elapsed=01:28:07`, `NTasks=1`, `AveRSS=42,349,812K`
and `MaxRSS=46,725,884K`. Interpreting `AveCPU` as task-aggregated
system-plus-user CPU time gives about 1.20 CPU core equivalents on average,
or 7.5% of the 16 allocated CPUs, if the accounting sample includes the
trainer's child processes. This suggests the current single trainer is not
CPU-bound, but is not per-process CPU profiling. It does not prove the right
request for concurrent trainers; two-trainer sustained profiling is still
needed before increasing the worker count. Slurm step disk counters were
357,304,752,583 bytes read and
353,367,603,967 bytes written at this snapshot; these are job-step counters,
not isolated PPO process I/O.

At 00:45:27 EEST, the A0 attempt showed 3,702,784 steps since its run
directory was created at 23:19:50 EEST, a cumulative observed rate of about
721 steps/sec. Straight-line arithmetic implies about 7.7 hours for one
20-million-step run and 138.7 hours for 18 such runs at this rate. This is a
single partial run, so the projection is a scheduling estimate only; it omits
long-run rate changes, reporting overhead, retries, and allocation waits.

### Campaign resume-policy audit — 2026-10-14

Inspection of `scripts/matrix/campaign_controller.py` shows that `--resume`
verifies successful artifacts but resets every `running`, `interrupted`, and
`failed` job to `pending`. A training-equity-gate rejection is stored as
`failed` by `verify_run_artifacts`, so a plain resume retries that scientific
failure as well as infrastructure interruptions. Each retry increments the
attempt number, creates a new attempt directory, and invokes
`scripts/train/train_one_variant.sh` with a fresh output path; it does not load
the prior attempt's periodic PPO checkpoint. Failed attempts remain on disk,
but repeated attempts under the same seed can create selection bias if a later
passing run is treated as replacing an earlier gate failure.

Do not resume this campaign blindly at an allocation boundary. Preserve
completed gate failures as outcomes, and retry only work interrupted by
infrastructure under a predeclared policy. The current controller does not
distinguish those cases in its resume transition. The active allocation was
left running; no retry or manifest edit was performed. Safe multi-allocation
continuation remains an open operational gate for this campaign.

Allocation-boundary procedure: wait until Slurm reports the allocation
terminal, then inspect the final campaign manifest and failure summary. Use
the original `--resume` path only if no completed run has failed its
training-equity gate. If a gate failure exists, keep that outcome and its
attempt artifacts terminal; continue only pending or infrastructure-
interrupted keys in a new campaign slice with a distinct campaign ID and the
same frozen per-run settings. The final 18-row report must link each key to
its originating manifest and attempt, preserving gate failures and any
restarted partial attempt. Do not select the best attempt for a key or erase
the original failed result.

### Four-trainer failure artifact audit — 2026-10-14

Reopened the retained `concurrent_4x8_local_v2` artifacts from
`outputs/gpu_campaigns/ram-validation-20261010/reports/`. The launcher status
records all four workers exiting with code `-15` after 435.21 seconds. Worker
logs A–D are each zero bytes; no worker report JSON exists, and the launcher
script only starts the four subprocesses, waits for them, and writes their
exit codes. The retained artifacts contain no CPU, GPU, memory, or I/O samples
for this attempt, and do not identify who sent SIGTERM. Therefore they do not
establish startup latency, filesystem contention, or any other root cause.
Treat four-worker scaling as unsupported by this attempt; use the separate
short 2x8/2x16 reports only within their stated short-run limits.

### 18-worker GH200 placement correction — 2026-10-14

The first one-node launch, Slurm job `20943263`, requested two GH200s, 64 CPUs,
1000 GB RAM and six hours on `gpuarm1`. Live telemetry showed all 18 worker
steps assigned to physical GPU 0, leaving GPU 1 idle. At the last live sample,
GPU 0 was at 13% utilization and 26,141 MiB used; GPU 1 was at 0% and 0 MiB.
The job cgroup reported 235,046,842,368 bytes current memory and
238,271,791,104 bytes peak. All 18 jobs were active with `n_envs=4` (72 vector
environments); six logs had entered PPO and no checkpoint had been written.
The sampled aggregate CPU consumption was 1,317,697,919 usec after about 7.5
minutes, approximately 2.9 CPU core equivalents on average.

The allocation was stopped before any model checkpoint or completed run. Slurm
records job `20943263` terminal as failed after cancellation, with worker steps
cancelled. The campaign manifest was reconciled to 18 `interrupted` runs; the
pre-reconciliation manifest and stop report are retained at
`outputs/gpu_campaigns/ladder18_seed42_20m_exploratory_selection_acknowledged_18way_20261014/reports/campaign_manifest_before_gpu_binding_reconciliation.json`
and `.../campaign_stop_reconciliation.json`. Its partial logs are not training
results.

The controller now explicitly balances workers across GPUs using Slurm
`--gpu-bind=map_gpu:<id>`. A live Slurm probe confirmed map IDs 0 and 1 select
different device UUIDs on `gpuarm1`. The CPU-only sampler no longer reserves a
GPU. The replacement campaign keeps the same 18 variants, seed, 20M steps,
data, and exploratory evaluation policy, but has a distinct identity:
`config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml`.
It requests 18 workers split 9 per GH200, 3 CPUs and 4 environments per
worker, one node, 64 CPUs and 1000 GB RAM. Resource sizing remains exploratory
until sustained telemetry is reviewed. The 17 campaign-artifact tests pass;
scoped Ruff, Python compilation, shell syntax and the 18-run dry-run pass. A
`srun --test-only` check on 2026-10-14 forecast 2026-10-16 13:52 EEST; it is
not a reservation or guaranteed start time.

The corrected persistent supervisor is running in tmux session
`thesis-18way-gpu-balanced-20261014`. Slurm job `20943576` started on
`gpuarm1`; the new manifest reports 18/18 worker slots running, assigned 9 to
each GPU. At 49 seconds elapsed, none had entered PPO yet: cgroup memory was
87,814,737,920 bytes, CPU usage 142,299,279 usec (about 2.9 core equivalents
over the elapsed interval), and both GH200s were at 0% with only 3 MiB visible
per device. This is startup telemetry, not a steady-state resource profile.

### Evaluation reporting and live 18-worker resource probe — 2026-10-14

The training dashboard's close-reason aggregation now classifies each close:
structural TP slots are grouped, EMA exits are named, and unrecognized reasons
are retained as `Other / unlisted`, so the shares use all closes as their
denominator and sum to 100%. This repairs the older display that showed 95.7%
structural stops at 115 closes while silently omitting the remaining five. The
reward table and chart now label the PnL term as a reward contribution, clearly
separate from account-currency PnL.
The exact identities of those five historical closes are not recoverable from
that aggregate line alone.

Periodic evaluation is labeled as a training-prefix monitor. Its plots now
separate return/drawdown, account-currency PnL, risk ratios, profit factor,
win rate, trade count and turnover. Final train-in-sample and test-report-only
replays now log net PnL with account currency and the key metrics separately,
and emit
`evaluations/train_in_sample.json`, `evaluations/test_report_only.json`,
`evaluations/summary.json` and `evaluations/summary.md`. The test split is
explicitly report-only because it was previously inspected. Focused evidence:
35 tests passed across the dashboard, train/evaluation artifact and resource
sampler suites; Ruff format check and Ruff check passed on all touched paths.

These code changes were made after the currently running workers imported the
training modules and therefore do not retrofit their output. Revalidated the
live resource probe as Slurm job `20944844`, RUNNING on `gpuarm2` with two
GH200s, 128 CPUs and 900 GB requested RAM; tmux session `thesis-organizer`
still has its `probe18` window. Its 18-worker, 100,000-step, 64-environment
probe manifest reported 16 jobs running and 2 failed at the latest poll. The
failed seed-42 variants are `ladder_a1_sl_po3` and `ladder_a4b_po3_ifvg`; both
exited 1 after failing the in-sample training-equity acceptance gate. Their
distinct attempt logs and artifact hashes remain recorded. The other 16
workers were still in evaluation/export at this poll. This is a resource probe
on selection-acknowledged development data, not the frozen 20M thesis
campaign or candidate-selection evidence. Do not infer completion from the
manifest while the Slurm job is live, and do not cancel or restart it.

The resource sampler recorded 62 process-tree samples from 00:14:46 to
00:58:01 UTC. Process-tree PSS averaged 238.6 GiB and peaked at 251.2 GiB.
The raw two-device `nvidia-smi` output is malformed into one CSV field by the
older sampler; parsing its two raw rows gives sampled mean utilization of
65.8% and 66.5% (peaks 92% on both devices), with 44,167 MiB and 43,583 MiB
reported used on the latest row. Treat those per-device utilization values as
reconstructed from the raw rows. The recorded cgroup values (16 MB current,
33 MB peak, `memory.max=max`) are from the sampler's own step, not the worker
allocation, and are invalid as job-memory measurements. This sampler version
also has no process-tree CPU-time field; CPU sizing therefore remains open.
These measurements are a partial resource observation while final evaluations
were running, not a terminal peak or a production resource recommendation.
Reconcile the terminal resource sample, Slurm accounting, all 18 outcomes,
and controller state once job `20944844` ends.

The production readiness decision remains **NOT READY**. This update does not
close the independent research sign-off, fresh pristine holdout for
confirmatory claims, final 20M campaign, sustained resource-sizing review, or
the remaining research acceptance gates in the readiness plan.

### Live 100k resource probe checkpoint — 2026-10-14 01:09 UTC

Re-polled Slurm and the on-disk controller manifest. Job `20944844` is still
RUNNING on `gpuarm2` with 2 GH200s, 128 CPUs and 900 GB requested RAM. Manifest
status is 3 failed, 15 running, 0 succeeded. The additional failure since the
previous 16/2 checkpoint is `ladder_a0_flat`; it joins `ladder_a1_sl_po3` and
`ladder_a4b_po3_ifvg` as terminal train-equity failures. No failed attempt was
retried.

The failed workers' saved training metrics are:

| Variant | Return | Net PnL (USD account) | Max DD | Sharpe | PF | Win rate | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ladder_a0_flat` | -2.67% | -2,670.60 | 6.72% | -0.364 | 0.875 | 39.3% | 89 |
| `ladder_a1_sl_po3` | -4.96% | -4,956.43 | 7.13% | -0.642 | 0.802 | 34.5% | 58 |
| `ladder_a4b_po3_ifvg` | -1.84% | -1,841.15 | 1.84% | -1.249 | 0.000 | 0.0% | 8 |

All are training replay metrics, not test evidence or a cross-variant
conclusion. The resource CSV has 78 rows (00:14:46–01:09:13 UTC), mean
process-tree PSS 235.9 GiB and peak 251.2 GiB. Its 16-column header lacks the
CPU-time field present in the current sampler source, so no CPU utilization
claim can be made for this run. Cgroup memory fields are sampler-step scoped,
not allocation scoped. The live process therefore does not validate a 900 GB
minimum or the CPU sizing. Continue to leave the allocation untouched; perform
terminal resource reconciliation and summarize all 18 outcomes only after it
ends. Readiness remains **NOT READY**.

Slurm `sstat` supplies cumulative `AveCPU` for the 15 active worker steps.
Dividing each step's CPU time by elapsed time gives 0.27–0.34 CPU cores per
worker on average during the first ~56 minutes, about 4.4–5.7% of the six CPU
cores assigned per worker. The 15 workers averaged roughly 4.8 CPU cores in
aggregate during this interval. This is a time average, not an instantaneous
sample, and may change by phase; it indicates low CPU use in the observed
window, not a sustained final allocation recommendation. The three failed
workers' Slurm MaxRSS values were about 14.9–20.6 GiB per step; per-step peaks
do not replace allocation-wide PSS or cgroup peak accounting.

### Evaluation runtime and deterministic gate correction — 2026-10-14 01:15 UTC

Parsed phase timings from all 18 live worker logs: PPO learning median 9.4
minutes (range 5.9–13.7); final test replay median 16.1 minutes (10.4–24.6);
and seven completed full training replays median 27.6 minutes (18.6–31.8).
These observed replay times exceed PPO learning time in this 100k probe and
identify evaluation as the current wall-time bottleneck. Timing boundaries
include environment setup and the replay but precede later report export; the
measurements are incomplete for training replay and are not production
forecasts.

The current checkout before the fix passed `deterministic=False` for the full
training replay used by the acceptance gate, while the final test call was
deterministic. Both final splits now go through `_evaluate_final_split`, which
forces deterministic policy actions; the periodic dashboard monitor remains
stochastic. The split JSON now records `policy_action_mode: deterministic`,
and the Markdown summary states that both final replays are deterministic.
The checkpoint-determinism and report-artifact test files pass 15/15 combined;
scoped Ruff format and lint pass. The live attempts have a different recorded
`train_rl.py` source hash than the current checkout and no source snapshot, so
the exact action flags used by those workers cannot be proven from the current
source. The change applies only to future launches; any future campaign must
have a new fingerprint. This code correction does not certify any live result.

### Standard multi-timeframe feature configuration — 2026-10-14

The feature audit found that most MTF blocks accepted explicit standard
resampling timeframes, but Idea 1/2's confirmed PO3 sweep references only
accepted M5/M15/H1. Added `features.po3_reference_timeframes`, preserving that
default explicitly in both Idea YAMLs and allowing M5/M15/M30/H1/H4/D1. The
standard resampler supports this set; arbitrary minute intervals are not
claimed. M1 remains the decision/feature spine. MTF structure was exercised
using all six standard higher-timeframe values.

Pivot availability now uses the next actual resampled timestamp as the candle
completion boundary. D1 uses the next local calendar day to handle 23/25-hour
days at DST changes. Focused feature tests pass **44/44**, covering all six
HTF sweep identifiers, structure output, prefix stability, invalid/duplicate
timeframes and a D1 Helsinki DST boundary. Ruff format and lint pass on the
touched Python files. Existing Idea defaults and feature schemas remain the
same; changed config values produce different feature-cache identities.
This is implementation/test evidence, not proof that MTF features improve a
strategy or real-history causality for every configuration/timezone.

### 18-run interim outcome update — 2026-10-14 01:16 UTC

Job `20944844` remains RUNNING on `gpuarm2`. The controller manifest advanced
to 12 failed, 6 running, 0 succeeded at 01:16:26 UTC. Fourteen worker logs
contain a failed train-equity gate; twelve are terminal, while
`ladder_a4a_entry_fsm_hidden_dist` and
`ladder_a4a_entry_fsm_hidden_po3` are still running during artifact export.
None was retried.

All 18 workers have logged final test replay outcomes. Twelve fail the
130-day survival metric. The remaining six survive with at most two trades;
two of them place zero trades, and the others return between -0.10% and
-0.51%. No variant has an active positive full-period test result. These are
report-only outcomes on a previously inspected split, not confirmatory
evidence. Per-run details remain in
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/logs/workers/`.

The six workers still active at this snapshot are `ladder_a3_multi_tp`,
`ladder_a4a_entry_fsm_hidden`, `ladder_a2_tp_po3`,
`ladder_a4a_entry_fsm_hidden_po3`, `ladder_a4b_distribution`, and
`ladder_a4a_entry_fsm_hidden_dist`. Poll the same Slurm job and reconcile all
18 terminal outcomes before calling the resource probe complete. Its current
evidence does not justify selecting a strategy or starting the 20M campaign.
Readiness remains **NOT READY**.

**Near-terminal state at 2026-10-14 01:17 UTC:** manifest status is 17 failed,
1 running, 0 succeeded. The sole active worker, `ladder_a3_multi_tp`, has
finished its report-only test replay and is executing its in-sample training
replay. The Slurm allocation remains RUNNING; wait for the worker and
controller to reach terminal state before reconciling this as a completed
resource probe.

### Campaign failed-attempt resume contract re-audit — 2026-10-14

The current `campaign_controller.py` was re-read after the earlier resume-policy
audit. It converts `running` to `interrupted` if a controller ended while a
worker was active, and only `interrupted` attempts return to the pending queue.
`failed` attempts are left terminal; the runtime scheduler only launches
`pending` jobs. `test_failed_attempt_artifacts_are_hashed_and_remain_terminal_on_resume`
verifies the failed exit code, attempt number and retained hashes stay intact
on resume without a retry. The focused campaign artifact suite passed **17/17**.
The earlier audit's statement that this source resets scientific failures is
superseded for the current controller. The live probe remains untouched; its
two observed gate failures are terminal outcomes and must not be retried under
the same campaign identity.

### Standard multi-timeframe feature configuration — 2026-10-14

The feature audit found that most MTF blocks accepted explicit standard
resampling timeframes, but Idea 1/2's confirmed PO3 sweep references only
accepted M5/M15/H1. Added `features.po3_reference_timeframes`, preserving that
default explicitly in both Idea YAMLs and allowing M5/M15/M30/H1/H4/D1. The
standard resampler supports this set; arbitrary minute intervals are not
claimed. M1 remains the decision/feature spine. MTF structure was exercised
using all six standard higher-timeframe values.

Pivot availability now uses the next actual resampled timestamp as the candle
completion boundary. D1 uses the next local calendar day to handle 23/25-hour
days at DST changes. Focused feature tests pass **44/44**, covering all six
HTF sweep identifiers, structure output, prefix stability, invalid/duplicate
timeframes and a D1 Helsinki DST boundary. Ruff format and lint pass on the
touched Python files. Existing Idea defaults and feature schemas remain the
same; changed config values produce different feature-cache identities.
This is implementation/test evidence, not proof that MTF features improve a
strategy or real-history causality for every configuration/timezone.

### Final 18-worker resource-probe reconciliation — 2026-10-14 01:25 UTC

Reconciled the current campaign manifest, per-attempt artifacts, sampler CSV,
and Slurm accounting for job `20944844`. Controller state is terminal at
18 failed / 0 interrupted / 0 pending / 0 running / 0 succeeded. All 18
attempts have `train_equity_gate.json`, `training/metrics.json`, and
`testing/metrics.json`; all train-equity gates failed. The three A0 variants
also all failed their gates. On the report-only test replay, 12 variants did
not survive all 130 days, while the remaining six traded zero to two times.
No active positive full-period result is established. No retry was launched.

Generated the checksummed non-ranking report at
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md`
and `.json`. The report contains per-variant train/test return, account-currency
PnL, trade counts, test drawdown/Sharpe/profit factor/survival and hashes for
the manifest, controller status and per-attempt evidence. Validation: the
focused report-builder suite passes (2 tests), and Ruff check passes for the
report script and test. Added a per-split total-PnL reconciliation against
final account equity; all 36 saved train/test splits match within $0.02, with
zero missing or mismatched PnL records.

The 95-row resource CSV has 94 active samples; process-tree PSS averaged
218.9 GiB and peaked at 251.2 GiB. Reconstructed per-device GPU utilization
averaged 66.4% on both GH200s, peaking at 97% and 96%. Slurm `AveCPU` over
each completed worker step yields a mean of 0.35 core equivalents per worker
(range 0.31–0.47; about 5.9% of each six-CPU allocation). The sampler lacks
process-tree CPU time and allocation-wide cgroup memory scope, so these
measurements do not certify production allocation sizing. The final sampler
row at 01:19:14 UTC has no campaign processes. Job `20944844` itself remains
RUNNING as a retained allocation; it was not canceled, per the researcher's
instruction.

This completed 100k resource probe is distinct from the frozen 20M campaign,
which has not launched. Test results are on previously inspected,
selection-acknowledged data and are descriptive only. The researcher permits
using currently available data through 2026-07-03 for exploratory work; a
pristine holdout remains necessary for confirmatory claims. Independent
research review, a passing training gate, remaining execution/reward/termination
and model-selection acceptance, and production resource sizing remain open.
The thesis/production readiness status is **NOT READY**.

### Three-worker 20M run interruption reconciliation — 2026-10-14 01:30 UTC

Inspected the separate campaign
`ladder3_strategy_seed42_20m_exploratory_selection_acknowledged_20261014`.
Its controller snapshots are stale at 3 running, while Slurm accounting
shows allocation `20944286` FAILED after 35 minutes and worker steps
`20944286.2`–`.4` CANCELLED. The allocation supervisor log records the srun
step aborted by SIGKILL. The three latest saved dashboard rows are at 704,512
steps (baseline and PO3) and 606,208 (distribution); no run reached its
20,000,000-step request.

No train/test `metrics.json` or final evaluation summary exists for any of
these workers. The progress dashboard CSV contains return percentage and
trade count but no account-currency eval PnL, so these partial prefix replays
cannot be reported as final train/test PnL. Added an evidence-preserving
reconciliation at
`outputs/gpu_campaigns/ladder3_strategy_seed42_20m_exploratory_selection_acknowledged_20261014/reports/campaign_reconciliation.md`
and `.json`, plus the captured `sacct_20944286.txt`. The stale original
controller manifest was preserved. Do not use `--resume` merely to refresh
its status because resume relaunches interrupted variants. This partial run
is separate from the 18-worker 100k resource probe and remains incomplete;
overall readiness remains **NOT READY**.

### Retained allocation and report-index recheck — 2026-10-14 01:37 UTC

Read-only `squeue`/`sacct` confirms retained allocation `20944844` remains
RUNNING on `gpuarm2` (128 CPUs, 900G requested), but its 18 campaign worker
steps `.7`–`.24` all ended FAILED with exit 1. The allocation wrapper step
`.0` remains live; the last sampler row at 01:19:14 UTC recorded
`campaign_active=0` and zero processes. Earlier `.2`–`.4` steps were cancelled
when the researcher requested stopping the initial three workers. No cancel or
replacement step was issued.

The 18-worker summary report was regenerated with direct per-row links to the
worker log and run-artifact directory. The Markdown and JSON reports show
18/18 train-equity gate failures and 36/36 saved split PnL values reconciled to
final equity; report-builder tests pass 2/2 and scoped Ruff passes. This closes
only report navigation and evidence reconciliation, not any scientific gate.
Production readiness remains **NOT READY**.

### Saved training-gate reason analysis — 2026-10-14 01:40 UTC

Extended `scripts/matrix/summarize_campaign_results.py` to report the
overlapping reasons from each saved train-equity gate. Current 18-worker probe
counts: end equity below start **17/18**, nonpositive equity slope **15/18**,
max-loss breach stopped trading **12/18**. Updated the report JSON/Markdown;
all 36 train/test PnL records still match saved final-equity deltas. Report
builder tests pass 2/2, scoped Ruff passes, and `git diff --check` passes.
These outcomes do not identify a controller bug and do not justify weakening
acceptance criteria. Production readiness remains **NOT READY**.

### Execution/reward regression refresh — 2026-10-14

The focused current-worktree fill, stop-causality, termination, entry-state,
sizing, broker/engine parity, reward-attribution and strategy-reward suite
passes **107 tests without warnings**. Corrected the golden no-naked-position
test fixture to include entry-gate columns and make the gate permit the test
entry request, while leaving swing-stop features missing. This removes its
unrelated permissive-fallback warning and makes the regression exercise the
intended stop-selection condition. Exact command and limits are recorded in
`docs/thesis/research_path_review_2026-10-13.md`. This bounded suite does not
close independent academic review, full production/live parity or strategy
acceptance; readiness remains **NOT READY**.

### 100k probe source-fidelity audit — 2026-10-14 01:43 UTC

Added repeatable source-hash comparison tooling at
`scripts/matrix/audit_campaign_source_fidelity.py` with a focused regression.
Across all 18 attempt provenance records, the same 194-file launch inventory
is recorded; 186 files match the current checkout and 8 have changed hashes.
No files are missing. The artifacts store hash identifiers and dirty-file
hashes, not the old source contents, so the 100k probe is not fully replayable
from the current checkout. Its metrics/equity artifacts remain inspectable and
all 36 PnL checks still reconcile. Reports:
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/source_fidelity_audit.md`
and `.json`. The main campaign report links the audit. The new audit test and
report-builder tests pass 3/3; Ruff and `git diff --check` pass. This narrows
the reproducibility claim; readiness remains **NOT READY**.

### Evaluation report clarity and live-data audit continuation — 2026-10-14 02:44 UTC

Backfilled a clearly labeled per-run evaluation summary for the saved
`ladder_a0_flat_po3` 100k probe and linked it from the campaign summary table:
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/runs/attempts/ladder_a0_flat_po3__seed42/attempt1/ladder_a0_flat_po3__seed42/20261014_031516_135481_rl_train_seed42_tcn/evaluations/summary.md`.
The report separates the one train-prefix dashboard monitor from the two final
split replays. The final saved PnLs are +$2,141.26 in-sample (81 trades; gate
FAIL) and -$6,213.34 on the report-only previously inspected test split (212
trades). The early dashboard snapshot has 115 close events, about 110
structural stops (95.7%), and five close events (4.3%) with no reason labels in
the old schema; their exact categories are unrecoverable. The saved final
in-sample replay has 81 entries and every lot size is nonzero. No strategy
claim or readiness status changes.

The current dashboard source includes explicit close categories/counts,
including `Other / unlisted`, explicit train-prefix-monitor labeling, monitor
PnL, and a JSONL monitor log; final train/test metrics are separately written
to `evaluations/summary.md` and JSON. The 100k run predates these changes, as
confirmed by source-fidelity differences. The campaign report builder now links
a run's evaluation summary when present. Its regenerated Markdown/JSON report
succeeds and `git diff --check` passes.

The real-market feature-prefix causality re-audit is still running on login
CPU as PID 877558, started with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-x86/bin/python scripts/matrix/audit_real_market_feature_prefix_causality.py --out outputs/gpu_campaigns/readiness_reaudit_20261014/feature_prefix_causality.json`.
At this checkpoint it has built the baseline full-history reference and
completed the first cutoff comparison; it is rebuilding the 2026-03-09 prefix.
It has used about 57 minutes, one CPU core and under 9 GiB RSS. No final audit
JSON exists yet, so no real-data causality result is claimed. The retained
Slurm allocation `20944844` remains RUNNING but has no active campaign workers;
no training or additional Slurm step was started. Run the full frames/shared-
bundle real-market parity audit only after this prefix audit finishes.

### Prefix-audit live recheck — 2026-10-14 03:52 UTC

Re-polled the same process/session: PID 877558 and exec session 91767 remain
live on the login CPU, about 2h06m elapsed, ~99% of one CPU and 6.0 GiB RSS.
The last progress output was the PO3 profile's full-history feature build
(rows=528,760); the baseline profile has completed its three cutoff checks,
but no final JSON has been written. The retained Slurm allocation
`20944844` remains RUNNING on `gpuarm2`; the 18 campaign worker steps are
terminal and no new step or training was started. Do not restart the audit
while this process remains active. Full frames/shared-bundle real-market
parity is still queued behind this run.

### Prefix-audit live recheck — 2026-10-14 04:36 UTC

The same process/session (PID 877558 / session 91767) remains live. At
04:36 UTC it has run for 2h50m, used about 2h49m of CPU at ~99% of one core,
and held about 6.8 GiB RSS. Its latest emitted stage remains the PO3
2026-03-09 prefix (414,573 rows); the final JSON is not present yet. This is
an active CPU-bound build, not a dead process or an observation timeout. The
retained allocation `20944844` remains RUNNING on `gpuarm2`; no training or
additional Slurm step was started. Keep parity queued until this exact audit
terminates, then inspect the full report before deciding whether parity can
run.

### Prefix-audit CPU and profiler recheck — 2026-10-14 04:49 UTC

The same audit remains live at PID 877558 / session 91767 after about 3h02m.
`ps -L` shows its Python main thread using ~99% of one CPU; the only other
thread is an idle jemalloc background thread. RSS is about 9.3 GiB, with
roughly 73 GiB available on the host. No final report has appeared. A
read-only `pstack` attempt was denied by host ptrace policy (`Operation not
permitted`), so no attached stack profile is available. Slurm allocation
`20944844` remains RUNNING on `gpuarm2`, and its worker steps are terminal.
Keep this computation intact; run the frames/shared-bundle parity audit only
after the final causality report is written and inspected.

### Feature-audit stage advancement — 2026-10-14 04:54 UTC

The same live prefix-audit process/session advanced beyond the March cutoff:
its last flushed output says it is rebuilding the PO3 profile through
2026-06-15 12:37 Europe/Helsinki (509,438 rows). At the latest process check
it remains CPU-bound at ~99% of one core, with RSS around 4 GiB and ample
host memory available. The final JSON is still absent. This confirms forward
progress after the earlier long March-prefix build; preserve the process and
continue monitoring it. The frames/shared-bundle audit remains queued.

### Retained allocation shell reconciliation — 2026-10-14 05:00 UTC

Read-only `sacct`, `squeue`, process-tree and tmux checks clarify that job
`20944844` is not an idle compute shell. The 128-CPU/900G allocation and step
`.0` remain RUNNING; the tmux `organizer` window is executing
`run_campaign_3way_shared_bundle_n64_supervisor.sh`, while its `probe18` window
is a login-shell prompt. The separate 18-worker step `.5` and all worker steps
`.7`–`.24` are terminal FAILED; the earlier three worker steps `.2`–`.4` are
CANCELLED. The allocation must remain in place per the researcher's request,
but there is no reusable interactive compute shell for the parity audit and
the allocation is still occupied by its supervisor. No additional Slurm step
was started.

The login-node feature-prefix audit (PID 877558 / session 91767) remains live
at ~99% of one CPU core, with no final report yet; it has advanced to the PO3
June 2026 cutoff. Continue the existing audit. Start the real-data
frames/shared-bundle parity step only after the audit finishes and after an
authorized compute shell is available.

**Retained-allocation process check — 2026-10-14 05:01 UTC:** the separate
three-strategy campaign manifest records all three runs as `interrupted`.
`squeue --steps` lists only allocation step `.0` and `.extern`; `sacct` shows
the earlier three worker steps cancelled and the 18-worker attempt failed.
The still-running `.0` allocation wrapper has only 4 seconds accumulated CPU,
about 32 MiB average RSS and about 2.0 GiB maximum RSS over five hours. This
supports the conclusion that no trainer is active, while the node's 128 CPUs,
900G memory and two GH200s remain reserved. Preserve the wrapper/allocation;
do not treat its reserved resources as an available interactive shell.

**Supervisor retry hazard — 2026-10-14 05:09 UTC:** the 3-strategy
`campaign.json` still has three `interrupted` runs, and
`run_campaign_3way_shared_bundle_n64_supervisor.sh` explicitly counts
`interrupted` as active. When the current six-hour `srun` ends, its loop will
wait 30 seconds and request another allocation unless the campaign becomes
terminal. This would restart the stopped 3-run campaign. Preserve the current
allocation until its scheduler time limit; when its current `srun` returns,
stop the login-side supervisor before it issues another request. Do not use
`scancel`; no trainer worker step is currently active.

### Prefix audit and retained-allocation guard recheck — 2026-10-14 05:14 UTC

The same real-market feature-prefix audit remains live as PID `877558` and
exec session `91767`, now about 3h27m elapsed, using ~99% of one CPU and
10.6 GiB RSS. It has not written its final JSON, and its progress file is not
present; preserve this exact process and do not restart it. The full
frames/shared-bundle trace audit remains queued behind it.

Slurm job `20944844` is still RUNNING on `gpuarm2` with 2 GH200s, 128 CPUs,
900G requested memory, and a six-hour time limit. Its only live step is the
allocation supervisor `.0`; the earlier three worker steps `.2`–`.4` are
CANCELLED by the researcher, the 18-worker campaign step `.5` FAILED, and all
18 worker steps `.7`–`.24` FAILED. The campaign sampler and worker logs show
no active trainers. Tmux `thesis-organizer:0.0` still runs the login-side
three-strategy supervisor (PID `4046630`) while its current `srun` parent and
child remain live (PIDs `4046645` and `4047359`).

Started the guarded post-allocation supervisor stop as exec session `62783`,
using `logs/stop_supervisor_after_allocation.sh`. It waits for a new
`allocation returned exit=` record, the current `srun` process to exit, and
the expected supervisor command/pane before sending Ctrl-C. It does not call
`scancel` and does not interrupt the current reservation. Confirm the guard
and natural allocation completion before proceeding; no additional Slurm step
has been started. The 18-run 100k probe remains 18/18 train-equity failures,
and the frozen 20M campaign remains unlaunched. Overall readiness remains
**NOT READY**.

### Completed-worker CPU accounting cross-check — 2026-10-14 05:16 UTC

Requeried `sacct` for the 18 terminal worker steps `.7`–`.24` in allocation
`20944844` and recomputed CPU seconds from each step's `TotalCPU`. The workers
accumulated 22,180.6 CPU-seconds over 63,626 combined worker-seconds, a
weighted average of 0.349 CPU cores per worker (range 0.306–0.470), against
six CPUs allocated per worker. Across the overlapping worker interval
03:14:45–04:19:15 UTC, this is about 5.7 aggregate CPU cores against 108
worker CPUs allocated, or 5.3% average use. This is a Slurm whole-step average
including preprocessing, PPO, evaluation and report export; it is not a
training-only or peak CPU profile. It corroborates that requesting more CPUs
would not address a demonstrated CPU bottleneck in this probe. The existing
resource report's PSS/GPU figures remain bounded samples, not allocation-wide
cgroup peaks. No new training or Slurm step was started.

**Prefix-audit profile advancement — 2026-10-14 05:20 UTC:** the same live
audit emitted the next profile marker for `ladder_a0_flat_dist` (528,760
source rows) after the prior PO3 profile's three cutoffs completed. This
confirms the audit has progressed through the baseline and PO3 groups and is
now building the distribution reference; the final report is still pending.
The retained allocation `20944844` remains RUNNING, and the guarded stop
session `62783` remains active. No new Slurm step was started.

### Walk-forward evaluation determinism correction — 2026-10-14

Source review found the canonical purged walk-forward selection path used
`deterministic=False`, unlike the final train/test reports. It now routes each
fold replay through `_evaluate_final_split`, which forces deterministic policy
actions. This aligns future fold scores with the validation protocol and
removes sampled-action noise from that selection measure. The focused existing
deterministic-evaluation test passes 2/2, covering the shared helper and
checkpoint callback; source inspection confirms the fold path calls that
helper. No walk-forward training, result selection or new strategy evidence
was produced. Scoped Ruff format/lint and `git diff --check` pass for the Python
source changes. Updated the seed-variance diagnostic documentation to separate
its intentionally sampled-action sensitivity results from the deterministic
final reports. Independent review and strategy acceptance remain open.

**Live process checkpoint — 2026-10-14 05:34 UTC:** PID `877558` / exec
session `91767` remains CPU-bound (~99% of one core, ~4.9 GiB RSS) in the
distribution profile; no final causality JSON exists yet. Slurm allocation
`20944844` remains RUNNING at 5h34m elapsed, with no campaign worker steps
active. The guarded login-side stop remains live as exec session `62783`.
Continue polling these exact handles; do not restart the audit or cancel the
allocation.

### Explicit checkpoint reanalysis action mode — 2026-10-14 05:39 UTC

Corrected the checkpoint reanalysis CLI's misleading stochastic-sampling help
and warning. It now defaults explicitly to deterministic policy actions and
exposes sampled actions only through `--stochastic`; `--seed` controls that
stochastic replay and both action mode and seed are recorded in the reanalysis
manifest and output `training_log.json`. Added focused parser and replay-wiring
regressions. `tests/test_eval/test_eval_run_provenance.py` passes 15/15; scoped
Ruff format/lint and `git diff --check` pass. No archived checkpoint was
replayed, so existing evaluation results are unchanged.

The real-market feature-prefix audit remains live as PID `877558` / exec
session `91767`; its JSON output is absent and the current distribution-profile
comparison has not completed. Slurm allocation `20944844` is still RUNNING
with the supervisor step only; no trainer is active. Guard session `62783`
remains responsible for preventing the login supervisor from issuing another
allocation after natural completion. Do not restart the audit or cancel the
allocation. The 18-run campaign remains NOT READY and unlaunched at 20M;
independent research review and a fresh untouched holdout remain open.

**Natural allocation timeout and guard verification — 2026-10-14 05:43 UTC:**
Slurm job `20944844` reached its six-hour walltime and exited `124`; no
`scancel` was issued. The watchdog log confirms it observed the `srun` return
and sent Ctrl-C to the login-side three-strategy supervisor, preventing an
automatic replacement allocation. `squeue` no longer lists the job, the
supervisor process is gone, and tmux `thesis-organizer` retains only its
unrelated probe window. The active real-market causality audit PID `877558` /
session `91767` continues on login CPU at ~99% of one core and ~10 GiB RSS;
72 GiB was available in the shared login-node memory snapshot. Its final JSON
has not appeared. Preserve this audit; do not restart it. No trainers are
active and the frozen 18-run campaign remains unlaunched.

### Live broker stop and aggregate-risk hardening — 2026-10-14 05:52 UTC

Source tracing found that `RLRobot` and `MultiSymbolRobot` sized orders using
`stop_loss_pips` but submitted them with `sl=None`. `RiskManager` also summed
absolute floating PnL as a proxy for aggregate open risk, so its 2% cap did not
represent loss-to-stop. Changed both robots to attach the price stop built by
the same pip/tick conversion used for sizing. `RiskManager` now floors volume
to the broker step, refuses sizes below minimum rather than forcing minimum
volume, calculates each open position's stop risk from entry/SL/volume/tick
value, and blocks new exposure when position data or any existing stop risk is
unknown. Its constructor defaults now match config (1% per position, 2%
aggregate); the standalone RL robot defaults to paper mode.

Verification: `pytest tests/test_live -q` passes **47 tests, 2 known feature
fixture warnings**; scoped Ruff format/lint and `git diff --check` pass. Added
mocked tests for buy/sell stop price, broker stop submission, aggregate
stop-risk cap, unknown/missing stops and position-query failure. No MT5 broker
was available. The live bridge still maps to direction-only signals and the
fixed 50-pip stop; training's structural exits and cumulative daily/max-loss
guardrails are not implemented live. `PAPER_TRADING=true` remains required;
this is not a live-deployment readiness claim.

The real-market feature-prefix audit remains PID `877558` / exec session
`91767`, CPU-active at roughly 99% of one core and about 6.6 GiB RSS at the
latest check; no final JSON exists. The prior 2-GPU allocation timed out
naturally and its login supervisor was stopped by the guard. No trainer or
Slurm allocation is active. Do not restart the audit; do not launch the
18-variant campaign until the scientific gates are resolved.

### Live-risk regression rerun and audit checkpoint — 2026-10-14 05:55 UTC

Re-ran the complete live test package after adding the minimum-volume-step
rounding regression: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 ./.venv-x86/bin/pytest tests/test_live -q` passes **47
tests, 2 known feature-fixture warnings** in 1.80 seconds. The warning is the
existing bridge fixture missing London/Asian levels and `volume_spike`; it is
not a live-risk assertion failure.

The real-market prefix-causality audit remains the same live process, PID
`877558` / session `91767`, at 99% of one CPU and about 5.7 GiB RSS. It has
written no final JSON and has emitted no new progress in the re-polled session;
preserve it and keep the full frames/shared-bundle parity audit queued. Slurm
queue inspection is empty after allocation `20944844` ended naturally. The
single `thesis-organizer` tmux session still has one live `probe18` shell, but
there is no active trainer or Slurm allocation. No job was cancelled or
submitted during this recheck. Readiness remains **NOT READY**.

**Post-format verification — 2026-10-14 05:57 UTC:** formatted the one
regression-test file Ruff identified, reran `tests/test_live` (**47 passed, 2
known warnings in 1.95s**), and passed scoped Ruff format/lint plus
`git diff --check`. The same audit PID remains CPU-active (99%, about 6.0 GiB
RSS at this sample); its final JSON is absent. `squeue` is empty and
`sacct` records allocation `20944844` as terminal `FAILED`, exit `124`; no new
allocation was started.

### Higher-timeframe feature regression verification — 2026-10-14 05:58 UTC

Ran the targeted higher-timeframe/PO3 feature suite:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
./.venv-x86/bin/pytest tests/test_features/test_htf_sweeps.py
tests/test_features/test_po3_config.py tests/test_features/test_po3_state.py
tests/test_features/test_po3_ifvg_state.py
tests/test_features/test_session_ohlc.py
tests/test_features/test_build_pipeline.py -q` passes **117 tests**. Scoped
Ruff format/lint across those six source and six test files, plus
`git diff --check`, passes. This verifies the configured standard timeframe
path and its feature regressions; it is not a substitute for the in-progress
full-market prefix audit or a strategy-performance result.

The same real-market prefix audit PID `877558` / session `91767` remains
running after 4h11m, using about 99% of one CPU and 10.0 GiB RSS at the latest
sample. It has not written `feature_prefix_causality.json`; preserve the live
process and keep real-data backend parity queued. Slurm has no active jobs.
Readiness remains **NOT READY**.

### Campaign gate, resolver and frozen dry-run recheck — 2026-10-14 05:59 UTC

Read-only inspection of `campaign.json` and the checksummed results report
confirms the shared-bundle 100k resource probe has **18/18 terminal failed
attempts**, with no pending or running variants. A controller dry-run of
`18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml`
returns exactly 18 unique seed-42 run keys, 20,000,000 requested steps each
(360,000,000 total), with the launcher defaults 18 workers, 4 environments
each, 3 CPUs per worker, and two GH200s on `gpuarm1`. The dry-run created no
allocation or trainer. This placement is a launcher configuration, not a
validated 20M resource profile; the failed pilot is not eligible for resume or
candidate selection.

Campaign admission, result-summary, source-fidelity, variant-resolver and
run-manifest tests pass **38/38**. This verifies those local contracts, not a
successful strategy or launch readiness. The real-data prefix audit remains
active at PID `877558`; no report is present. Overall readiness remains
**NOT READY**.

### Execution, reward and survival regression refresh — 2026-10-14 06:00 UTC

Re-ran the focused execution/reward/survival regression set documented in the
research review. The 13 selected test modules pass **107/107**. Scoped Ruff
format/lint for the corresponding environment, backtest, reward-attribution
and test paths passes; `git diff --check` passes. This is focused source-level
regression evidence and does not demonstrate profitable behavior or full
training/live parity.

At 06:00 UTC, the existing real-market prefix-causality process remains alive
(PID `877558`, session `91767`) at 99% of one CPU; its output JSON is still
absent. No Slurm jobs are queued, and no training/allocation has been started.
The source-causality result and subsequent full-market backend audit remain
open; the campaign remains **NOT READY**.

### Dashboard and evaluation-report regression refresh — 2026-10-14 06:02 UTC

The current dashboard, train/test report writer, deterministic checkpoint
evaluation and reanalysis-provenance tests pass **45/45**. Scoped Ruff
format/lint for the affected train/eval modules and tests, plus
`git diff --check`, passes. These contracts ensure future runs distinguish
the stochastic train-prefix progress monitor from deterministic final train
and report-only test replays and include account-currency PnL in monitor and
summary outputs. The historical 100k artifacts retain their old schema and
their five unclassified close reasons remain unrecoverable.

The live prefix audit remains PID `877558` / session `91767`, CPU-active at
99% and about 4.2 GiB RSS in the latest sample; final JSON is still absent.
The Slurm queue is empty. Readiness remains **NOT READY**.

### Raw 18-run outcome reconciliation — 2026-10-14 06:04 UTC

Independently re-read the 18 run-artifact directories referenced by the
checksummed campaign report. All manifest jobs are terminal failures; all 18
`train_equity_gate.json` files say `ok=false`. Reconciled each saved training
and test `equity.csv` against its split `metrics.json`: **36/36** PnL values
match final equity deltas exactly at the stored precision (maximum difference
$0.00); the 18 training gate end-equity values also match their train equity
curves within $0.02. The underlying outcomes agree with the report: overlapping
train-gate reasons are end equity not above start 17/18, nonpositive slope
15/18, and max-loss stop 12/18; test survival fails 12/18 and the other six
surviving runs make at most two trades. This confirms accounting consistency,
not profitable or confirmatory performance. No training was started.

### Data split and normalization regression refresh — 2026-10-14 06:05 UTC

Re-ran causal-feature, split-boundary, assembled-causality, cache fit-scope,
train-only normalization and purged walk-forward contract tests: **28 passed,
2 expected Transformer warnings**. Scoped Ruff format/lint and
`git diff --check` pass. This verifies local leakage-prevention contracts;
real-market prefix invariance remains pending the active audit and the exposed
2026 test interval remains non-confirmatory.

### Resource-probe raw sampler reconciliation — 2026-10-14 06:08 UTC

Independently re-parsed the resource CSV referenced by the report and verified
its SHA-256. Its 95 rows contain 94 active samples. Recomputed process-tree PSS
at 218.90 GiB mean and 251.22 GiB peak; the two GH200s show 66.44%/66.43%
average sampled utilization and peaks of 97%/96%, with per-device sampled VRAM
peaks of 44,167/43,583 MiB. These values match the checksummed summary report.
The sample contains no allocation-scoped cgroup memory or process-tree CPU
time, and it covers a 100k probe, not sustained 20M training. Use it as a
bounded resource observation, not proof that 18 × 20M at `n_envs=64` is safe.

### Independent-review packet and audit-handle recheck — 2026-10-14 06:15 UTC

Prepared `docs/thesis/independent_research_review_checklist_2026-10-14.md`
with evidence paths, decision questions and a reviewer disposition record. Its
relative links and the links from the plan/research review were checked. This
prepares an external review; no reviewer has signed it, so the research
sign-off gate remains open.

Re-polled the existing feature-prefix audit: PID `877558` / session `91767`
is alive, CPU-active at 99% after 4h28m, and has not written its output JSON.
Preserve the run and keep current-source frames/shared-bundle parity queued.
No Slurm jobs are active. Readiness remains **NOT READY**.

### Feature-prefix audit continuation checkpoint — 2026-10-14 06:19 UTC

Polled the same audit process and tool session rather than starting a duplicate:
PID `877558`, exec session `91767`, running the real-market feature-prefix
causality command. It remains in running state at 99% CPU after 4h32m; RSS at
this sample is about 3.6 GiB, down from the prior 6.3 GiB sample. Its latest
flushed progress entered the final `ladder_a0_flat_dist` feature group and
started rebuilding the March 2026 prefix. The expected JSON report is still
absent. `squeue` is empty and `thesis-organizer` retains only its idle `probe18`
shell; no new allocation or training process exists. Keep the audit alive and
the current-source frames/shared-bundle parity audit queued until the JSON is
produced and validated. Readiness remains **NOT READY**.

### Campaign launch-source snapshot provenance — 2026-10-14 06:26 UTC

Closed an actionable future-run reproducibility gap: on **new campaign
creation**, `campaign_controller.py` now writes `source_snapshot.zip` before
launching workers. The deterministic archive contains the exact tracked and
untracked campaign source/config files covered by the campaign's source
hashes, plus an internal file-hash manifest and Git SHA. The writer checks the
source bytes against the hashes already used to build campaign identity,
verifies the archived bytes, and records the archive SHA-256 and file count in
`campaign.json`. Resume verifies the archive when that field exists; legacy
campaigns remain unchanged and are not assigned a retroactive snapshot.

Verification: the source-snapshot and campaign artifact/source-fidelity checks
pass **20/20**; the test-worker controller fixture also exercised snapshot
creation and verification on resume. Scoped Ruff format/lint and
`git diff --check` pass. No Triton campaign was created or launched. A direct
helper-level check against the frozen 20M spec selected all **196** expected
source/config files, created a verified 688,685-byte temporary archive, and
retained no archive or campaign output.
This does not recover source contents for the historical 100k resource probe;
its report remains hash-only. At 06:28 UTC the existing feature-prefix audit
PID `877558` / session `91767` remains CPU-active at 99% after 4h41m, with no
report JSON yet. Current-source frames/shared-bundle parity remains queued.
Readiness remains **NOT READY**.

### Real-market feature-prefix audit live checkpoint — 2026-10-14 06:51 UTC

Re-polled the same PID `877558` / exec session `91767`; it remains live at
99% CPU after 5h04m. Its latest flushed progress is rebuilding the final
distribution-feature prefix through 2026-06-15 (509,438 rows). The final JSON
is still absent because the audit writes it only after all profile/cutoff
comparisons complete. CPU time and minor page faults increased across the
latest 30-second sample, confirming active computation rather than an idle
process. The requested current-source frames/shared-bundle parity run remains
queued; Slurm has no active jobs. Preserve this audit and inspect its JSON
before starting parity. The 18-run 100k probe remains 18/18 failed training
gates; the frozen 20M campaign remains unlaunched. Research sign-off and a
pristine confirmatory holdout remain open. Readiness remains **NOT READY**.

### CPU parity allocation admission — 2026-10-14 07:00 UTC

The full current-source frames/shared-bundle parity audit was submitted from
the existing `thesis-organizer` tmux window as one CPU-only `batch-milan`
request (`4` CPUs, `128G`, `12h`, Slurm job `20950768`). Slurm reported
`PENDING`, `StartTime=Unknown`, and no node. Under the repository's Triton
instructions, that just-submitted request was cancelled and no replacement
allocation was submitted. No parity worker ran and no parity result is
claimed. The single earlier `thesis-organizer` window remains in place. The
feature-prefix audit PID `877558` / session `91767` is still live on the login
node; preserve it and continue to validate its output before parity. The
18-run pilot remains 18/18 failed training gates, the frozen 20M campaign is
unlaunched, and readiness remains **NOT READY**.

### Dashboard and evaluation interpretation — 2026-10-14 07:05 UTC

Inspected the saved `ladder_a0_flat_po3` attempt after the dashboard question.
The 8,192-step snapshot is 115 rollout closes: 95.7% structural stops is
about 110 closes, and the missing 4.3% (about five closes) has no reason
category in this older dashboard CSV. Its later 106,496-step monitor in the
saved worker log is +3.86%, 57 trades; the pasted -6.76% / 122-trade EVAL and
1,061-close snapshot do not match this attempt and cannot be attributed to it.
The final saved deterministic split replays are train +2.14% / +$2,141.26 / 81
trades and report-only test -6.21% / -$6,213.34 / 212 trades; these are in the
run's `evaluations/summary.md` and JSON records. The monitor is a third,
train-prefix progress check and its historical CSV does not retain dollar PnL.
The saved old `dashboard_eval.png` puts unlike metrics on one axis and has one
monitor point. Current working-tree source already separates evaluation
charts, writes monitor PnL to new dashboard CSVs, and classifies unlisted close
reasons; the pilot's source hashes show these dashboard/report files changed
since that run, so its missing labels and monitor PnL cannot be backfilled by
claim. The feature-prefix audit PID `877558` / session `91767` remains active;
no JSON output is present yet. Campaign readiness remains **NOT READY**.

### Runtime-screen regeneration and causality audit poll — 2026-10-14 07:07 UTC

Re-read the regenerated runtime screen JSON and Markdown: target window is
5.75 hours, requiring 966.2 training steps/s per worker, 5.14x the sample
median and 3.20x the sample maximum. This is a linear screen only; the sample
uses bundle/64 envs/6 CPUs while the target uses frames/4 envs/3 CPUs, so it
is not a completion-time forecast. The compact campaign summary rounds 5.75
to 5.8 hours; the threshold retains the exact value. Re-polled the existing
feature-prefix audit PID `877558` / session `91767`: it remains live at 99%
CPU; process CPU time advanced 30 seconds during a 30-second observation, RSS
fluctuated from 7.4 to 10.1 GiB, and the JSON is still absent. Preserve this
single computation. The new CPU-only parity allocation question is pending;
no training or replacement Slurm request was started. Readiness remains
**NOT READY**.

### Current-source real-market feature-prefix audit completed — 2026-10-14 07:11 UTC

The preserved PID `877558` / session `91767` completed and wrote
`outputs/gpu_campaigns/readiness_reaudit_20261014/feature_prefix_causality.json`.
It covered all 18 registered variants in 3 feature profiles and passed 9/9
comparisons at 2025-12-15, 2026-03-09 and 2026-06-15. Each comparison has
zero changed features and maximum absolute delta 0.0. Recomputed all 32 source
hashes, the audit-script SHA-256, and both raw M1 input hashes against current
files; all match. The inputs cover 528,760 rows from 2024-12-30 through
2026-07-03. This establishes prefix invariance only at the tested cutoffs; it
does not establish every execution/label path or backend parity. The plan and
review checklist now cite this bounded result. Slurm has no active jobs; the
CPU-only parity allocation question remains unanswered, so no new allocation
or training was started. The failed 100k pilot and unlaunched frozen 20M
campaign remain unchanged; readiness is **NOT READY**.

### Current-source parity audit provenance hardening — 2026-10-14 07:14 UTC

Closed a reproducibility gap before running parity: the prior audit's
hand-picked hash list omitted the variant resolver and strategy configs. The
audit now inventories every `.py` file under `quant_rl/` and every `.yaml` /
`.yml` file under `quant_rl/` and `config/`, plus `pyproject.toml`, `uv.lock`,
the variant list and the audit script. It records Python/platform and relevant
package versions, plus Git branch, HEAD and dirty paths in both progress and
final artifacts. Added a unit contract for the source inventory. Verification:
the focused audit tests pass **3/3**, Ruff format and lint pass on both touched
Python files, and `git diff --check` passes. The existing real-market prefix
report remains hash-valid; its result was added to the plan and independent
review packet. No parity trace ran: the CPU-only allocation request is still
awaiting the researcher's answer. No training was started. Campaign readiness
remains **NOT READY**.

### Frozen 20M campaign shape dry-run — 2026-10-14 07:15 UTC

Ran the controller's `--dry-run` path for
`18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml`
with the intended one-node GPU-balanced settings (18 workers, 4 envs/worker,
3 CPUs/worker, two GH200s). The persisted report
`outputs/gpu_campaigns/readiness_reaudit_20261014/production_shape_dry_run.json`
validates 18 unique run keys, seed 42, 20,000,000 steps each, 360,000,000
total steps, frames backend, and the exact campaign/output paths. The
controller's dry-run returns before output initialization or worker startup;
the target campaign output root was pre-existing and left untouched. `squeue`
is empty and no trainer/controller process was active after the command. This
closes only the run-key/shape dry-run check; it is not allocation or launch
authorization and does not override the failed pilot gate. The frozen 20M
campaign remains unlaunched and readiness remains **NOT READY**.

### Resource sampler CSV contract — 2026-10-14 07:19 UTC

Added an end-to-end test that runs one synthetic sampler iteration through
`main()` and checks that `process_tree_cpu_time_usec`, `cgroup_memory_scope`,
and allocation memory peak are written under the correct CSV columns. This
complements the existing CPU-stat parser and nested cgroup v1/v2 tests. The
focused sampler suite passes **8/8**, Ruff format/lint passes on the touched
test, and `git diff --check` passes. This verifies instrumentation only; the
historical 100k resource CSV predates these fields, and no new sustained
allocation profile has run. Campaign readiness remains **NOT READY**.

### Frozen campaign seed and budget reconciliation — 2026-10-14

Closed the seed/budget checklist item for the frozen GPU-balanced campaign,
with scope limited to that campaign. Its YAML selects the sole seed 42 and
20,000,000 requested steps per variant. The allocation launcher passes the
spec path and does not override seed or steps; the controller loads those
fields from the spec and rejects campaign-identity CLI overrides when
`--campaign-spec` is present. The saved dry-run confirms 18 unique seed-42
keys, 20,000,000 steps each, and 360,000,000 total. The registry's historical
100,000-step defaults and other legacy launch paths are not claimed to be
globally reconciled. Evidence: `config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml`,
`scripts/matrix/run_campaign_allocation.sh`, `scripts/matrix/campaign_controller.py`,
and `outputs/gpu_campaigns/readiness_reaudit_20261014/production_shape_dry_run.json`.
This closes only the frozen campaign identity/budget check; no worker or
training job was launched. Scientific and resource gates remain open, and
readiness remains **NOT READY**.

### Current-source parity audit started — 2026-10-14

Re-polled Slurm job `20951231`; it is **RUNNING** on `milan20` with 4 CPUs,
128 GiB and a 12-hour limit. Its log confirms the batch allocation and both
instrument tick caches loaded (49,160,484 US100 ticks and 31,134,588 US500
ticks). The progress JSON has one variant, `ladder_a0_flat`, in `running`
state; the audit had been active for about three minutes at the observation.
No final parity report exists yet. The two visible `srun` PIDs are the parent
and child for this same job ID. Keep observing job `20951231`; do not start a
duplicate or cancel it. Parity and overall readiness remain OPEN / **NOT
READY** until the complete report is validated.

### Campaign artifact-admission contract rechecked — 2026-10-14

Re-read `verify_run_artifacts` and its controller call site. A run can enter
`succeeded` only when its worker exits 0 and the verifier accepts the attempt
provenance (campaign fingerprint, run key, variant, seed, expected budget,
attempt and identity hash), completed timestep budget, required finite metrics,
feature-cache hash format and feature-schema/effective-config hashes,
training-equity gate, checkpoint archive integrity, CPU model load, model
timestep count, and action/observation schema. Failed
attempts retain their exit code and artifact hashes; a failed equity gate is
not admitted as a success. The focused `tests/test_campaign_artifacts.py`
result recorded above is **17/17 passed**; this turn made no test run.

The launch-readiness checklist now marks this *controller enforcement*
requirement complete. It does not claim that the 100k pilot has a successful
run: its 18 training-equity gates failed, so the frozen 20M campaign remains
unlaunched. Readiness remains **NOT READY**.

### Current-source frames/shared-bundle parity job queued — 2026-10-14

Submitted CPU-only Slurm job `20951231` to `batch-milan` for the full current-
source real-market parity audit (4 CPUs, 128 GiB, 12 hours) using the existing
exposed M1/tick data and 10,000 active-action bars per variant. It runs in the
existing `thesis-organizer` tmux pane and writes to
`outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.json`,
with its log at `outputs/gpu_campaigns/readiness_reaudit_20261014/logs/parity_srun.log`.
At the initial `squeue`/`scontrol show job` check, Slurm reported PENDING with
a scheduled start of `2026-10-14T14:01:58` on `milan1`. At the 10:28 local
recheck, it was still PENDING and the estimate had moved to
`2026-10-14T14:27:00`; no audit process or final report exists yet. Keep this
exact job; do not cancel it or submit a duplicate.
Parity remains OPEN, the independent research review remains unsigned, the
100k pilot still fails all 18 training-equity gates, and the frozen 20M
campaign remains unlaunched. Readiness remains **NOT READY**.

### Entry-gate warning and golden-trace regressions — 2026-10-14

Ran the focused continuous-action/entry-gate warning, entry-rejection counter,
strict-eligibility wiring and golden-environment trace tests:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-x86/bin/python -m pytest \
  tests/test_envs/test_continuous_actions.py \
  tests/test_envs/test_entry_rejection_counters.py \
  tests/test_envs/test_eligibility_wiring.py \
  tests/test_envs/test_trading_env_golden_traces.py -q
```

Result: **30 passed in 3.50 seconds** on Python 3.12.14 / pytest 9.1.1.
This verifies the fixture-level warning/fallback and strict rejection paths,
not full real-market eligibility behavior. Together with the 126 MTF/swing and
29 split/fit-scope/leakage checks above, the focused software regressions pass.
Current-source market frames/bundle parity still awaits Slurm job `20951231`;
the independent reviewer has not signed. Readiness remains **NOT READY**.

### Focused MTF/swing causality regression — 2026-10-14

Ran the seven current-tree feature test files covering HTF close-time
alignment, MTF feature variants, pivots/swings, HTF sweeps, PO3/IFVG zone
timing, PO3 state and IFVG state:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-x86/bin/python -m pytest \
  tests/test_features/test_htf_alignment.py \
  tests/test_features/test_variants.py \
  tests/test_features/test_pivots_swings.py \
  tests/test_features/test_htf_sweeps.py \
  tests/test_features/test_po3_config.py \
  tests/test_features/test_po3_state.py \
  tests/test_features/test_po3_ifvg_state.py -q
```

Result: **126 passed in 5.76 seconds** on Python 3.12.14 / pytest 9.1.1.
This is fixture-based regression evidence only. Combined with the separate
real-market feature-prefix report (9/9 cutoffs), it strengthens the bounded
feature causality evidence but does not close train-only fit-scope, all-path
leakage, OOS isolation, or full-market current-source backend parity. Slurm
job `20951231` is RUNNING on `milan20`; its progress file shows the first
variant processing, but no final result exists. Readiness remains **NOT READY**.

### Split, fit-scope and leakage regression — 2026-10-14

Ran the six focused files for causal-feature truncation, split boundaries,
fold normalization, feature-cache/fit scope, assembled-matrix causality and
leakage detectors:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-x86/bin/python -m pytest \
  tests/test_causal_features.py tests/test_data_split.py \
  tests/test_features/test_fold_normalize.py \
  tests/test_features/test_feature_cache_and_fit_scope.py \
  tests/test_features/test_assembled_causality.py \
  tests/test_evaluation/test_leakage.py -q
```

Result: **29 passed, 2 expected Transformer warnings in 7.01 seconds** on
Python 3.12.14 / pytest 9.1.1. Together with the 126-test MTF/swing suite
recorded above, the focused feature/split/cache/leakage regression scope now
passes. These are unit/fixture results and do not prove end-to-end real-market
execution/label-path causality. The 2026 report-only interval is exposed, not
pristine confirmatory OOS. Current-source full-market frames/bundle parity
job `20951231` is RUNNING on `milan20` with no final output yet. Readiness
remains **NOT READY**.

### Slurm CPU accounting for 18-worker resource probe — 2026-10-14

Queried completed Slurm accounting for job `20944844` and preserved the raw
27-row `sacct` result at
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/slurm_accounting_20944844.psv`
(SHA-256 `49785713e4fa627407b8a4787a8ab431e39afd4699dfc4203dce0128040585b9`).
Selected worker steps `.7` through `.24` (18 workers, 6 CPUs each), excluding
the earlier three-worker steps `.2`–`.4`. `TotalCPU` sums to **6.16 CPU-hours**;
dividing each worker step's CPU time by `ElapsedRaw` gives an elapsed-weighted
mean **0.35 CPU cores per worker step**, or **5.8%** of its six-CPU assignment.
The sampler's active 64-minute window makes summed worker consumption about
**5.78 CPU cores aggregate** on average; this is approximate because accounting
uses each full step lifetime. Per-step average use ranged 0.31–0.47 cores.
This is Slurm step accounting, not sampled process-tree CPU telemetry.

The per-step rows and derivation are in
`reports/slurm_cpu_accounting_20944844.json` and `.md`, and the raw accounting
is retained. This indicates low CPU use in the 100k shared-bundle,
`n_envs=64` probe; it does not show that more CPUs would speed training or
size the frozen frames/`n_envs=4`/20M campaign. The probe's PSS peak was
251.2 GiB, but it has no reliable allocation-wide cgroup memory peak, and all
18 runs failed their training-equity gates. Resource sizing remains OPEN and
readiness remains **NOT READY**.

### Current-source full-calendar frames/shared-bundle parity — 2026-10-14

CPU-only Slurm job `20951231` completed with exit code 0 on `milan20`: 4 CPUs,
128 GiB requested, 41:40 elapsed. The final report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.json`
with SHA-256
`372e565c913aacd0c564e19f03503d749ee4d2b1a3988f8a9aac289ab3481044`.
The readable result summary is `outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.md`;
raw Slurm accounting is retained at
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/slurm_accounting_20951231.psv`
(SHA-256 `5d24a06f811deef676fe21bb4ed8af5477d30ca228dd0a67f8ea28f6f4318f15`).
All **18/18 variants passed**. Each variant ran a 49,925-step full test-calendar
trace and a 10,000-step active-action trace. All 36 differential traces had
zero mismatches and maximum scalar delta 0.0. The exposed test calendar has
177,210 M1 rows from 2026-01-02 01:05 through 2026-07-03 19:59 Europe/Helsinki.
The 166 project source/config hashes, four raw-data hashes, and recorded dirty
path set were recomputed against the current checkout and matched. The Slurm
step reported MaxRSS 26,734,688 KiB (about 25.5 GiB) and TotalCPU 40:25.5;
these characterize this audit, not PPO training resource requirements.

This closes current-source backend parity only for the fixed traces and exposed
calendar described above. It does not establish exhaustive policy-state
equivalence, strategy viability, all-path feature/execution causality, or
confirmatory holdout evidence. The 100k 18-variant pilot still failed every
train-equity gate; independent research review remains unsigned, resource
sizing for the frozen frames/`n_envs=4`/20M shape remains open, and the frozen
20M campaign is still unlaunched. Readiness remains **NOT READY**.

### Pure legacy A0 baseline correction — 2026-10-14

Following the researcher's explicit choice, changed only `ladder_a0_flat` in
`config/experiments.yaml` to `strategy_actions=false` and
`agent_direction_control=false`. The resolver now selects the legacy
`Discrete(20)` action path and default feature profile for this variant; the
other 17 core variant settings are unchanged. This is a registry/config
correction, not a new result. The previous 100k A0 pilot resolved
`strategy.name=baseline` but `strategy_actions=true`; its `training/trades.csv`
records 55,900 `manipulation_unconfirmed` rejections, 89 opens, and lot sizes
0.4622–84.4835 (median 12.9653). Preserve its actual failed outcome, but do not
use it as the pure legacy baseline result. A0 now differs from A1 in action
schema and feature profile as well as the rung flags, so the A0/A1 contrast is
not a one-factor SL ablation.

The prior action-space and full-market frames/bundle parity reports were
generated before this correction. Their old results remain historical evidence
for the previous A0 configuration; re-run the current action/config audit and
parity before relying on them for the corrected registry. Do not launch the
20M campaign: the other 17 pilot train-equity gates still failed, and this
pure-baseline configuration has not been trained. Overall readiness remains
**NOT READY**.

### One-year train / test-window evaluation and acceptance metrics — 2026-10-14

The researcher confirmed that each final trained policy must be evaluated on
both the complete one-year in-sample training split and the following
five-to-six-month test split, with model weights frozen before test replay.
The current trainer already performs deterministic final replays on both
splits and writes `evaluations/train_in_sample.json`,
`evaluations/test_report_only.json`, and `evaluations/summary.md`; dashboard
train-prefix checks remain progress diagnostics, not final split reports.
The active campaign uses train bars through 2025-12-31 (351,550 bars) and test
bars from 2026-01-01 through 2026-07-03 (177,210 bars). The test interval was
previously inspected and remains exploratory, not pristine confirmation.

Per the clarified acceptance policy, net PnL, equity slope, and trade count
are metrics. They are not hard gates and must not stop training by themselves.
The train survival gate remains hard for a max-loss breach or peak trailing
drawdown above 10%. Default no-trade early-abort is disabled so activity stays
a measured outcome. These semantics are implemented in source commit
`16b13ae`; full-tree Ruff format/lint and Mypy passed, and 47 focused tests
passed.

### Live 18-variant exploratory campaign — current state 2026-10-14

Slurm allocation `20954895` is RUNNING on gpuarm1/gpuarm2 with one GH200,
48 CPUs, and 512 GiB per node; time limit is six hours. It contains 16
workers under the main campaign and two cache-recovery workers under a
separate campaign directory. The current manifests list those 18 workers as
running, with the main campaign's two original stale-cache attempts preserved
as failures. At the latest check, logged PPO progress was 589,824 steps for
`ladder_a0_flat` and 393,216 for `ladder_a4b_distribution`; these are early
progress snapshots against 20M requested steps, not completed results.

The running campaign was created from commit `937bae529fdf5d6c17db6174880ef448320e2883`.
Its identity fingerprint includes the source commit and source hashes; the
subsequent metric-policy commit `16b13ae` changes that identity. Current worker
processes are unaffected, but an automatic `--resume` from the updated
checkout will fail its fingerprint check. Do not rewrite the manifest or
attempt provenance to bypass that guard. Before another allocation, either
resume under the exact pinned source snapshot or create a new, explicitly
lineaged campaign from a validated checkpoint migration. Do not claim the
18-run campaign complete or READY on current partial evidence. Thesis
readiness remains **NOT READY**: independent research review, a pristine
confirmatory holdout, final campaign evidence, and sustained training resource
sizing remain open.

### Live 18-worker progress and resource sample — 2026-10-14 10:30 UTC

Slurm job `20954895` remains RUNNING on gpuarm1/gpuarm2 (one GH200, 48 CPUs,
512 GiB RAM per node; six-hour limit). The main 18-variant manifest currently
records 16 running jobs and two failed first attempts. The failures are
`ladder_a0_flat_po3` and `ladder_a0_flat_dist`; both stopped before PPO because
their configured Idea 1/Idea 2 prebuilt feature-cache hashes differed from
the effective data/config hashes. The failures remain preserved. Those
variants are active in the separate
`outputs/gpu_campaigns/ladder2_cache_recovery_seed42_20m_shared_bundle_n64_20261014`
manifest. Therefore 18 variant workers are running across the two manifests,
with the original failures still visible. At the latest worker-log sample,
16 workers had logged between 491,520 and 884,736 PPO steps; the two stale-
cache attempts logged no training steps. No final evaluations or outcomes
exist.

The resource CSVs contain 70–71 samples over about 39–40 minutes. Process-tree
PSS averaged 91.9 GiB on gpuarm1 and 118.7 GiB on gpuarm2, with peaks of 98.8
and 128.7 GiB. Slurm-job cgroup memory averaged 135.8 and 126.6 GiB; sampled
current maxima were 151.2 and 134.9 GiB, and recorded cgroup high-water marks
were 152.9 and 136.5 GiB against 512 GiB per node. OOM counters were zero.
Process-tree CPU averaged 3.91 and 4.87 cores against 48 CPUs per node.
Sampled GPU utilization averaged 56.2% and 52.1% (peaks 96% and 92%); GPU
memory averaged 38.0 and 40.2 GiB (peaks 43.3 and 45.7 GiB). This is sustained
early-phase evidence for shared-bundle `n_envs=64` at 18 workers, not final
phase sizing or evidence that more CPUs would improve throughput. It does not
establish final checkpoint/evaluation peaks or scratch-I/O needs. Sources are
the two `outputs/gpu_campaigns/ladder18_seed42_20m_purelegacy_shared_bundle_n64_20261014/logs/resources_gpuarm*.csv`
files, worker logs, and both campaign manifests.

The main campaign identity remains pinned to
`937bae529fdf5d6c17db6174880ef448320e2883` with fingerprint
`b19cf865ae01a900c425b9e7ab45fd922b5ac14b54364171cca34574ea270556`; current
repository HEAD is `6d13698`. Do not alter campaign provenance to force a
resume. Overall readiness remains **NOT READY** pending completed training
and train/test evaluations, independent research review, a pristine holdout
for confirmatory claims, and full-phase resource evidence.

### Active campaign acceptance-policy provenance caveat — 2026-10-14

The 18-worker campaign was created from commit `937bae529fdf5d6c17db6174880ef448320e2883`,
before the user-approved acceptance change in `16b13ae`. Source comparison
confirms that the pinned code's train gate requires positive end equity and
positive equity slope, and its default no-trade early abort is enabled after
the configured minimum-step threshold. The approved policy instead makes PnL,
equity slope, and activity reported metrics and disables no-trade early abort;
max-loss breach and peak trailing drawdown above 10% remain the hard
train-survival gates. The live workers therefore do not train under the latest
accepted policy. Preserve their outputs as an exploratory run under the earlier
semantics and do not interpret them as evidence for the updated acceptance
policy. Any resume must retain the exact source fingerprint and this caveat;
a campaign under the updated policy requires a new identity and source
fingerprint. The current running allocation was left untouched.

To preserve exact-source continuation for the existing campaign, created the
detached worktree
`/scratch/work/nguyenl37/Aalto_MS_Thesis_pinned_937bae` at the manifest's
`937bae5` revision. Its 201 source files match the launch archive's
`source_manifest.json`; shared data, caches, outputs, and `.venv` are linked
to the existing paths. Updated the two-node allocation launcher to detect a
manifest/current-checkout SHA mismatch and dispatch through this pinned
worktree. Shell syntax, primary campaign dry-run, and source/hash checks pass.
This retry path is prepared but not yet exercised; the active allocation and
workers were not touched. The separate cache-recovery campaign does not yet
have an automatic resume path: its spec lives under a symlinked output
directory, and a pinned-source dry-run rejects it as outside the repository.
Retain those partial outputs; do not claim the two recovery runs will resume
automatically. A provenance-preserving migration remains open.

### Live campaign follow-up — 2026-10-14 10:46 UTC

At this snapshot, Slurm job `20954895` is RUNNING with 1:20 elapsed. The main
campaign has 16 running jobs and two preserved cache-mismatch failures; the
separate recovery campaign has two running jobs. All 18 current worker logs
show PPO progress between 786,432 and 1,179,648 (median 983,040) steps, and
ten 1M-step checkpoints are present across the manifests. No final evaluations
are present.

Current sidecar hashes are baseline `656b50e064710408`, Idea 1
`75feaab2164f0ee9`, and Idea 2 `c5c8b3d9e890a760`; both idea caches have the
required HTF sweep audit files. This validates the current cache artifacts,
not the two failed attempts.

The latest resource window contains 99/97 samples over 55.5/55.8 minutes.
Process-tree PSS averages/peaks are 93.9/102.4 GiB on gpuarm1 and
121.1/131.4 GiB on gpuarm2. Slurm-job cgroup memory average/current max/high
water are 134.4/151.2/152.9 GiB and 129.2/138.8/140.7 GiB, respectively,
against 512 GiB per node. All cgroup max/OOM counters are zero. Process-tree
CPU averages 3.88/4.88 cores. GPU utilization averages 56.3%/54.6%, with
98%/96% peaks; VRAM averages 39.6/41.9 GiB, peaking at 44.0/46.8 GiB.
These measurements are early-phase only.

The primary campaign dry-run passed under the pinned 937bae source. The
separate recovery spec fails the pinned controller's in-repository path check
through the shared-output symlink; automatic recovery resume was removed from
the launcher and remains open for explicit migration. The active training
still uses acceptance semantics predating `16b13ae`.

### Live campaign follow-up — 2026-10-14 10:49 UTC

Slurm allocation `20954895` is RUNNING on gpuarm1/gpuarm2 with 4 h 37 m
remaining at the latest query. Main campaign status is 16 running and two
preserved stale-cache failures; the separate recovery campaign is 2 running.
The current primary worker logs show 786,432–1,277,952 PPO steps (median
1,081,344), and both recovery worker logs show 983,040. No worker has reached
the 20M budget and no final evaluations are present.

The user's confirmed reporting contract is implemented in the trainer:
deterministic final evaluation on the complete 2025 training split and the
following 2026 test split, written separately as
`evaluations/train_in_sample.json` and
`evaluations/test_report_only.json`. This is a source-level verification of
the output path, not evidence that the running workers completed either
replay. The test period was previously inspected and remains exploratory.

The current allocation supervisor monitors only the primary manifest. It does
not include the separate two-variant cache-recovery manifest in its completion
decision or automatically resume those runs after the allocation ends. The
current allocation and workers were left untouched. Fixing this continuity
gap requires a provenance-preserving coordinator that resumes both manifests
and reports their combined status; do not merge, rewrite, or mark either
manifest complete by hand. All active workers also remain under the older
`937bae5` acceptance semantics, not the later policy in `16b13ae`.

### Provenance-preserving policy migration tooling — 2026-10-14

Implemented a checkpoint-map builder and optional initial-checkpoint input in
`scripts/matrix/campaign_controller.py`. The map requires exact run-key
coverage, validates each source campaign snapshot and attempt identity,
checks the checkpoint hash/ZIP CRC and embedded `num_timesteps`, and selects
only checkpoints at or below the requested budget. The controller archives
the checkpoint bytes in the successor campaign and includes parent fingerprints,
attempt provenance hashes, and step counts in the new campaign identity. On
later interruption, it prefers a successor-attempt checkpoint and falls back
to the archived parent checkpoint. A full-budget checkpoint can therefore be
re-evaluated under the newer survival gate without claiming additional PPO
steps.

Prepared a distinct seed-42, 18-run, 20M shared-bundle campaign spec using the
current survival-only acceptance policy and a two-node allocation/supervisor
pair. Building a map from the live primary plus recovery manifests and running
the controller dry-run validated all 18 keys and the requested GH200/CPU/env
placement. Focused tests passed 24/24; full-tree Ruff format, Ruff lint, Mypy
(404 source files), both launcher syntax checks, and `git diff --check` passed.
No Slurm request was made and no live process or manifest was modified.

This migration is not yet launched. Do not start its supervisor while the old
supervisor continues to allocate for the primary manifest. The migration
supervisor checks the existing organizer process and `thesis18x2` Slurm job,
waiting rather than requesting a competing allocation; it also fails closed
if `squeue` cannot be queried. Once the old organizer is terminal, rebuild the
map from the newest intact checkpoints across both manifests. The new
controller runs a CPU `PPO.load` preflight for
every initial checkpoint before worker launch, but this has not run on the
GH200 runtime yet. Current evidence verifies map construction and file
integrity, not real model deserialization. Keep the old attempt history and the new
lineage distinct, and label any resulting training as a checkpoint continuation
with the source-policy boundary recorded.

The successor preflight now first attempts a strict reuse of the existing
primary campaign's content-keyed shared-environment bundles. It places new
observation memmaps in the successor output and checks existing bundle key,
version, config, source fingerprints, labels, columns, array metadata and
array values against freshly prepared current inputs. It never repairs or
overwrites a parent bundle in reuse-only mode. If any bundle is absent or
mismatched, the launcher falls back to a fresh parity cache in the successor
output. This avoids rebuilding the 7.7 GiB shared-env portion of the current
13 GiB parity cache when exact reuse succeeds; actual compute-node reuse has
not yet been exercised.

Before adding this path, rechecked the parent parity report: all 18 variants
and 36 traces had passed, all four configured raw-data SHA-256 hashes still
matched, and all 167 source paths in the old report were present. Of those,
163 hashes still match; the only four changed files are the training default,
early-abort callback, survival gate, and train-loop reporting/gate behavior.
The feature, data pipeline, trading environment, shared-bundle, environment
specification, worker, and parity-audit source hashes are unchanged. The new
parity audit still runs both traces and validates freshly prepared arrays; it
does not merely accept the old report as a current-source pass. If runtime
data/config/cache identity differs, it falls back to a fresh successor cache.

### User scratch quota recheck — 2026-10-14

`quota -s` reports 138 GiB of the user's 536 GiB scratch quota used and
456k/1.048m files, leaving about 398 GiB and 592k files of quota headroom.
The aggregate `/scratch` filesystem capacity is not the applicable personal
limit. The active primary/recovery campaign outputs occupy about 14 GiB and
100 MiB. No active campaign cache or training output was removed.

### Initial checkpoint deserialization audit — 2026-10-14 11:43 UTC

The read-only initial-checkpoint audit completed successfully for all 18
current seed-42 lineages (`passed=18`, `failed=0`) using CPU `PPO.load`; each
loaded model's embedded timestep matched its expected value. The checkpoints
were between 1M and 2M steps, so this is an early-checkpoint compatibility
check, not a final 20M checkpoint validation. The machine-readable report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_20261014.json`
and the execution log is
`outputs/gpu_campaigns/readiness_reaudit_20261014/logs/checkpoint_load_cpu_18_seed42_20261014.log`.
The report records checkpoint and source-provenance hashes, source campaign
fingerprints, runtime, repository SHA `602f5cbe40e007169e7dd826d8c0c164b08586bb`,
and a clean working tree. This does not establish GH200 deserialization,
continued training, final train/test evaluations, or policy quality.

At the latest live query, Slurm job `20954895` remained RUNNING on
`gpuarm[1-2]`; no workers or allocation were stopped. The old campaign remains
in progress, its two stale-cache attempts remain preserved, and the separate
recovery workers remain a distinct lineage. The provenance-safe successor is
still not launched. Readiness remains **NOT READY** pending final training and
both evaluations, the independent research review, and a fresh locked holdout.

### Live campaign continuity check and migration watcher — 2026-10-14 11:45 UTC

Revalidated live state instead of using the stale `reports/campaign_status.json`
snapshots (primary timestamp 09:50 UTC; recovery 09:55 UTC). Slurm job
`20954895` is RUNNING on gpuarm1/gpuarm2, with a 6-hour limit and 96 CPUs,
2 GH200 GPUs and 1 TiB aggregate requested memory. The existing primary
controller has 16 active runs and two cache-failed entries; the separate
recovery controller has two active runs. Worker logs last written around
14:45 Europe/Helsinki show approximately 1.6–2.7M steps in primary and
2.0–2.1M in recovery. No train or test final-evaluation JSON exists yet.

The primary supervisor's completion loop reads only the primary manifest and
has a three-allocation-error limit. Its controller changes interrupted runs
back to pending on resume, but leaves the two failed primary entries failed.
Recovery was launched as a distinct overlapping step after its initial
12-CPU attempt failed the `n_envs=64` CPU guard; the retry launched both runs,
but that manifest is outside the primary supervisor's status/exit decision.
These are continuity limits, not a reason to interrupt the live allocation.

Started the already-prepared migration supervisor as a background watcher in
the existing tmux server; it added no tmux window and has not requested an
allocation. Its log is
`outputs/gpu_campaigns/ladder18_seed42_20m_policy_migration_20261014/logs/allocation_supervisor.log`.
At startup it reported that it is waiting for the original supervisor and
allocation to stop. It will then build one provenance-checked 18-key map from
the latest primary and recovery checkpoints and request the successor
two-node allocation, preserving the existing job and all output. This is a
planned continuation, not completed training. The successor remains subject
to the current 20M total-step budget, separate train/test evaluations, and
report-only treatment of the exposed 2026 test window. Readiness remains
**NOT READY**.

The approximately 1.98-hour resource-sampler window through 11:48 UTC measured
process-tree CPU averages of 3.84 and 4.89 cores on nodes with 48 allocated
CPUs each (8.0% and 10.2%). GPU utilization averaged 55.6% and 55.5%; VRAM
averaged 42.5 and 45.6 GiB, with peaks of 46.2 and 51.1 GiB. Cgroup memory
averaged 135.9 and 134.9 GiB and peaked at 152.9 and 147.8 GiB, against
512 GiB/node; OOM counters remained zero. This sample indicates substantial
CPU and RAM headroom with moderate GPU utilization. It does not establish
full-run peaks or justify changing `n_envs` mid-campaign; changing vector
environment count would change PPO rollout/update semantics and comparability.
Measurements are in the campaign's `logs/resources_gpuarm1.csv` and
`logs/resources_gpuarm2.csv`.

### Latest-checkpoint CPU compatibility refresh — 2026-10-14 11:56 UTC

Repeated the checkpoint deserialization audit against the newest intact
checkpoints available from the two active source manifests. All 18 distinct
variant/seed lineages passed (`passed=18`, `failed=0`), with embedded model
timesteps matching provenance: 13 checkpoints at 2M and 5 at 1M. The report
is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_latest_20261014.json`;
it records the exact checkpoint hashes, parent campaign fingerprints,
attempt-provenance hashes, current HEAD `ef58a49b10a95a9f7fa60b52745fa5579497aeea`,
and a clean tree. The active workers have continued beyond these checkpoint
levels in their logs; these are the newest checkpoint files available at the
audit snapshot, not their final models. CPU loading does not prove GH200
compatibility or strategy performance. The live allocation and waiting
migration supervisor remain unchanged; final train/test evaluations are
still absent and readiness remains **NOT READY**.

### Migration supervisor no-progress retry fix — 2026-10-14 11:59 UTC

Fixed the prepared continuation supervisor so expected six-hour allocation
boundaries do not consume its three-error budget when training produced new
checkpoints. It now fingerprints checkpoint path/size/mtime metadata
before and after each allocation; checkpoint progress resets the no-progress
error count, while three consecutive nonzero allocations without a checkpoint
change still stop the supervisor for inspection. Restarted only the waiting
login-side migration watcher to load this change. The current Slurm job
`20954895` remained RUNNING, and the watcher continued waiting; no training
worker or Slurm allocation was stopped or requested. Validation: `bash -n`
passed, the focused campaign checkpoint migration tests passed 5/5, full-tree
Ruff format/lint and Mypy passed, and `git diff --check` passed. The retry
contract is about safe continuation; it is not evidence of training quality.

### Live checkpoint-lineage dry-run refresh — 2026-10-14 12:04 UTC

Built a fresh, read-only migration map from the current primary and recovery
manifests and ran the successor controller in `--dry-run` mode. The map
contains 18 distinct seed-42 keys with verified source hashes and step counts:
one at 3M, 16 at 2M, and one at 1M. The dry-run accepted all 18 for the
prepared 20M target, with shared-bundle data, `n_envs=64`, 18 workers, two
GH200 nodes, 5 CPUs per worker, one GPU per node, and fast reports. The map is
`outputs/gpu_campaigns/readiness_reaudit_20261014/checkpoint_map_snapshot_20261014_latest.json`.
This snapshot is only a readiness audit artifact; the migration supervisor
will rebuild its own map after the original organizer/allocation ends, so it
will capture newer checkpoints. The dry-run verifies lineage and placement,
not CPU/GH200 deserialization of the new 3M checkpoint, completed training,
evaluations, or policy quality. The live Slurm job remained RUNNING and the
watcher remained waiting; no allocation was requested or stopped.

Separately loaded the selected 3M `ladder_a0_flat__seed42` checkpoint with
`PPO.load(device="cpu")`; the embedded timestep matched 3,000,000. Its
SHA-256 is `dcd573a8dbbfc2bbd313e10539d8005f5b65d396e34b39a294beaeda43e16787`.
The report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/checkpoint_load_cpu_ladder_a0_flat_3m_20261014.json`.
This verifies one current checkpoint on CPU, not GH200 runtime, policy quality,
or completed train/test evaluation.

### Live 18-worker campaign refresh — 2026-10-14 12:10 UTC

Read-only verification on the login host found Slurm allocation `20954895`
still RUNNING on `gpuarm[1-2]` (6-hour limit; 96 CPUs, 2 GH200 GPUs and 1 TiB
aggregate memory). The existing two-node campaign supervisor and the
login-side migration watcher are both alive. The migration watcher remains
waiting for the original supervisor/allocation; it has not requested a new
allocation. No Slurm job or worker was stopped.

The primary campaign manifest still records 16 running workers and two
startup failures. Both failures are the Idea 1/Idea 2 prebuilt-cache hash
mismatch at 12:50 Helsinki time and have no checkpoint. The separate recovery
manifest has those same two run keys running, each with a verified 2M
checkpoint. Together, the manifests therefore represent 18 active unique
trainers: 16 primary workers plus the two recovery workers. The newest primary
checkpoint files are at 3M for `ladder_a0_flat` and 2M for each of the other
15 active primary runs; the two recovery checkpoints are also at 2M. The
primary and recovery source SHA is `937bae529fdf5d6c17db6174880ef448320e2883`.
This is partial training progress, not 20M completion. Neither the training
in-sample nor report-only test final evaluation JSON exists yet.

The failed primary attempts remain preserved in their logs and manifest; the
recovery campaign is a separate lineage and should not be merged or marked
complete by hand. The migration watcher is configured to rebuild a
provenance-checked 18-key checkpoint map after the original supervisor ends.
This handoff has not been exercised by a completed allocation boundary. The
the resource sampler remains active; its updated 09:50–12:11 UTC window is
recorded below. Scratch quota reported 139 GiB used
of 536 GiB and 457k files of 1.048m, with no quota pressure. Thesis readiness
remains **NOT READY**: the 20M campaign and both final evaluations are
incomplete, the 2026 test period is selection-acknowledged rather than a fresh
holdout, independent research review is unsigned, and sustained full-run
resource sizing remains open.

### Live campaign resource sample — 2026-10-14 12:11 UTC

The existing per-node sampler produced 247 samples on gpuarm1 and 239 on
gpuarm2 from 09:50 to 12:11 UTC. Process-tree CPU averaged 3.83 and 4.89
cores, respectively, out of 48 allocated per node. GPU utilization averaged
56.1% and 55.7%. Cgroup RAM averaged 137.2 and 136.8 GiB and peaked at 152.9
and 152.7 GiB; GPU memory averaged 43.1 and 46.6 GiB and peaked at 47.0 and
52.6 GiB. The recorded cgroup `oom`, `oom_kill`, and `oom_group_kill` counters
were zero throughout both sample files. The CSV evidence is in
`outputs/gpu_campaigns/ladder18_seed42_20m_purelegacy_shared_bundle_n64_20261014/logs/resources_gpuarm1.csv`
and `.../resources_gpuarm2.csv`.

This is a fresh, measured 2-hour-21-minute profile of the current shared-bundle
campaign and confirms that CPU and RAM are not near their allocation limits in
this interval; the GPUs are moderately utilized. It is not a full-run peak or
a completed throughput benchmark, and does not justify changing the current
PPO rollout configuration mid-run. Sustained resource sizing remains open.

### Live training-prefix behavior snapshot — 2026-10-14 12:14 UTC

Parsed the latest `total_timesteps` rows from the 18 active worker logs: each
has progressed to 2,064,384–3,342,336 of 20,000,000 steps. Six workers'
latest `TRAIN-PREFIX MONITOR (NOT FINAL TEST)` blocks report `FAILED trades
0`; their campaign manifests still say `running`, and their PPO logs show
subsequent rollout updates. The label is generated by the dashboard when that
train-prefix replay has zero trades; it is not a Slurm or worker terminal
state. `dashboard.py` is unchanged from the pinned worker source commit
`937bae5`. This is a meaningful interim no-trade signal for those snapshots,
but does not establish a completed one-year evaluation or final policy failure.
Full training-split and report-only test replays remain pending at the end of
the 20M budget.

### Fresh checkpoint-lineage and CPU load audit — 2026-10-14 12:22 UTC

Rebuilt the lineage map from the live primary and recovery manifests into
`outputs/gpu_campaigns/readiness_reaudit_20261014/checkpoint_map_snapshot_20261014_1216.json`.
It covers exactly 18 run keys: the newest resumable baseline checkpoint at
3M and the other 17 at 2M. The successor controller's read-only dry-run
accepted all 18 keys for the 20M, shared-bundle, `n_envs=64`, two-GH200-node
shape with 5 CPUs per worker. Its output was captured in the tool result; no
Slurm step was started. The machine-readable dry-run output is
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/campaign_migration_dry_run_20261014_1216.json`.

Then CPU-loaded all 18 selected PPO archives with the lineage audit utility;
all 18 passed and each embedded timestep matched its provenance record. The
report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_latest_20261014_1216.json`.
This validates checkpoint integrity and CPU deserialization for the current
snapshot, not GH200 loading, continued training, final evaluations, or policy
quality. Slurm job `20954895` remained RUNNING during the audit; no training
allocation or worker was stopped.

### Live campaign and resource refresh — 2026-10-14 12:35 UTC

Slurm job `20954895` remains RUNNING on `gpuarm[1-2]`; the primary campaign supervisor and migration watcher are alive. The watcher is still waiting for the current organizer/allocation and has not requested a second allocation. There are 18 active unique seed-42 variant lineages: 16 running in the primary manifest, plus two running cache-recovery replacements. The primary manifest preserves the two original stale-cache startup failures. Latest worker logs show 2,359,296–3,833,856 of 20,000,000 requested steps. No lineage has reached the budget, and no final train or test evaluation artifact exists.

Each latest monitor is a 20,000-bar training-prefix replay. Six of 18 report zero trades; across all monitors the observed trade counts range from 0 to 99. These are early, noisy diagnostics and do not indicate worker termination or final full-year policy behavior. The captured per-run values and input hashes are in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/train_prefix_monitor_snapshot_20261014_1235.json` (SHA-256 `23e0aca6318ac9b1763f98a552d97f9c657acbc6819d845eb8cb804747ffc68b`).

The live resource CSVs now cover about 2.76 hours. Process-tree CPU averaged 3.83 and 4.89 cores on gpuarm1/gpuarm2 (48 CPUs allocated per node); allocation-scoped cgroup RAM averaged 138.8 and 138.9 GiB and peaked at 151.9 and 154.8 GiB (512 GiB limit per node). GPU utilization averaged 56.1% and 55.4%; GPU memory peaked at 48.1 and 54.4 GiB. Latest OOM counters are zero. The derivation and raw-file hashes are recorded in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/resource_sample_20261014_1235.json` (SHA-256 `1f6ec8c3009fb753d059022ab3c53addfd8111c41e67966e7f04250c8bdc4f64`). This is a bounded interval, not a full-run resource-sizing result.

The configured trainer performs deterministic final replay on the complete training split and the report-only test split, writing separate `evaluations/train_in_sample.json` and `evaluations/test_report_only.json` reports. Those reports remain absent because training is incomplete. This campaign is exploratory and uses previously inspected 2026 data; it cannot provide confirmatory holdout evidence. Research review, a pristine confirmatory holdout, a passing survival result, final train/test reports, and sustained resource sizing remain open. Overall readiness remains **NOT READY**.


### Active checkpoint policy boundary — 2026-10-14 12:38 UTC

A per-run config audit found that all 18 live checkpoints still train from source `937bae5`, and their saved effective config has `training.early_abort.stop_on_no_trades=true`, `min_timesteps=5,000,000`, and `window=4`. Thus an active run can stop learning if its latest four completed training episodes each have zero trades. The current 20,000-bar dashboard's zero-trade count does not establish that condition, and no early-abort event is being inferred. The current repository policy has `stop_on_no_trades=false`, matching the researcher's instruction that trade count and PnL/slope remain metrics. The prepared checkpoint migration will use that current policy after the original organizer/allocation ends; it has not started. The live runs were left untouched.

The machine-readable comparison includes all 18 effective run configs, config hashes, manifest hashes, and latest worker progress: `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/active_run_policy_boundary_20261014_1238.json` (SHA-256 `305b84c3705af52012fd3a9c4abe57113a9677a656ef987786caea50b28fd012`). At observation, all 18 unique lineages were running, with 2.46M–3.93M of 20M steps. Preserve this source-policy boundary in any eventual interpretation; the results are exploratory and readiness remains **NOT READY**.


### Prebuilt-cache recovery verification — 2026-10-14 12:42 UTC

The two preserved primary startup failures were caused by stale content-hash sidecars: Idea 1 expected `75feaab2164f0ee9` but found `a17248b11065d6a8`; Idea 2 expected `c5c8b3d9e890a760` but found `dcb308a95a71f5db`. The recovery workers then logged successful hash validation for the current Idea 1 and Idea 2 parquet caches and built separate shared-environment bundles. Their live training continues under the separate recovery manifest; the failed original attempts remain preserved. The full cache, sidecar, log, and bundle provenance is recorded in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/prebuilt_feature_cache_recovery_20261014_1242.json` (SHA-256 `f91106dd6568ffaaaccb5055767cea9783349f206c81e165234f3604b8d4c24a`). This resolves the cache-startup blocker for those two attempts, but does not establish strategy quality or final training results.


### Pinned-resume provenance audit — 2026-10-14 12:45 UTC

Verified the old campaign's prepared allocation-boundary resume path before it is needed. The pinned checkout exists at commit `937bae5` with a clean worktree; its `outputs` symlink resolves to the active shared output tree and the campaign manifest hash matches. All four raw input SHA-256 values, the source snapshot ZIP hash, and the current baseline/Idea 1/Idea 2 cache sidecar hashes match the active manifest or expected config hashes. The machine-readable evidence is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/pinned_source_resume_audit_20261014_1245.json` (SHA-256 `104b071afdc2ae35d6c8bfc957126dfaba93d91e19f641a202ea1233e372ca6e`). This confirms the original supervisor can safely continue the existing lineage under its pinned source; it also means the updated no-trade metric policy will not apply until the separate checkpoint-migration campaign. No allocation was started or stopped.


### Full active-versus-current policy comparison — 2026-10-14 12:45 UTC

The source comparison extends beyond the no-trade callback. All 18 live runs use commit `937bae5`: their saved configs enable the 5M/four-episode no-trade abort, and that source's final train gate also requires positive net equity and positive equity-time slope. Current `HEAD` follows the researcher's updated acceptance decision: no-trade, PnL, and slope remain metrics; only a max-loss breach or peak trailing drawdown above 10% fails the train survival gate. No live worker has logged `EARLY_ABORT`; the 20K-bar monitors do not determine the four-episode callback condition. The prepared checkpoint migration uses the current policy and is waiting for the original organizer/allocation to end. The live workers were not modified.

The source-blob and per-run config comparison is saved at `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/active_run_policy_boundary_20261014_1245.json` (SHA-256 `d985790a1c88b7051f6bdafda2455e251c03f59c6da1923b9c1833c8c5e6a83e`). It records 18 active unique lineages at 2.56M–4.13M steps, zero observed early-abort events, and the exact old/current gate source hashes. Treat any eventual result as a checkpoint continuation across this recorded training-policy boundary; readiness remains **NOT READY**.


### Extended live resource sample — 2026-10-14 12:48 UTC

The allocation sampler now covers approximately 2.97 hours. Process-tree CPU averaged 3.83/4.89 cores on gpuarm1/gpuarm2; allocation-scoped cgroup RAM averaged 139.7/140.0 GiB and peaked at 154.9/157.8 GiB against 512 GiB/node. GPU utilization averaged 56.3%/55.6%, with GPU memory peaks of 48.5/55.3 GiB. All sampled OOM-related counters remained zero. The derivation and raw CSV hashes are in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/resource_sample_20261014_1248.json` (SHA-256 `352745c81160d4ef73fd557922c8fcfbb00b4b2376c225f5514b550c978183dd`). CPU use remains low relative to allocation; this observation is not a full-run sizing guarantee. The campaign remains in progress and readiness remains **NOT READY**.


### Live training monitor refresh — 2026-10-14 12:51 UTC

The same Slurm allocation remains RUNNING with 18 active unique seed-42 lineages. Their latest logs span 2,654,208–4,227,072 of 20,000,000 steps; no worker has logged `EARLY_ABORT`. Seven of 18 latest 20,000-bar training-prefix monitors show zero trades. This is a rising but still interim monitor count; it is not the four completed training-episode condition used by the pinned old callback, and it is not a final full-year result. The checksummed per-run snapshot is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/train_prefix_monitor_snapshot_20261014_1251.json` (SHA-256 `e91e660b09f3c168e050374b8e885eed16cfcd30b33dfebb96dd05e613a2c17e`). Final train/test evaluation artifacts remain absent. Readiness remains **NOT READY**.


### Checkpoint continuation budget audit — 2026-10-14 12:50 UTC

Source review confirms migrated checkpoints count toward, rather than add to, the 20M-step per-variant target: the trainer loads `model.num_timesteps`, rejects checkpoints above target, computes `learning_timesteps = target - loaded_steps`, and calls PPO with `reset_num_timesteps=False`. At an already-target checkpoint it skips learning but proceeds to final train/test evaluation. The controller archives and provenance-records the checkpoint before passing it through the variant launcher. Focused checkpoint-migration, campaign-artifact, and training-decomposition tests pass **37/37**. The audit is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_resume_budget_contract_20261014_1250.json` (SHA-256 `ecd1ff7f376b4f2052103039b7e097f8cddfabd29558b4db51e9d6fc8756ed0a`). This is code/test evidence; the successor has not yet resumed on GH200, so runtime continuation and final evaluations remain unverified.


### Resource sample extension — 2026-10-14 12:53 UTC

The allocation-scoped sampler now covers 3.05 hours. Per-node cgroup RAM averaged 140.0/140.4 GiB and peaked at 154.9/158.1 GiB against 512 GiB; process-tree CPU averaged 3.83/4.89 cores; GPU utilization averaged 56.3%/55.7%; GPU memory peaked at 48.7/55.7 GiB. Latest OOM counters remain zero. The derivation and raw CSV hashes are in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/resource_sample_20261014_1253.json` (SHA-256 `880911df2770f74d87c4ab145ca2df418d366f00f821a9d6376cdd80cfc7c349`). This extends the bounded resource sample; it still does not measure a complete training/evaluation cycle or establish full-run peak requirements. Overall readiness remains **NOT READY**.


### Live campaign continuation and resource snapshot — 2026-10-14 13:09 UTC

Slurm job `20954895` remains RUNNING on `gpuarm[1-2]` with 2 GH200 GPUs, 96 CPUs, 1 TiB total RAM (512 GiB/node), and a 6-hour limit. The primary campaign manifest has 16 active workers and two historical cache-start failures; the separate recovery manifest has both affected variants running. Reconciled by variant, there are **18 active unique seed-42 lineages**. Their latest training logs span 2,949,120–4,620,288 of 20,000,000 requested steps; no `EARLY_ABORT` markers were found. The 20,000-bar prefix monitors are interim diagnostics only. No final train-in-sample or report-only test report exists.

The latest allocation-sampler snapshot covers 3.32 hours: process-tree CPU averaged 3.82/4.89 cores; allocation cgroup RAM averaged 141.2/141.8 GiB and peaked at 158.7/162.6 GiB; GPU utilization averaged 56.5%/55.6%; GPU memory peaked at 50,499/58,175 MiB; sampled OOM counters were zero. These measurements show no RAM pressure and low CPU use in this interval, but are not a full-run resource guarantee. The machine-readable snapshot is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/live_campaign_snapshot_20261014_1309.json` (SHA-256 `af80c4cf4d7be47d9adb1a9caad263970559a619c9f51bf833e97f0e11075e07`); it records worker log paths/mtimes and the resource CSV hashes. The recovery controller log also contains variant-list and vector-environment/CPU guard error text even though the recovery manifest and advancing worker logs show both lineages running; retain this discrepancy for invocation-path review and do not classify those two workers as failed from those messages alone.

All workers remain pinned to source `937bae5`; current `HEAD` uses the later survival-only acceptance policy. Results remain exploratory across this policy boundary, and the previously inspected 2026 test period remains selection-acknowledged. Research review, fresh confirmatory holdout, full-run resource sizing, and passing final train/test survival reports remain open. Readiness remains **NOT READY**.


### Current checkpoint and allocation refresh — 2026-10-14 13:28 UTC

The allocation-boundary checkpoint audit completed successfully: the newest selected checkpoint for each of the 18 unique seed-42 lineages passed CPU deserialization and embedded-timestep validation (**18 passed, 0 failed**; selected checkpoints were 3M–4M steps). The machine-readable audit is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_live_20261014_1316.json` (SHA-256 `2ecff25d092504db675dcceb0ab38b08d996a81c31a63be89af471ca989fb9f7`). This validates local CPU loadability and lineage metadata, not GH200 loading, training completion, or strategy quality.

At 13:28 UTC, Slurm job `20954895` remained RUNNING (4:03 of 6:00) and the original allocation supervisor plus the migration watcher were alive; the watcher was still waiting, with no successor allocation requested. Primary worker logs span 3,145,728–5,111,808 steps; the separate cache-recovery logs span 3,932,160–4,030,464. Across the 18 lineages, no final train/test report exists. Preserve the earlier 13:09 resource sample as bounded interim evidence; readiness remains **NOT READY**.

### Live worker refresh — 2026-10-14 13:32 UTC

Read-only reconciliation found Slurm job `20954895` RUNNING at 4:06:32/6:00 on two nodes. The latest worker logs cover all 18 unique lineages, spanning 3,244,032–5,210,112 PPO timesteps. Seven of the 18 latest 20,000-bar training-prefix monitor records report zero trades; these are interim diagnostics, not final train/test evaluations or worker termination. No `EARLY_ABORT` marker was present in the inspected worker logs. No `evaluations/train_in_sample.json` or `evaluations/test_report_only.json` artifacts exist yet. The reproducible snapshot, including per-log and per-monitor hashes, is `docs/thesis/live_campaign_snapshot_2026-10-14_1332.json` (SHA-256 `05eb275738e408b2b1c47d84c54eba2e7ded9d8eb3d94be1e6bc913f0a810c91`). The campaign remains exploratory and readiness remains **NOT READY**.

### Live checkpoint and worker refresh — 2026-10-14 13:48 UTC

The latest resumable checkpoint selected for each of the 18 unique seed-42 lineages passed provenance/hash validation, CPU deserialization, and saved-timestep validation (**18 passed, 0 failed**, loaded checkpoints spanning 3M–5M timesteps). This does not establish GH200 loading, successful resume, or strategy quality. The audit is `docs/thesis/checkpoint_load_cpu_18_seed42_live_20261014_1338.json` (SHA-256 `75581ab0b06107c232e458479673d57134dfcac167884a2f080cfdd31e41f1e3`).

At the same refresh, Slurm job `20954895` remained RUNNING at 4:23:13/6:00. All 18 live worker logs span 3,440,640–5,701,632 PPO timesteps; six latest 20,000-bar training-prefix monitors show zero trades. These remain interim diagnostics. No `EARLY_ABORT` marker or final train/test evaluation artifact was found. The hashed worker-level snapshot is `docs/thesis/live_campaign_snapshot_2026-10-14_1348.json` (SHA-256 `efdf664e4d32320d601fdc41f3ac581d19fcdb1d947b9fa01e40b9428aac17d9`). The campaign remains exploratory and readiness remains **NOT READY**.

### Four-hour resource sample — 2026-10-14 13:52 UTC

The resource samplers cover 4.02 hours of active campaign measurements. With 48 CPUs and 512 GiB requested per node, process-tree CPU averaged 3.82 cores on `gpuarm1` and 4.89 on `gpuarm2`; cgroup RAM averaged 144.4/145.5 GiB and peaked at 166.5/170.0 GiB. GPU utilization averaged 57.0%/55.9%, with VRAM peaks of 51.2/60.0 GiB. Sampled OOM counters were zero on both nodes. These are interim measurements of the current 18-worker run, not full-run peak requirements or proof that increasing concurrency/env count would improve throughput. The checksummed resource CSVs and contemporaneous worker/checkpoint status are recorded in `docs/thesis/live_campaign_snapshot_2026-10-14_1352.json` (SHA-256 `fc11afa4ac1779fb1179bbac452e369fa3fbfce262a536c010483d058082f209`). Final train/test evaluations remain absent; readiness remains **NOT READY**.

### Live campaign and resource refresh — 2026-10-14 14:06 UTC

Slurm job `20954895` was RUNNING at 4:41:15 on `gpuarm[1-2]`; the migration watcher remained waiting for the original allocation, and no successor allocation had been requested. Latest saved PPO checkpoints cover all 18 unique lineages at 3M–6M of 20M steps (median 4M). No `EARLY_ABORT` markers or final train/test evaluation JSON files were found. These are partial checkpoints and interim status only.

The allocation-scoped resource sample spans 4.26 hours. Average process-tree CPU was 3.82/4.89 cores per node against 48 allocated; cgroup RAM averaged 145.6/146.8 GiB and peaked at 170.0/172.7 GiB of 512 GiB; mean GPU utilization was 57.1%/55.9%, and VRAM peaks were 51.9/61.1 GiB. Sampled OOM counters remained zero. The machine-readable report, including both raw CSV hashes and the derivation method, is `docs/thesis/live_resource_sample_2026-10-14_1406.json` (SHA-256 `c7fd5f8f46e8a35cf3dd3990b5041ba2835fd346c76890395c9c8f6b9f06d7a9`). This interval is not a full-run resource-sizing guarantee. Final evaluations, independent review, viability, and confirmatory holdout requirements remain open; readiness remains **NOT READY**.

### Interim train-prefix monitor refresh — 2026-10-14 14:12 UTC

A hashed snapshot of the latest 20,000-bar training-prefix monitor for each of the 18 unique seed-42 lineages reports seven zero-trade lineages. Across these interim monitors, trades range from 0 to 73, PnL from -$1,599.27 to +$2,650.36, and monitor timesteps from 3,801,088 to 6,201,344. The per-lineage records and source-file/record hashes are in `docs/thesis/live_train_prefix_monitor_2026-10-14_1412.json` (SHA-256 `2012a42f7e579b309f8fc55591ab2e2ea85126e4ee5b6caceb55230a180acdbd`). These are partial 20,000-bar diagnostics, not full-year train/test evaluations, survival determinations, or strategy-selection evidence; no final train/test report is present. Readiness remains **NOT READY**.

### Training-year budget and coverage clarification — 2026-10-14

A reproducible audit reconciles 351,550 raw training M1 rows with 99,504 eligible NY-session agent steps through 2025-12-31. The environment's agent-step unit is an eligible NY bar: `TradingEnv._resolve_ny_indices` builds the NY index and `_advance_ny_step` moves between those indices, so 99,504—not all 351,550 raw rows—define one full-year episode. At 20M aggregate PPO timesteps and 64 vector environments, each environment receives 312,500 nominal steps, or about 3.14 full-year episode equivalents only if episodes reach the final training bar. Risk breaches can end training episodes early, so this is budget capacity rather than proof of a completed full-year learning episode. The final deterministic train-in-sample and test-report-only evaluations must replay the complete splits; neither report is present yet. The audit records distinct raw-row and eligible-session counting methods, source/config/log hashes, and environment step semantics at `docs/thesis/train_year_coverage_audit_2026-10-14.json` (SHA-256 `74f98db62dbcc89f90fca7876c705e1d52e79706208219fef45da0e0450a3a8f`).

**Episode-coverage instrumentation:** current-source training now appends each completed episode to `episode_equity.csv` with environment index, episode step count, whether the full data split was reached, and the ending reason. Appending at episode completion preserves records if a worker later stops between checkpoints. This is report-only and does not change training or risk behavior. The active workers are pinned to `937bae5`, whose episode records do not include these fields; the measurements apply only to continuation under the instrumented source and cannot reconstruct the earlier phase from checkpoints. Do not claim per-environment full-year training exposure until the resulting episode records are reviewed alongside final train/test reports.

Focused coverage tests passed 24/24 (`tests/test_train/test_equity_gate.py` and `tests/test_envs/test_year_episode_survival.py`); full-tree Ruff format, Ruff lint, and Mypy also passed. This verifies fixture behavior and CSV persistence, not GH200 continuation runtime; no instrumented campaign artifact exists yet.

### Campaign episode-coverage reporting — 2026-10-14

The standardized campaign summary now reads `model/episode_equity.csv` for each run, hashes the artifact, and reports completed full-training-split episodes by environment. Missing, legacy-schema, unreadable, and unknown-completion telemetry remain explicit; missing telemetry is not interpreted as zero episodes or as proof that a year was not reached. The Markdown table links the per-run episode file, shows each attempt and any failure/exit detail, and the report-level counts show how many runs have compatible telemetry. Full failure text remains available in JSON and worker logs. The summary no longer labels every campaign as a 100k probe or describes every test replay as report-only; wording follows the manifest's requested budget and declared evaluation role.

`campaign_controller.py` now invokes the summary after writing its final allocation-boundary or terminal status snapshot. Summary-generation failure is logged as a warning and does not alter worker results or campaign return status. This report-only change will be present in the separately named checkpoint-migration campaign. It cannot retrofit telemetry into the currently running `937bae5` workers or reconstruct their earlier episodes from checkpoints.

Verification: `tests/test_campaign_results_summary.py` and `tests/test_campaign_artifacts.py` passed **23/23**; `uv run ruff format --check .` passed for **405 files**, `uv run ruff check .` passed, and `uv run mypy .` passed for **405 source files**. Commands used `.venv-x86` because the current shell is x86_64 and the default `.venv` is aarch64.

At 2026-10-14 15:03 UTC, Slurm job `20954895` was still RUNNING on gpuarm[1-2] at 5:38/6:00. The primary manifest showed 16 running and two cache-hash startup failures; the separate two-run cache-recovery manifest showed both replacements running. The two original logs record Idea 1 and Idea 2 sidecar mismatches before training; those failures remain preserved. These are campaign-state observations, not final evaluation evidence. No readiness gate is cleared; status remains **NOT READY**.

### Extended live resource sample — 2026-10-14 15:09 UTC

The allocation-scoped CSVs now cover about 5.32 hours. Per-node RAM averaged 150.7/152.4 GiB and peaked at 180.6/184.0 GiB of 512 GiB; primary sampler process-tree CPU averaged 3.82/4.89 cores; GPU utilization averaged 57.0%/56.4%; VRAM peaks were 54.6/65.9 GiB; sampled OOM counters remained zero. RAM and physical GPU samples are Slurm-job scope and include the two recovery workers; the CPU process-tree counter may exclude their separate supervisor. This is still a partial campaign sample, not full-run sizing. The parsed report is `docs/thesis/live_resource_sample_2026-10-14_1508.json` (SHA-256 `a015994a3f2b176626b45f0bd66e5508166a73f490f246ff2f4c35d3818b50d4`). Exact raw CSV snapshots are preserved at `outputs/gpu_campaigns/ladder18_seed42_20m_purelegacy_shared_bundle_n64_20261014/reports/resource_snapshot_20261014_1508/` and their gzip hashes are recorded in the JSON. At the same observation, Slurm job `20954895` remained RUNNING at 5:44/6:00; the migration watcher continued waiting. No final train/test report exists; readiness remains **NOT READY**.

### Checkpoint-resume vectorization failure and correction — 2026-10-14 18:58 EEST

After job `20954895` reached its time limit, the first migration supervisor requested job `20962662` (two GH200 nodes, 48 CPUs and 512 GiB per node, six hours). Its 18-way causal feature/bundle parity preflight passed 18/18. Runtime inspection then found all 18 resumed PPO models reporting effective `n_envs=1`, although the campaign and saved configs requested 64. `PPO.load(..., env=train_env)` had attached only the single raw bundle environment and bypassed the `SubprocVecEnv` construction used for fresh training. The workers were stopped through the controller's SIGTERM handler, with no `scancel`; the manifest records 18 interrupted attempts. The short attempt wrote no new numbered checkpoints, so the verified source map remains at 5M–8M timesteps.

The same job ended naturally after the login-side supervisor failed to parse status: its inherited AArch64 `.venv/bin/python3` was not executable on the x86 login host. A shared internal VecEnv builder now serves fresh and resumed PPO paths; effective-PPO logging rejects a mismatch between the model and resolved `n_envs`. A regression test saves a single-env PPO checkpoint, attaches the requested two-env vector wrapper, and verifies the loaded model reports both environments. The new supervisor runs its manifest parser using `/usr/bin` system Python. A new campaign ID/spec prevents these changes from being mixed with the invalid attempt.

Evidence: `tests/test_models/test_agent_vector_env.py` passed (1/1); full-tree Ruff format, Ruff lint, and Mypy passed (406 files). New source/config files are prepared for `ladder18_seed42_20m_policy_migration_nenv64_20261014`; training has not yet resumed. No final train/test evaluation exists and readiness remains **NOT READY**.


### Live vector-resume campaign and continuity watcher — 2026-10-14 16:57 UTC

Recorded the current 18-worker campaign state in [`live_campaign_snapshot_2026-10-14_1656.json`](live_campaign_snapshot_2026-10-14_1656.json), SHA-256 `a5f98d2c20835caca79d58fa1398d7b90c7027f7f0f7f6ac0aa6c998446569e4`. Job `20963043` remains active with all 18 `n_envs=64` workers; no new numbered checkpoint or final evaluation artifact exists yet. Latest resource and disk-quota evidence, current report hashes, and the per-worker logs are included in the snapshot.

Started one `thesis-organizer:campaign-chain` tmux window. It waits for the current job to terminate and for two idle GH200 nodes with the authorized memory shape before calling the existing vector-resume supervisor. It does not cancel or duplicate the active allocation. This continuity setup does not resolve open research review, confirmatory holdout, full-year episode, survival, or viability gates; readiness remains **NOT READY**. No code/config files or campaign source inputs were changed, and no commit was created.

### Live episode-telemetry refresh — 2026-10-14 17:06 UTC

Slurm job `20963043` was `RUNNING` at 1:02:28 / 6:00:00. Its manifest listed 18/18 workers running and all 18 worker logs had been updated within five minutes. A first numbered continuation checkpoint, `ppo_ckpt_9000000_steps.zip`, is present for `ladder_a0_flat__seed42`. No final `metrics.json` or `summary.md` artifacts were present.

The current-source episode telemetry available at this observation is limited to `ladder_a1_sl__seed42`: nine completed environment episodes, 7,338–13,630 steps each, zero full-training-split completions, and all nine ended on `trailing_dd` risk breach. Each episode recorded 89–169 trades. This is an interim training signal for one variant, not a final train-split evaluation or a verdict on the completed policy. The CSV SHA-256 is `b3f063960d883d50f74a5de8ab4fb1a79a7cd17d388db213693479067c570386`; checkpoint lineage starts from the 6M-step checkpoint in the prior `937bae5` campaign. The machine-readable observation is `docs/thesis/live_campaign_snapshot_2026-10-14_1706.json` (SHA-256 `589278652a60337dabe1c5b34400d15310a3806f2dc460cd44345fb9b26a92a4`).

The active run and its checkpoint lineage remain exploratory; no final viability, full-year episode, fresh-holdout, independent-review, or production-readiness claim follows from this partial telemetry. Overall readiness remains **NOT READY**.

### Current-source episode-telemetry continuation — 2026-10-14 17:07 UTC

A later immutable snapshot at `docs/thesis/live_campaign_snapshot_2026-10-14_1707.json` (SHA-256 `eb757f5256841f2cc85b14b6bc5775acd4d0f9c1c1d6269b99f2de4185b60500`) preserves the episode CSVs as of 17:07 UTC. It extends the earlier 17:06 observation: `ladder_a1_sl__seed42` had nine early episodes and `ladder_a1_sl_po3__seed42` had one. All ten episodes ended on trailing-drawdown risk breach before completing the train split. The Idea 1 variant episode lasted 14,731 steps and recorded 69 trades. These are interim current-source training events from two of 18 variants, not final evaluation results. The campaign remains active and final metrics are absent; readiness remains **NOT READY**.

### Episode-telemetry refresh — 2026-10-14 17:09 UTC

An immutable snapshot (`docs/thesis/live_campaign_snapshot_2026-10-14_1709.json`, SHA-256 `11132512e5f0e3a5048e0b53fee94552a205794a9d3898f016a631ac99c8cfa8`) preserves the current active-worker observations and copies of both available episode CSVs. `ladder_a1_sl__seed42` now has 10 episodes and `ladder_a1_sl_po3__seed42` has one; all 11 ended on trailing-drawdown breach before completing the training split. The second episode file therefore adds another early-breach observation since 17:07. All 18 workers were still running with fresh logs, one numbered post-resume checkpoint existed for `ladder_a0_flat`, and final metric/summary artifacts were still absent. This is interim telemetry, not a final train/test verdict; readiness remains **NOT READY**.

### Live 18-worker resource and episode refresh — 2026-10-14 17:11 UTC

The 50.5-minute allocation-scoped sample averages 124.12/122.70 GiB cgroup RAM per node, with recorded peaks of 132.14/132.80 GiB out of 512 GiB; process-tree CPU averages 4.939/4.904 cores; GPU utilization averages 55.2%/54.9% and peaks at 98%; GPU-memory peaks are 60,951/101,645 MiB; sampled OOM events are zero. This remains partial live-run evidence, not final sizing. The machine-readable report is `docs/thesis/live_resource_snapshot_2026-10-14_1710.json` (SHA-256 `b15fa89bce224ac65e063100835ace24fc22d465a2ca6a7007993befcccbc31c)` with immutable raw CSV copies and hashes.

The 17:11 campaign snapshot (`docs/thesis/live_campaign_snapshot_2026-10-14_1711.json`, SHA-256 `50b2558b0c6e27e01107d00ab9245afae3c696f4a81b6acb809c2f8d84ee8bad`) records 18/18 workers active and fresh logs; three numbered continuation checkpoints at 7M–9M; twelve episode records across three variants, all ending in trailing-drawdown breach before completing the train split; and no final metrics or summaries. These data are exploratory progress signals, not final results. Readiness remains **NOT READY**.

### Active 18-run campaign evidence — 2026-10-14 17:13 UTC

The immutable campaign snapshot [`live_campaign_snapshot_2026-10-14_1713.json`](live_campaign_snapshot_2026-10-14_1713.json), SHA-256 `c9df47a35c5ed838f31969385c9b49b723e3bf99a623ee614ddaa48dd2a3ef12`, records job `20963043` running at 1:09:45/6:00:00, 18/18 workers active, all worker logs fresh, and eight numbered continuation checkpoints. Current dashboard steps range from 5.70M to 9.18M of 20M total; models resumed from 5M–8M checkpoints. Mean added progress is about 0.90M per worker over 54 minutes (about 276 steps/s); a linear projection leaves about 13.4 hours to the 20M target and is not a runtime guarantee. The continuation supervisor is configured to request successive allocations when needed.

The snapshot preserves 14 completed environment episodes across three variants; all ended in trailing-drawdown risk breach, and none completed the full training split. No final metrics or final summaries exist. These are interim training alarms, not the final deterministic train/test evaluations or evidence that the finished policies fail or succeed.

The paired partial resource report [`live_resource_snapshot_2026-10-14_1713.json`](live_resource_snapshot_2026-10-14_1713.json), SHA-256 `7c81adfb211cd90c4f6ed0f7b00800708910d02ad5d8dd56c68c6cf5aab803fc`, covers about 53.6 minutes: mean job-cgroup RAM was 123–124 GiB per 512-GiB node, recorded peaks 132–134 GiB; process-tree CPU averaged about 4.9 cores/node while 45 worker CPUs were requested; GPU utilization averaged 55–56% and peaked at 98–99%; GPU-memory peaks were 61,065 and 101,645 MiB; sampled OOM counters were zero. This is partial resource sizing, not a full-run profile. Readiness remains **NOT READY**.

### Requested CI checks and live campaign refresh — 2026-10-14 17:16 UTC

The default `.venv` could not run on this x86 login host (`Exec format error`); rerunning through the repository's `.venv-x86` succeeded: `ruff format --check .` reported 406 files formatted; `ruff check .` passed; `mypy .` reported no issues in 406 source files; and focused campaign/report/resource tests (`tests/test_triton_resource_sampler.py`, `tests/test_campaign_results_summary.py`, `tests/test_campaign_artifacts.py`) passed 31/31. `git diff --check` passed. No training-source code changed.

The newest point-in-time snapshot is `docs/thesis/live_campaign_snapshot_2026-10-14_1715.json` (SHA-256 `d3d74060630193779b6302dab79f7d4388d751185e2d1fd2b8990dfa4d58eb12`). Job `20963043` was still RUNNING at 1:11:51/6:00:00 with all 18 workers active and fresh logs. Ten numbered checkpoints existed. Fifteen recorded episodes across `ladder_a0_flat_dist`, `ladder_a1_sl`, and `ladder_a1_sl_po3` had all ended on trailing-drawdown breach before the full train split; no final metrics or summary existed. This remains interim evidence, not the final policy decision.

Full-tree format/lint/type and scoped tests are complete for the current docs/evidence work. The requested commit/push remains deferred until the campaign and continuation watcher are terminal: the resume fingerprint includes both Git SHA and source-file hashes, so a docs-only commit would change the fingerprint even though worker source hashes still match. Only docs/report artifacts are modified. Do not claim readiness: independent review remains unsigned, the campaign is incomplete, and exposed data remains exploratory. Overall readiness remains **NOT READY**.

### Live campaign refresh — 2026-10-14 17:19 UTC

The point-in-time [campaign snapshot](live_campaign_snapshot_2026-10-14_1719.json), SHA-256 `d0a3fa93bf2382fdd3f4a8f25f5b114523112bf425feb6fb5b3ab1558db3cfee`, records job `20963043` running at 1:15:42/6:00:00, all 18 workers active with fresh logs, 13 numbered checkpoints, and latest dashboard steps from 5.80M to 9.30M of the 20M target. No final metrics or summaries exist.

Nineteen completed environment episodes are currently recorded across three variants; all nineteen ended on trailing-drawdown breach before completing the training split. This is an interim survival signal, not a final evaluation. The paired [resource snapshot](live_resource_snapshot_2026-10-14_1719.json), SHA-256 `a2ddc3fddc59a9c9f39d73074f9c9d4df2509d94a1e2524937a2a888320615cd`, covers about 60 minutes: cgroup RAM averages 123.9–124.8 GiB/node and peaks at 132–134 GiB of 512 GiB; process-tree CPU averages 4.90–4.94 cores/node; GPU utilization averages 55–57% and peaks at 98–99%; VRAM peaks at 61,315 and 101,645 MiB; sampled OOM counts remain zero. This is still a partial run profile. The source fingerprint remains unchanged, and the run is left active under the existing continuation watcher. Readiness remains **NOT READY**.

### Episode-outcome interpretation — 2026-10-14 17:20 UTC

Using the immutable episode CSV copies listed in `docs/thesis/live_campaign_snapshot_2026-10-14_1719.json` (snapshot SHA-256 `d0a3fa93bf2382fdd3f4a8f25f5b114523112bf425feb6fb5b3ab1558db3cfee`), I checked the actual end reasons and account deltas. All 19 recorded episodes across `ladder_a0_flat_dist`, `ladder_a1_sl`, and `ladder_a1_sl_po3` ended specifically on `trailing_dd`; none reached the 99,504-step full training split. Variant means were: A0 distribution, 14,611 steps / 61.7 trades / −$6,765 equity delta (3 episodes); A1 stop-loss, 13,059 steps / 157.5 trades / −$5,977 (15 episodes); A1 stop-loss plus PO3, 14,731 steps / 69 trades / −$6,732 (one episode). The longest recorded episode covered only 14.8–17.6% of the complete train split, depending on variant.

This rules out “no activity” as the explanation for these specific episodes: they traded, then hit a trailing-drawdown terminal. It is serious interim negative survival evidence, but the agents are still training and these are not final deterministic train/test evaluations. Keep the run’s eventual gates and complete-split reports as the decision evidence; do not label these partial episodes a final campaign failure or a successful strategy.

### Trailing-drawdown terminal adjudication — 2026-10-14 17:23 UTC

A source/config check confirms the observed episode endings are consistent with the configured training guardrail, not evidence of an accidental episode-length cap. All 18 current effective `config.yaml` files set `ftmo.trailing_dd_limit=0.07` and `env.max_episode_steps=null`. For example, A1-SL's effective config is `outputs/gpu_campaigns/ladder18_seed42_20m_policy_migration_nenv64_20261014/runs/attempts/ladder_a1_sl__seed42/attempt1/ladder_a1_sl__seed42/20261014_191951_745461_rl_train_seed42_tcn/config.yaml` (SHA-256 `c36f0f45246c5b417a710c8647fb69d3c99ede8878c9590fd13f5e78a14c0b2b`). `FTMOGuardrails.check_trailing_dd()` ends PPO training episodes when the peak-to-current equity drop reaches 7% of initial balance ($7,000 on $100,000); `TradingEnv._check_guardrails()` terminates episodic training on that breach. Non-episodic evaluation latches the year failure and continues the rollout. The focused `test_trailing_dd_terminates_year_episode` regression reports a pass.

Therefore the episodes' end equity delta from starting balance must not be compared directly to the $7,000 peak-to-trough threshold: prior gains can make peak equity exceed starting balance. The repeated breaches are genuine training-sample survival failures under the configured gate, not a code defect established by current evidence. They remain interim while the policies train; require the eventual full-split train/test reports before making a final policy decision.

### Continuation review update — 2026-10-14 17:29 UTC

The live campaign remains job `20963043` under runtime source `c2261ca`, resuming checkpoint lineages from prior source revisions including `937bae5`. All 18 worker logs were fresh at the check, with 5.88M–9.57M PPO steps of 20M. The current policy-migration configs disable no-trade and interim equity-gate early aborts; final train survival still fails on max-loss breach or peak trailing drawdown above 10%.

Twenty-one current-source episodes across three variants all ended on `trailing_dd` at 7,338–19,575 steps, with no full training-split completion. No final train-in-sample or report-only test reports exist. A 1.14-hour resource sample averages 125.5/125.0 GiB Slurm-cgroup RAM, 4.93/4.90 process-tree CPU cores, and 56.9%/55.5% GPU utilization on the two nodes, with no sampled OOM events. This remains interim exploratory evidence, not a final campaign decision or a full resource-sizing profile. The readiness plan and independent-review checklist have been refreshed; overall status remains **NOT READY**.

### Training-year reset-semantics audit — 2026-10-14 17:34 UTC

The point-in-time audit `docs/thesis/train_episode_reset_coverage_audit_2026-10-14.json` (SHA-256 `8f0e564510cd0cbf481bcc3576e392b603eb6e35a85d5939b6d7910c66f177f9`) checked all 18 effective configs, active campaign/checkpoint lineage, environment and vector-reset source, and episode records. It captured 22 completed episodes across three variants; all ended on trailing-drawdown breach after 7,338–21,414 of 99,504 NY agent steps. Training reset returns to the first eligible NY bar, so these terminated episodes restart at the beginning rather than a later date. This proves repeated-prefix exposure for the captured episodes, not that every vector environment terminated or that no other environment reached later dates. The active run remains exploratory and does not yet demonstrate the requested full-year training exposure; final full-split evaluations are separate evidence. Overall readiness remains **NOT READY**.

### Full-split episode fix and live status — 2026-10-14 17:45 UTC

The isolated branch `codex/full-year-episode-coverage`, commit `5f17bd5`, adds `training.episode_mode=full_split` to both frame-backed and bundle-backed PPO training, rejects artificial episode caps, and keeps `risk_terminated` for explicit legacy runs. Risk breaches remain recorded; the environment follows its non-episodic guardrail behavior and continues to the training split boundary. The focused suite passes 28/28; full-tree Ruff format/lint and Mypy pass on 406 files. The branch was pushed, but remains unmerged and is not loaded by the active campaign.

Job `20963043` remains active at 17:45 UTC with 18 fresh worker logs and progress at 6.18M–9.87M/20M. The 17:45 reset-semantics audit `docs/thesis/train_episode_reset_coverage_audit_2026-10-14_1745.json` (SHA-256 `aa3875d6074b84eb9be7a0b8cf8ba881b6e8d5aa840a5066a9f4a886597979e6`) captures 28 completed episodes, all trailing-drawdown breaches at 7,338–25,113 steps, and zero full-split completions. Final train/test reports remain absent; the current campaign does not yet establish full-year training exposure. Readiness remains **NOT READY**.


### Current pure-legacy A0 contract audit — 2026-10-14 17:51 UTC

Added [`a0_current_action_feature_audit_2026-10-14.json`](a0_current_action_feature_audit_2026-10-14.json), SHA-256 `2637a60b14496d12f7f29802d16043addfa8834da696e9b794b2ec6e9bdf7887`. The active 20M A0 attempt resolves to the authorized legacy action and default feature profile, loads the prebuilt baseline cache, and resumes a checkpoint whose lineage is the prior pure-legacy campaign. Source gates confirm the PO3 manipulation side-lock and strategy-action open blackout do not apply with `strategy_actions=false`. This closes the planned A0 config/action audit. A separate bounded corrected-A0 frames/shared-bundle parity trace has since passed; all campaign outcome gates remain pending.


### Active 18-worker allocation resource sample — 2026-10-14 17:55 UTC

Added [`live_resource_snapshot_2026-10-14_1755.json`](live_resource_snapshot_2026-10-14_1755.json), SHA-256 `0dd5a117a746397072b30e78ee6bedcd9a2cc9289bcf841d9c4d8c76c54aace4`, with immutable raw sampler CSVs. Over about 1.58 hours, per-node cgroup RAM averaged about 128 GiB and peaked at 138.6/140.2 GiB of 512 GiB; GPU utilization averaged 58.5%/57.1% and VRAM peaked at 61.3/99.3 GiB. Slurm step AveCPU was about 4.9 core-equivalents per node across nine trainers requesting 45 worker CPUs. This is interim evidence only; it does not determine the full-run peak or completed throughput.


### Corrected A0 backend parity — 2026-10-14 21:03 Helsinki time

The A0-only current-source frames/shared-bundle audit passed: maximum scalar difference was zero over a 49,925-step test-calendar trace and a 10,000-step active-action trace, which opened 46 trades. The report `outputs/gpu_campaigns/readiness_reaudit_20261014/a0_current_parity/a0_frames_bundle_parity.json` is SHA-256 `6d40de0e4429ed2db0ce7016b0583d8362374f40a373834cea5df793c0921f86`; the retained driver is SHA-256 `81897d053d5ef7ee4190c8bb9e7e4421d6842dc63267d413f66e0b24527c094e`. This is bounded backend parity evidence only; it does not establish viability, full-year learning, independent sign-off, or all-variant current-source parity.


### Live campaign and fingerprint check — 2026-10-14 19:39 UTC

Job `20963043` is RUNNING at 3:35:47/6:00:00 on two nodes, one GH200/48 CPUs/512 GiB per node. All 18 worker logs were fresh; PPO progress was 7.85M–12.52M/20M (median 9.74M). The slowest latest rate was 248 FPS, implying about 13.61 hours remaining if sustained; no final train/test reports exist. The continuation watcher is still waiting for the current job, then two idle GH200 nodes, before resuming the same campaign; no new allocation is queued.

The active run remains on source `c2261ca`; full-split training mode commit `5f17bd5` is isolated and not loaded. The last episode audit recorded 28 episodes across three variants, all ending at trailing drawdown before the 99,504-step train-split end. This does not establish full-year exposure for the active policy. A source review of `campaign_controller.py` confirms its resume fingerprint contains both `git_sha` and source-file hashes; committing the pending docs/evidence changes would change the Git SHA and invalidate automatic resume despite unchanged code hashes. Commit/push is therefore deferred until the campaign and watcher are terminal. Readiness remains **NOT READY**.


### Throughput scaling evidence — 2026-10-14 19:40 UTC

The prior three-trainer shared-bundle GH200 run at `n_envs=64` logged 1,256–1,455 FPS/worker (sum 4,017 steps/s) with 32 CPUs requested per worker and two GPUs shared by three workers. The current 18-trainer shared-bundle run at the same `n_envs` logs 247–385 FPS/worker (median 319.5; sum 5,486 steps/s), with five CPUs/worker and nine workers/GPU. Thus per-worker throughput dropped about 4x while aggregate steps/s increased about 37%. The comparison is not controlled: source revisions differ, and A0 `strategy_actions` differs between the two configs. `agent.py` creates a separate SB3 `SubprocVecEnv` worker for each of the 64 environments; current resource samples counted about 615 processes per node. The sixfold increase in trainers/GPU and the smaller per-trainer CPU allocation plausibly increase worker scheduling and rollout synchronization. Early resource samples show low aggregate CPU use, ample RAM headroom, and GPU utilization averages below 60% with high peaks; there is no evidence of aggregate CPU/RAM exhaustion or sustained GPU saturation, and no CPU throttling counters were captured. Details and limits are in the readiness plan's “Three-trainer versus 18-trainer throughput comparison” section.

### Live resource refresh — 2026-10-14 19:49 UTC

The immutable [`live resource sample`](live_resource_snapshot_2026-10-14_1949.json), SHA-256 `3143b15ff663449fa8db1fd026fa4636500e50846f9f010ed7e3440984a756ba`, extends the active job `20963043` sampler through 19:48 UTC (about 3.49 hours; sampler began 15 minutes after allocation start). Per-node average cgroup RAM was 137.88/138.77 GiB and maximum sampled current use was 158.40/159.94 GiB of 512 GiB; cgroup-reported peaks were 159.99/161.90 GiB. Process-tree CPU averaged 4.92/4.89 cores, GPU utilization averaged 58.7%/59.0% with sampled peaks of 99%/100%, and VRAM peaks were 65.87/99.26 GiB. Sampled OOM-kill counters remained zero. The report preserves raw sampler CSV copies and SHA-256 hashes. This strengthens the evidence against RAM exhaustion and system-wide CPU saturation during this interval, but CPU pressure/throttling counters were not collected and GPU peaks do not establish sustained saturation. It does not isolate the cause of per-trainer FPS reduction, establish full-run resource peaks, or turn the old-versus-new comparison into a controlled benchmark. Job `20963043` remains active; the 18 workers were at 8.05M–12.72M/20M in the latest logs, and final train/test reports remain absent. Readiness remains **NOT READY**.

### Live episode-coverage and campaign-status audit — 2026-10-14 19:50 UTC

The point-in-time [`live campaign snapshot`](live_campaign_snapshot_2026-10-14_1950.json), SHA-256 `1f38c5eb626741a3221c96ebe879e147dc8369e1976c2b0eb34e6db3da9110af`, captures Slurm job `20963043` RUNNING at 3:46:37/6:00:00, 18 fresh worker logs, and latest PPO steps from 8.05M to 12.82M of 20M (median 9.98M). It archives and hashes all eight available current-source `episode_equity.csv` files. Those contain 225 completed episodes across eight variants; all 225 ended on trailing-drawdown breach after 7,338–71,857 steps, and none completed the 99,504-step training split. Ten variants had no episode telemetry file at this capture; absence is not counted as zero episodes. No final `train_in_sample.json` or `test_report_only.json` artifact exists. The campaign's `reports/campaign_status.json` was last captured at 16:19:40 UTC and is stale relative to current workers; the generated 22:50 summary was marked superseded and must not be treated as current. These are serious interim survival/coverage signals, not final deterministic policy verdicts. The full-year episode-mode fix remains isolated and is not loaded into this run. Readiness remains **NOT READY**.

### Full-split correction and fresh 18-run campaign — 2026-10-15

The later 20M checkpoints cannot satisfy the requested one-year training
exposure: the episode audit found zero environments that completed the full
training split because risk breaches reset episodes early. They are therefore
not reused as initial weights. The default training episode mode now traverses
the full chronological split, and the runner rejects a non-null
`env.max_episode_steps` in this mode. The new campaign specification starts
all 18 seed-42 variants from scratch at 20M steps each, `n_envs=64`, with the
shared-bundle backend. Its controller dry-run passed with 18 jobs and 360M
total requested timesteps; allocation sizing is nine workers per node at five
CPUs each, matching the 48-CPU guard and the bundle memory guard.

The old finalization snapshot remains at
`outputs/gpu_campaigns/ladder18_seed42_20m_finalization_20261015/reports/`:
17 artifact admissions passed, and A0 was rejected because its serialized
Gymnasium `Discrete.n` is `numpy.int64`. The verifier now handles
`numbers.Integral`; the report is not rewritten. All old test replays were
low-activity diagnostics, not evidence of a viable strategy.

On 2026-10-15 the full suite passed **1,465 tests** (38 warnings), full-tree
Ruff formatting and lint passed, and full-tree Mypy passed on 407 source files.
The campaign and its allocation-resume supervisor are defined in the new
full-split YAML and scripts under `config/campaigns/` and `scripts/matrix/`.
The two-node GH200 request `20973488` is pending with `StartTime=Unknown`
because `gpuarm2` is down after a reboot timeout; `gpuarm1` is idle. No
full-split training has started yet. Keep the queued request and existing
artifacts intact, then start and resume the fresh campaign when both nodes are
available. Thesis readiness remains **NOT READY**: no new training result,
independent review, or valid confirmatory holdout exists.

### Trainer speed optimization plan

[`trainer_speed_optimization_plan.md`](../training/trainer_speed_optimization_plan.md) records a staged, measurement-first plan for phase profiling, trainer/GPU concurrency, `n_envs`, callback overhead, and PPO update settings. Current samples do not prove a single bottleneck; source/config changes and benchmarking are deferred until the live campaign and resume watcher are terminal so the campaign fingerprint remains valid.

### Campaign recovery and live allocation — 2026-10-16 12:04 Helsinki

The current 18-variant seed-50 campaign is
`ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016`, fingerprint
`3474621efd6a1481bd2a20bce0cfd2c0b69969df1849c6cd738d19dc5bb4ba34`.
Attempt 4 failed for all 18 identities before PPO training because the shared
bar-cache path was a self-referential symlink. The failure is classified and
retained in `reports/attempt4_infrastructure_reconciliation.json`; worker logs
and provenance are preserved. The cache path is now a real directory with 14
nonempty configured bar caches. The recovery record in
`reports/cache_recovery.json` preserves each attempt-4 log hash and the hash of
its attempt-3 resume checkpoint. The manifest's 18 resume pointers now refer to
attempt 3 (11M–14M steps), with pre-repair manifest and report backups retained.
The current manifest remains 18/18 `interrupted`; attempt 5 has not begun.

Slurm job `20985591` is RUNNING on `gpuarm1` and `gpuarm2`, requesting one
GH200, 48 CPUs and 512 GiB RAM per node, with a six-hour wall time ending
17:32 Helsinki. The live Slurm topology contains only allocation step `.0` and
`.extern`; no worker steps exist. The controller's start marker is absent. This
allocation predates the persistent per-node dispatcher code now in the working
tree, so it remains held rather than starting the fingerprinted controller
with its old worker-step fan-out. The guarded supervisor is still live in tmux
session `thesis-seed50`; its next allocation goes through the updated wrapper.

The wrapper, dispatcher, recovery validation, tests, and evidence-index builder
are uncommitted on canonical `main`. They are outside the campaign's 228 hashed
source paths; those source hashes were verified without mismatches before this
work. At 12:11 Helsinki, gpuarm1 completed full-tree Ruff format/check,
full-tree Mypy (436 source files), and 44 focused tests. It also generated
`reports/thesis_evidence_index_prelaunch_20985591.json`; the index verifies all
228 recorded campaign source hashes and all four recorded market-data input
hashes, and explicitly says full chronological split exposure and checkpoint
deserialization are not established. This verifies bounded tooling and
inventory only, not campaign readiness. The active allocation's preflight
process predates the dispatcher and remains held without a training marker; do
not release its older worker-step fan-out. The current captured state is
[`live_campaign_snapshot_2026-10-16_1158.json`](live_campaign_snapshot_2026-10-16_1158.json).

P1 remains open until all 18 runs resume, reach terminal dispositions, and
their chronological exposure is reconciled. P2's source implementation is
present, but deterministic recovery/evaluation using a real campaign
checkpoint remains unverified. No strategy result or readiness claim is
supported by these infrastructure failures or partial checkpoints.

### P2 replay audit and pushed recovery tooling — 2026-10-16 12:39 Helsinki

Commit `2211712` (`Add guarded campaign recovery and evidence indexing`) is
pushed to `origin/main`. Its gpuarm1 gate passed at 12:11: full-tree Ruff
format/check, full-tree Mypy over 436 source files, and 44 focused tests. A
prelaunch evidence index was generated at
`outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016/reports/thesis_evidence_index_prelaunch_20985591.json`;
it verifies all 228 campaign source hashes and four recorded input-data hashes.

For P2, the interrupted 14M-step `ladder_a0_flat__seed50` attempt-3 checkpoint
(`a5f4b47f8b1e523bde501c71a56f1f3e9ccae62674951fc83fd38e89fb422530`) was
loaded and replayed twice on gpuarm1 using the saved config, deterministic
actions, and both chronological splits. Each pass covered 351,550 training
bars and 177,210 test bars. The training result was Sharpe -3.8747, return
-7.3197%, 231 trades; the historical test result was Sharpe -0.9420, return
-0.6891%, 107 trades. Equity CSVs and metric JSONs were byte-identical between
passes. All other trade columns were identical, but `level_type` differed for
53 training and 23 test rows (`london_low` versus `asian_low`). The audit
`reports/p2_replay_determinism_audit_20261016.json` traces the likely cause to
`TradingEnv._matched_entry_level` taking the maximum timestamp from a set with
no deterministic tie-break. A 12-process characterization with tied
`london_low`/`asian_low` crossing times and different `PYTHONHASHSEED` values
returned each label for six seeds, confirming the source-level tie-break
defect. Since `level_type` drives temporal subgroup reports, full
replay/report determinism is not proven. These descriptive
results use a partial checkpoint and a previously inspected,
selection-acknowledged test split; they are not confirmatory strategy evidence.

P2 remains open: fix the tie-break only after the active campaign boundary,
then rerun the exact replay and action-dependent frame/shared-bundle parity
checks. Job `20985591` remains held without trainer steps; it predates the
dispatcher and will not be released into its old worker-step fan-out. The
campaign remains 18/18 interrupted and thesis readiness remains NOT READY.

### P2 real-checkpoint backend parity — 2026-10-16 13:00 Helsinki

Added `scripts/thesis/audit_checkpoint_frame_bundle_parity.py` and focused
comparison tests outside the campaign's source-hash manifest. Full-tree
gpuarm1 Ruff/Mypy and 46 focused tests passed at 12:50. The audit loaded the
same 14M-step attempt-3 checkpoint and ran its deterministic test policy for
177,210 bars through the frame evaluator and a new shared test-split bundle.
All 20 compared fields matched exactly: equity, trade rows, account/breaches,
actions, rewards, counters and aggregate metrics; each backend produced 107
trades. Full details and source lineage are in
`outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016/reports/p2_checkpoint_frame_bundle_a0_14m_20261016/audit.json`.
This closes backend parity only for this partial baseline checkpoint. It does
not resolve the separate cross-process `level_type` tie-break, prove report
parity for all variants, or supply confirmatory performance evidence.

P2 remains open for a post-campaign deterministic-label fix and repeated
checkpoint evaluation, generic management-parameter forwarding tests, and
parity against a completed selected checkpoint. Job `20985591` is still
running with no campaign trainers; the 18-run manifest remains interrupted.
