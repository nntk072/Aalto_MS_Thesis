# Entry-State Observability in Intraday Reinforcement-Learning Trading

**Working manuscript draft — 21 October 2026**

**Status:** methods draft with training-only development results. The corrective 18-variant iteration is terminal. A fresh seed-50 viability run for the frozen selected pair is active; its outcomes are not yet available. No paired-seed result or performance claim is available.

## Abstract

*To be written after the selected pair passes the viability gate and the paired-seed evaluation is complete.* The final abstract must state the question, data period, matched intervention, number of completed paired runs, principal estimate with uncertainty, cost-stress result, and limitations. It must not describe the exposed 2026 interval as a pristine holdout.

## 1. Introduction and research question

Intraday reinforcement-learning experiments require a policy to act on a time-ordered market state while trades pass through execution and risk rules that affect their outcomes. In this study, a trade opportunity also follows a causal lifecycle: a setup becomes a candidate, may become armed, and can then be entered. The primary question is whether exposing that lifecycle state changes the policy's net simulated return; entry activity and lifecycle counts describe how its behavior differs under the specified simulator.

The planned comparison holds the PPO algorithm, TCN encoder, reward and risk configuration, market inputs, lifecycle logic, requested training budget, and evaluation procedure fixed. Its declared configuration difference is `env.entry_state_observation`: the visible-state control includes four one-hot coordinates in the order `FLAT`, `CANDIDATE`, `ARMED`, `IN_POSITION`; the hidden-state treatment omits them. The pair's account observation therefore has widths 10 and 6, respectively. The estimand is limited to this observation change under the saved configurations and simulation protocol. The study does not compare algorithms or establish general profitability. Prior work motivates the method components but is not evidence about this configuration’s performance ([related work](related_work.md)).

The pair remains provisional. Earlier matched seed-50 runs completed their requested budgets but produced no trades in either arm. A common baseline/FSM defect was identified and repaired. The resulting P4 corrective iteration is terminal: 15 variants completed and three failed the unchanged training risk-survival gate. It was a warm-start continuation, and one nominally successful variant only re-evaluated an inherited 20-million-step checkpoint without new PPO updates. The P4 continuation used an explicit fingerprint override; those outcomes are exploratory. The selected pair's first post-repair continuation showed lifecycle activity but zero training reward in the hidden-state arm, so it did not pass the P6 viability gate. A new matched seed-50 pilot from scratch is now running. Failure of the fresh pilot will be reported as an implementation or viability failure, not interpreted as evidence for or against state visibility.

The primary outcome is total net return over the configured 2026 report interval. The interval has been inspected previously, so this paired estimate is exploratory and cannot provide fresh confirmatory out-of-sample evidence. This draft makes no efficacy claim and does not ask whether the PO3/IFVG rules are profitable generally or whether simulated fills establish live-market profitability.

## 2. Data and temporal protocol

The training interval is 30 December 2024 through 31 December 2025. The report-only interval is 1 January through 16 October 2026; the supplied M1 data contain no bars on 1 January, so the first test-session observations begin on 2 January. The 2026 interval has been inspected previously and is classified as exposed exploratory evidence. It is not a fresh or confirmatory holdout and is not used to select variants, change the pair, or tune the method.

The selected-pair runs use the US100 cash-index feed. The active split contains 351,432 M1 training bars and 79,069 M1 report bars. The configured date boundary and the first supplied report bar differ because no bar exists on 1 January.

| Partition | Configured date window | M1 rows | Role |
|---|---|---:|---|
| Training | 2024-12-30–2025-12-31 | 351,432 | Policy learning and training-only viability checks |
| Report | 2026-01-01–2026-10-16 | 79,069 | Exposed, report-only evaluation; no model or pair selection |

All supplied observations remain in their chronological partition, including bars from short sessions. The session schedule is a date-level input known before the first observation of each date. The policy is not expected to infer a scheduled early close from that day’s price path. Training and testing use separate inputs and caches.

M1 is the only policy timestep. Supplied M3, M5, M10, M15, M30, H1, H4, and D1 bars are retained for multi-timeframe feature construction and aligned to the M1 decision index only after their source bars close; they do not create additional policy decision timelines. The date-level session schedule marks a known early close from the first bar of that date. Bid/ask tick data drive simulated execution. The declared fill protocol draws a seeded arrival delay from the configured 0–30 millisecond interval and uses the first eligible bid/ask tick at or after order arrival. This is a reproducible simulator assumption; it does not model queue position, market impact, or all broker execution effects.

## 3. Policy and intervention

The candidate policies use PPO with a TCN encoder and the pair-specific saved effective configurations. Each run requests 20 million environment steps. The corrective screen uses seed 50 for one run per each of 18 variants; it is a repair and viability iteration, not the final paired-seed analysis. If the selected pair passes P6, the predeclared paired-evaluation plan uses matched seeds 101–105, with identical seed assignment in both arms. Earlier attempts using two of these seed IDs were interrupted under the defective implementation and are ineligible; the final paired campaign must start from scratch under a new fingerprint and must not reuse their checkpoints.

The entry-state machine represents a causal opportunity lifecycle: a qualifying setup creates a candidate, subsequent evidence may arm it, and an eligible policy action can trigger entry. The intervention changes whether the policy observation exposes the machine state. It does not change the market data, entry evidence, execution protocol, or risk rules. The effective configurations and run manifests must verify this matched design before paired-seed training begins.

The development screen trains each of the 18 registered variants once with seed 50 and a 20-million-step budget. It is a repair and viability screen, not the selected-pair estimate. The P4 corrective iteration is terminal and is classified as exploratory because it resumed warm-start checkpoints under an explicit fingerprint override. The current P6 pilot starts both frozen pair arms from scratch with seed 50 and a 20-million-step budget on the 2025 training partition; no test-split outcome is used for its viability decision. The paired evaluation, if the P6 regression and training-only viability gates pass, uses five matched training seeds (101–105) and the same requested budget in each arm. Exact PPO hyperparameters, runtime libraries, environment settings, and data identities are reported from each run's saved effective configuration and manifest rather than inferred from defaults.

The terminal P4 attempt warm-started all 18 variants from checkpoints in the predecessor corrective campaign. The checkpoint map records six initializations at 9 million steps, four at 10 million, seven at 11 million, and one at 13 million. The two selected FSM arms each started at 9 million steps. The selected checkpoints were produced under source revision `b3fe72dca3e17f85fb22c746d5e9efeb70587d90`; the continuation used reorganized source revision `ce714385a1866e95d0ecf1ce97a98f49b2c0d5c1`. These were warm-start continuations, not fresh 20-million-step runs under the current source. The continuation retained its canonical campaign fingerprint and recorded the override; its evidence is exploratory and cannot establish the selected-pair effect. The current P6 viability pilot uses a new campaign and fingerprint and starts both selected variants from scratch. The final five-seed P7 comparison, if authorized by the P6 gate, must also start from scratch and will not reuse P4 checkpoints.

## 4. Evaluation and analysis

The paired evaluation uses deterministic policy actions and a matched evaluation runtime and execution-seed protocol for each pair. The 2025 interval supports viability diagnostics and training-only evaluation. The exposed 2026 interval is report-only and cannot support model selection or a fresh confirmatory out-of-sample claim. The primary outcome is total net return percentage, calculated as `100 × total_pnl / initial_balance`. For each seed 101–105, the paired difference is hidden-treatment return minus visible-control return in percentage points. Report every seed-level pair, the mean paired difference, and its two-sided 95% Student-t interval across the five paired differences (four degrees of freedom). This interval describes training-seed variability conditional on the fixed 2026 interval; it does not represent uncertainty across market years. A completed full-interval no-trade run contributes 0% return and remains included. Trade/fill counts and full-period survival are secondary diagnostics. If any declared seed lacks a valid completed pair, do not replace it or report a complete five-pair estimate.

For every completed run, report the exact configuration, checkpoint, data and feature identities; requested and completed steps; trades; return; drawdown; risk-gate survival; and entry-lifecycle diagnostics. Preserve all declared seeds, including zero-trade runs, and state clearly when the five-pair analysis is incomplete.

The selected-pair cost-stress analysis is planned as a separate evaluation-only comparison. The saved configurations specify 5.0 points of slippage, so the intended comparison is effective 5.0-point baseline fills versus 10.0-point stress fills. The current saved-checkpoint tick path bypasses that slippage setting; the evaluation override must apply the adverse price adjustment to the selected tick quote while preserving its timestamp, latency draw, and execution seed. The checkpoint, data, action mode, and all other settings must remain fixed, with original and effective cost configurations and hashes recorded. Until that override is implemented and validated, no cost-stress result will be claimed.

## 5. Related work

Reinforcement learning has been applied to portfolio allocation and algorithmic trading with different instruments, action spaces, horizons, and transaction-cost assumptions. Examples include policy-gradient comparisons on daily Dow Jones stocks ([Yang et al., 2020](https://doi.org/10.1145/3383455.3422540)) and deep reinforcement learning on continuous futures ([Zhang, Zohren, and Roberts, 2020](https://doi.org/10.3905/jfds.2020.1.030)). These studies establish methodological precedent; they do not show that PPO or the structural entry rules used here are profitable on this US100 cash-index M1 simulation.

Differential-Sharpe objectives have been studied as online trading-system objectives ([Moody and Saffell, 1998](https://proceedings.neurips.cc/paper_files/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf)), while proximal policy optimization is a general policy-gradient algorithm ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). Temporal convolutional networks provide a sequence-modeling alternative to recurrent architectures ([Bai, Kolter, and Koltun, 2018](https://arxiv.org/abs/1803.01271)). This thesis uses its saved PPO, encoder, reward, and risk settings as fixed components of the selected comparison; it does not test which algorithm, encoder, or reward is best.

Selection among many trials can overstate performance, motivating transparent trial history and strict separation of selection from reporting ([Bailey et al., 2017](https://doi.org/10.21314/jcf.2016.322)). Here, the previously inspected 2026 interval is exposed, report-only evidence; the paired-seed interval describes variation across training seeds conditional on that fixed market period, not uncertainty across market years. The present comparison asks whether exposing a causal entry-lifecycle state changes the saved policy's simulated net return. The cited literature does not test this intervention or validate the project's PO3/IFVG definitions. Further source notes and limits on transfer are recorded in [related work](related_work.md).

## 6. Results and interpretation

### Checkpoint recovery and deterministic replay

A recovery smoke restored a saved seed-50 checkpoint and completed the requested 16,384 smoke steps. Two deterministic reanalyses used the same checkpoint, resolved configuration, feature-cache identity, action schema, evaluation backend, execution seed, and split boundaries, while setting `PYTHONHASHSEED` to 0 and 73. Their six training and six report-split numerical artifacts had identical SHA-256 hashes; the repeatability report records no errors. The adjacent pass marker's report hash matches the recovery report. This establishes checkpoint recovery and reproducible report generation for that artifact. The smoke trainer exited with code 1 and failed the configured training survival gate (`peak_trailing_dd,max_loss_stopped_trading`), so it is not evidence of strategy viability or performance. No report-split performance values are reported here. Evidence: [recovery report](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/p3_smokes/job_21040072/p3_recovery_evidence.json), [repeatability report](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/p3_smokes/job_21040072/p3_hashseed_repeatability.json), and [pass marker](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/p3_smokes/passed.json).

### Initial 18-variant development screen (before the FSM repair)

The first seed-50 screen targeted 20 million steps per variant. Its terminal record covers all 18 variants: 15 controller runs succeeded, while three failed the configured training risk-survival gate (`ladder_a2_tp`, `ladder_a2_tp_po3`, and `ladder_a3_multi_tp_dist`). Eight variants recorded at least one training trade and ten recorded none. The predeclared visible-state and hidden-state FSM variants both recorded zero training trades in this screen. These are descriptive training-period outcomes from the implementation later found to have a baseline/FSM candidate-generation defect; they do not estimate the effect of state visibility and are not evidence of strategy efficacy. The risk gate was not relaxed, and no variant was selected from the exposed 2026 interval.

| Variant | Training trades | Training risk gate |
|---|---:|---|
| `ladder_a0_flat` | 251 | Pass |
| `ladder_a1_sl` | 0 | Pass |
| `ladder_a2_tp` | 571 | Fail |
| `ladder_a3_multi_tp` | 0 | Pass |
| `ladder_a4b_entry_fsm` (visible state) | 0 | Pass |
| `ladder_a4a_entry_fsm_hidden` (hidden state) | 0 | Pass |
| `ladder_a0_flat_po3` | 0 | Pass |
| `ladder_a1_sl_po3` | 0 | Pass |
| `ladder_a2_tp_po3` | 295 | Fail |
| `ladder_a3_multi_tp_po3` | 0 | Pass |
| `ladder_a4b_po3_ifvg` | 11 | Pass |
| `ladder_a4a_entry_fsm_hidden_po3` | 11 | Pass |
| `ladder_a0_flat_dist` | 0 | Pass |
| `ladder_a1_sl_dist` | 0 | Pass |
| `ladder_a2_tp_dist` | 0 | Pass |
| `ladder_a3_multi_tp_dist` | 498 | Fail |
| `ladder_a4b_distribution` | 1 | Pass |
| `ladder_a4a_entry_fsm_hidden_dist` | 1 | Pass |

*Training-only diagnostics from the initial screen; zero-trade rows are retained. “Pass” and “Fail” refer only to the configured training risk-survival gate.*

The campaign fingerprint is `b8001ab6430bb2e23d5e044b1dba7bfe2b407323c5d1e13065f459db335701cb`. The campaign manifest SHA-256 is `5f472210a2409b6116b28dc09f60635244a09a1833460785887b73fdb7529eb8`, matching the terminal summary's recorded manifest hash. The training-only counts above were extracted from the terminal summary [JSON](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/reports/campaign_results_summary_terminal_20261019.json), SHA-256 `1f66593cf8cc7a9a2308861588f12f37d69405f81c368d0143589a3b345c98e1`; the companion [summary report](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/reports/campaign_results_summary_terminal_20261019.md) has SHA-256 `001e9ced9eeffb6235b6a050141dfd99f169ac283419b5d4f27317b42724245d`. This paragraph reports no test-split metrics.

### Corrective development iteration (warm-start, exploratory)

The terminal P4 iteration continued inherited checkpoints after the baseline/FSM repair; it was not a fresh matched RQ1 comparison. In the selected pair's 2025 training evaluation, the visible-state control recorded 6 trades, +$1,002.70 total PnL, and a reconciled Differential-Sharpe reward sum of 0.2483. The hidden-state treatment recorded 7 trades, +$3.18 total PnL, and reward sum 0. Both completed the training period with full-year survival and tick-sourced fills. The treatment's zero reward fails the P6 learning-viability criterion despite its trades and positive PnL. These exploratory warm-start figures are not an estimate of the state-visibility effect. The paired training-only diagnostics are available for the [control](../../outputs/gpu_campaigns/replacement18_iter1_seed50_train2025_test2026_20261020_reorg1/runs/attempts/ladder_a4b_entry_fsm__seed50/attempt2/ladder_a4b_entry_fsm__seed50/20261021_050914_357392_rl_train_seed50_tcn/training/evaluation_diagnostics.json) and [treatment](../../outputs/gpu_campaigns/replacement18_iter1_seed50_train2025_test2026_20261020_reorg1/runs/attempts/ladder_a4a_entry_fsm_hidden__seed50/attempt2/ladder_a4a_entry_fsm_hidden__seed50/20261021_050902_218959_rl_train_seed50_tcn/training/evaluation_diagnostics.json); both reconcile reward attribution with zero error and no unattributed reward.

The P4 corrective iteration is terminal. Its three training risk-gate failures and the no-update checkpoint re-evaluation are retained; the resumed outcomes are exploratory. The current P6 scratch pilot has not completed, so no viability conclusion is available. Report all no-trade and risk-failure outcomes; do not interpret implementation checks as policy efficacy. The five-seed selected-pair campaign, if the viability gate passes, supplies the predeclared paired comparison. Because the 2026 interval is exposed, its results remain exploratory/report-only and do not constitute fresh confirmatory out-of-sample evidence.

## 7. Limitations

The available 2026 interval is exposed and cannot establish untouched out-of-sample performance. The simulator’s tick-based fill rule is an explicit latency approximation, not a claim about real execution. The data feed, contract details, and external redistribution rights are not independently established by the training campaign. Model results apply to the saved configurations, supplied data, and simulator rules evaluated here; they do not establish live trading performance.

## 8. Submission checklist

- Replace this working draft with the accepted university manuscript format and complete front matter.
- Insert only verified P4–P7 results and their exact report identities.
- Complete independent review and freeze claims, limitations, source, data, checkpoints, and reports.
- Produce and inspect the supervisor PDF by 9 November 2026.
- Incorporate required supervisor corrections, then hash the exact final manuscript and submission files by 23 November 2026.
