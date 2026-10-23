# Entry-State Observability in Intraday Reinforcement-Learning Trading

**Working results draft — corrected P6/P7 evidence pending**

**Status:** not final. The P6/P7 results below were produced under the previous entry-transition rules. The corrected 2M-per-arm seed-50 pilot under campaign `rq1_entry_fsm_pair_seed50_train2025_20261024_entry_levels_pilot2m_v3` (fingerprint `9fafe8eccf1e38c2bc2863bc5e77661385e24e4ce8da47e38d8d8e5d12fd278e`) completed, but both arms failed the P6 training-survival gate. Diagnosis shows that the pilot did not have a 50% win rate and that evaluation did not enforce the report's 10% peak-drawdown threshold as an active stop. P6 remains open pending a bounded matched pilot with aligned drawdown behavior; do not treat earlier P6/P7 results as evidence for the corrected implementation. This Markdown file remains the requested deliverable; no full thesis, LaTeX document, supervisor PDF, or faculty-submission artifact is required.

## Abstract

This study evaluates whether exposing a simulator's causal entry-lifecycle state to a PPO trading policy changes simulated net returns. The selected visible-state control and hidden-state treatment are frozen, but the corrected 2M-step seed-50 pilot failed the training survival gate in both arms and is not a viability or comparative result. Reported win rates were 41.73% and 39.38%, not 50%; the hidden-state arm rose to $148,785 and later fell to $90,009, a 39.50% peak-to-trough drawdown. The evaluation used a $10,000 loss limit from initial equity but did not actively enforce the separate 10% peak-drawdown acceptance threshold. P6 remains open pending a bounded matched pilot with aligned drawdown behavior. The paired estimates, viability results, and cost-stress results retained below are historical results from the superseded transition rules and must not be used to draw conclusions about the corrected implementation. The 2026 report interval was inspected previously, so any results on that interval remain exploratory rather than fresh holdout evidence.

## 1. Introduction and research question

Intraday reinforcement-learning experiments require a policy to act on a time-ordered market state while trades pass through execution and risk rules that affect their outcomes. In this study, a trade opportunity also follows a causal lifecycle: a setup becomes a candidate, may become armed, and can then be entered. The primary question is whether exposing that lifecycle state changes the policy's net simulated return; entry activity and lifecycle counts describe how its behavior differs under the specified simulator.

The planned comparison holds the PPO algorithm, TCN encoder, reward and risk configuration, market inputs, lifecycle logic, requested training budget, and evaluation procedure fixed. Its declared configuration difference is `env.entry_state_observation`: the visible-state control includes four one-hot coordinates in the order `FLAT`, `CANDIDATE`, `ARMED`, `IN_POSITION`; the hidden-state treatment omits them. The pair's account observation therefore has widths 10 and 6, respectively. The estimand is limited to this observation change under the saved configurations and simulation protocol. The study does not compare algorithms or establish general profitability. Prior work motivates the method components but does not establish performance of this configuration.

The selected pair remains frozen: control `ladder_a4b_entry_fsm` exposes the lifecycle state, while treatment `ladder_a4a_entry_fsm_hidden` hides it. The P4 corrective screen is terminal and exploratory. The previous seed-50 P6 and five-seed P7 results were produced under the superseded transition rules; they remain documented below for historical traceability but do not pass the current P6/P7 gates. The corrected 2M-per-arm seed-50 pilot completed from scratch, but both arms failed survival. A fresh full-budget run is held until a bounded matched pilot tests the same explicit peak-drawdown behavior in both arms. The 2026 interval remains exposed exploratory evidence.

The primary outcome is total net return over the configured 2026 report interval. The interval has been inspected previously, so this paired estimate is exploratory and cannot provide fresh confirmatory out-of-sample evidence. This report makes no efficacy claim and does not ask whether the PO3/IFVG rules are profitable generally or whether simulated fills establish live-market profitability.

## 2. Data and temporal protocol

The training interval is 30 December 2024 through 31 December 2025. The report-only interval is 1 January through 16 October 2026; the supplied M1 data contain no bars on 1 January, so the first test-session observations begin on 2 January. The 2026 interval has been inspected previously and is classified as exposed exploratory evidence. It is not a fresh or confirmatory holdout and is not used to select variants, change the pair, or tune the method.

The selected-pair runs use the US100 cash-index feed. The split contains 351,432 M1 training bars and 279,355 M1 report bars. The configured date boundary and the first supplied report bar differ because no bar exists on 1 January. All ten terminal run manifests record matching train/report inputs across each same-seed pair, including bars, features, secondary data, and bid/ask ticks.

| Partition | Configured date window | M1 rows | Role |
|---|---|---:|---|
| Training | 2024-12-30–2025-12-31 | 351,432 | Policy learning and training-only viability checks |
| Report | 2026-01-01–2026-10-16 | 279,355 | Exposed, report-only evaluation; no model or pair selection |

All supplied observations remain in their chronological partition, including bars from short sessions. Configured session-schedule entries are date-level inputs available before the first observation of each date; the policy is not expected to infer those scheduled closes from that day’s price path. The training and checkpoint-reanalysis paths attach this context before splitting bars into training and report intervals, so configured dates receive the same schedule fields in both partitions. All ten P7 run manifests record the three schedule inputs in both the market-context and policy-sequence schemas, with schedule SHA-256 `7e9f5def3385185905f7265f946a0835ccb43c53f1351c057aa23f00480e6cbc`; the schedule has 16 configured dates. The 2025 shape audit found ten substantially shorter observed NY sessions per symbol (roughly 206–225 rows versus 391); every one overlaps a configured date. The additional below-391 counts are near-full days: US100 has 382 rows on 2025-12-02 and 388 on 2025-12-04; US500 has 390 on 2025-10-27, 383 on 2025-12-02, and 388 on 2025-12-04. The audit does not establish whether those few missing minute labels reflect feed gaps or a different provider schedule, so they are not reclassified as scheduled short sessions. No bars are removed. The report-period path receives the same date-level schedule before the split, but no per-date report-period shape audit establishes feed-specific calendar completeness. See the [training-period shape audit](../../outputs/data_refresh_20261017/replacement_schedule_shape.json) (SHA-256 `163ecbdb9e99ad28329652c642b537e458cc3cf8c0f585334624888575ed084e`) and the [configured schedule](../../config/ny_session_schedule.csv).

M1 is the only policy timestep. Supplied M3, M5, M10, M15, M30, H1, H4, and D1 bars are retained for multi-timeframe feature construction and aligned to the M1 decision index only after their source bars close; they do not create additional policy decision timelines. Configured schedule dates are marked from the first bar; the flags describe the declared calendar, not a classification inferred from observed row counts. Bid/ask tick data drive simulated execution. The declared fill protocol draws a seeded arrival delay from the configured 0–30 millisecond interval and uses the first eligible bid/ask tick at or after order arrival. The configuration also declares 5.0 points of slippage, but the strict tick-fill path currently executes at the selected raw bid/ask and does not apply this extra slippage. Thus the P6 run tests viability under tick quotes and randomized latency, not under the configured extra slippage. This is a reproducible simulator assumption; it does not model queue position, market impact, or all broker execution effects.

## 3. Policy and intervention

The candidate policies use PPO with a TCN encoder and the pair-specific saved effective configurations. Each run requests 20 million environment steps. The corrective screen uses seed 50 for one run per each of 18 variants; it is a repair and viability iteration, not the final paired-seed analysis. After the selected pair passed P6, the predeclared P7 paired evaluation started both arms from scratch under a new fingerprint, with matched seeds 101–105 and identical seed assignment in both arms. A separate earlier campaign using two of these seed IDs was interrupted under the defective implementation and is ineligible; none of its checkpoints were reused. Any resumed attempt within P7 continues only from a checkpoint produced by that same P7 campaign.

All ten terminal P7 manifests record the same training settings across both arms: a shared actor with a TCN encoder using channels `[128, 128, 256, 256]`, dilations `[1, 2, 4, 8]`, kernel size 3, and latent width 128; and a policy head with widths `[256, 128]`. PPO uses 64 logical environments, 128 steps per environment per rollout (8,192 transitions), minibatches of 4,096, 15 epochs, learning rate `3×10⁻⁴`, discount `γ=0.99`, GAE `λ=0.95`, clip range 0.2, and entropy coefficient 0.001. This gives two minibatches per epoch and 30 per rollout. The reward is differential Sharpe with `η=0.01`; continuous actions use a clamped diagonal Gaussian. The visible-state and hidden-state policies have 1,438,051 and 1,437,919 parameters, respectively; this count difference follows from the four additional observed state coordinates. Each arm completed at least 20 million training steps, with final and resume checkpoint hashes verified.

The entry-state machine represents a causal opportunity lifecycle: a qualifying setup creates a candidate, subsequent evidence may arm it, and an eligible policy action can trigger entry. The intervention changes whether the policy observation exposes the machine state. It does not change the market data, entry evidence, execution protocol, or risk rules. The frozen configuration audit records `env.entry_state_observation` as the sole difference; the terminal resolved configurations and manifests confirm this for all five pairs.

The development screen trains each of the 18 registered variants once with seed 50 and a 20-million-step budget. It is a repair and viability screen, not the selected-pair estimate. The P4 corrective iteration is terminal and is classified as exploratory because it resumed warm-start checkpoints under an explicit fingerprint override. The completed P6 pilot started both frozen pair arms from scratch with seed 50 and the same 20-million-step budget on the 2025 training partition; its viability decision used only the 2025 in-sample evaluation and reconciled reward diagnostics. After that gate passed, P7 started both arms from scratch for matched training seeds 101–105 with the same requested budget per run. Runtime libraries, resolved environment settings, and data identities are recorded in each run's effective configuration and manifest.

The terminal P4 attempt warm-started all 18 variants from checkpoints in the predecessor corrective campaign. The checkpoint map records six initializations at 9 million steps, four at 10 million, seven at 11 million, and one at 13 million. The two selected FSM arms each started at 9 million steps. The selected checkpoints were produced under source revision `b3fe72dca3e17f85fb22c746d5e9efeb70587d90`; the continuation used reorganized source revision `ce714385a1866e95d0ecf1ce97a98f49b2c0d5c1`. These were warm-start continuations, not fresh 20-million-step runs under the current source. The P6 viability pilot passed using a new fingerprint and scratch starts for both selected variants. The fresh five-seed P7 comparison then trained both arms from scratch, resumed only within its own fingerprint after allocation timeouts, and completed all five matched seeds; no checkpoints from P4 or the paused defective campaign were reused.

## 4. Evaluation and analysis

The paired evaluation uses deterministic policy actions and a matched evaluation runtime and execution-seed protocol for each pair. The 2025 interval supports viability diagnostics and training-only evaluation. The exposed 2026 interval is report-only and cannot support model selection or a fresh confirmatory out-of-sample claim. The primary outcome is total net return percentage, calculated as `100 × total_pnl / initial_balance`. For each seed 101–105, the paired difference is hidden-treatment return minus visible-control return in percentage points. Report every seed-level pair, the mean paired difference, and its two-sided 95% Student-t interval across the five paired differences (four degrees of freedom). This interval describes training-seed variability conditional on the fixed 2026 interval; it does not represent uncertainty across market years. A completed full-interval no-trade run contributes 0% return and remains included. Trade/fill counts and full-period survival are secondary diagnostics. If any declared seed lacks a valid completed pair, do not replace it or report a complete five-pair estimate.

The five paired runs are complete and identity matched. They belong to campaign `rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1`, canonical fingerprint `10e75ab7a25a53b3d4444098185fe45603811f09f2d556265c9f80ca402388ab`, and evidence role `matched_exploratory`; the campaign manifest SHA-256 is `a91ec6d3c685344b3a544cf658c55a9c187799005871d28a82f28a62984924ee`. Its source snapshot contains 539 files; `source_snapshot.zip` SHA-256 is `496bc6122be17f7015ab6db3c81b4e4b960899863ce9f14120e2ce2cae23e69d`, and its source-hash manifest SHA-256 is `dd1629ae9198a3ac792b32e8cbe6b38d930763c07ac7151b15dc947b5f258ba1`. The control's environment SHA-256 is `08b82f189bcd7d4dfa0971afa0386f438eb651f6ce8e8729baecf6bff61db09f` and the treatment's is `4ac1312093e436b99d31088b79764790cdc5fa230545f1b2a8340837b9c7ddcc`; these hashes are consistent across all five seeds. A direct readback of all five resolved-config pairs confirmed that their only substantive difference is `env.entry_state_observation` (recorded in both the override list and effective training arguments); the variant identifier labels the corresponding arm. For every seed, control/treatment provenance matches on campaign fingerprint, costs, dataset, report interval, execution, features, initial balance, model settings, reward, split ID, and completed training steps. The paired payload's 40 non-checkpoint source files (attempt provenance, run manifests, resolved configs, and report metrics) were SHA-256 checked against disk. Provenance for every pair matches on dataset SHA-256 `fa075b447bea41b57f664923751f93b880f11ec30e81752ed8c9640b2a741ff2`, split ID `d124a07434d1fa134e39a4098c3f819aa6d5396d318274f2ec7a0b0cf29bd353`, feature SHA-256 `076e4565f63d4d4a96d00c69511109c6b2a7c312f0e606d588bbd502920765a8`, and execution SHA-256 `976ebac0667cc32a8339f03c8ada941448a0d593fc32c558da5d7d88c29ac9e8`. The saved evaluation timestamps are 2026-01-02 01:05:00+03:00 through 2026-10-16 23:49:00+03:00. Seed-level returns and checkpoint lineage are also in the [machine-readable provenance payload](../../outputs/gpu_campaigns/rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1/reports/p7_paired_seed_metrics_exploratory_20261022_v2.json); its analyzer input SHA-256 is `2c206a0f9c8de691bbe902ed4c6767699a512c5551e2af22f19bb26efeaac136`. Preserve every declared seed, including low-trade outcomes, and do not replace these results with a different seed set.

The selected-pair cost-stress analysis is a separate evaluation-only comparison. The saved configurations specify 5.0 points of slippage; the replay therefore compares effective 5.0-point baseline fills with 10.0-point adverse stress fills. The evaluation override worsens only the executed quote side after tick selection, preserving the opposite side, selected tick, timestamp, latency draw, execution seed, and position-sizing reserve. Checkpoint, data, action mode, and all other settings are fixed. P6 and P7 training used raw selected tick quotes without this extra slippage, so the replay describes evaluation cost sensitivity, not the costs experienced during training. The 24-record ledger and 240 artifact hashes passed audit, including exact repeats of seed 101 for both arms at both cost levels. Numerical results and provenance are in Section 6.2 and the linked report.

## 5. Related work

Reinforcement learning has been applied to portfolio allocation and algorithmic trading with different instruments, action spaces, horizons, and transaction-cost assumptions. Examples include policy-gradient comparisons on daily Dow Jones stocks ([Yang et al., 2020](https://doi.org/10.1145/3383455.3422540)) and deep reinforcement learning on continuous futures ([Zhang, Zohren, and Roberts, 2020](https://doi.org/10.3905/jfds.2020.1.030)). These studies establish methodological precedent; they do not show that PPO or the structural entry rules used here are profitable on this US100 cash-index M1 simulation.

Differential-Sharpe objectives have been studied as online trading-system objectives ([Moody and Saffell, 1998](https://proceedings.neurips.cc/paper_files/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf)), while proximal policy optimization is a general policy-gradient algorithm ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). Temporal convolutional networks provide a sequence-modeling alternative to recurrent architectures ([Bai, Kolter, and Koltun, 2018](https://arxiv.org/abs/1803.01271)). This thesis uses its saved PPO, encoder, reward, and risk settings as fixed components of the selected comparison; it does not test which algorithm, encoder, or reward is best.

Partial-observability research studies when an agent's observation is not a sufficient state representation. Allen et al. introduce the λ-discrepancy as a way to detect this mismatch and demonstrate a recurrent memory function for mitigating it ([Allen et al., 2024](https://papers.nips.cc/paper_files/paper/2024/hash/73073ccb3bc559fd001e66b9079d6d5e-Abstract-Conference.html)). Nguyen et al. use a fully observable state expert during training to help a partially observable policy ([Nguyen et al., 2023](https://proceedings.mlr.press/v205/nguyen23a.html)). These works study broader sequential-decision and robotics settings. This thesis tests a narrower intervention: it exposes or omits four already-computed entry-lifecycle indicators while keeping the environment and training configuration fixed; it adds neither a learned memory nor a state-expert training objective.

Selection among many trials can overstate performance, motivating transparent trial history and strict separation of selection from reporting ([Bailey et al., 2017](https://doi.org/10.21314/jcf.2016.322)). Here, the previously inspected 2026 interval is exposed, report-only evidence; the paired-seed interval describes variation across training seeds conditional on that fixed market period, not uncertainty across market years. The comparison asks whether exposing the causal entry-lifecycle state changes the saved policy's simulated net return. The cited literature does not test this intervention or validate the project's PO3/IFVG definitions.

## 6. Results and interpretation

**Evidence status:** the historical P6/P7 results in this section were generated before the universal session-level entry gate was removed. They are retained to preserve the audit trail and are not results for the corrected implementation. The corrected 2M pilot is recorded below; both arms failed the training survival gate. P6 remains open pending a bounded risk-policy pilot; P7 and P9 remain gated on P6.

### Checkpoint recovery and deterministic replay

A recovery smoke restored a saved seed-50 checkpoint and completed the requested 16,384 smoke steps. Two deterministic reanalyses used the same checkpoint, resolved configuration, feature-cache identity, action schema, evaluation backend, execution seed, and split boundaries, while setting `PYTHONHASHSEED` to 0 and 73. Their six training and six report-split numerical artifacts had identical SHA-256 hashes; the repeatability report records no errors. The adjacent pass marker's report hash matches the recovery report. This establishes checkpoint recovery and reproducible report generation for that artifact. The smoke trainer exited with code 1 and failed the configured training survival gate (`peak_trailing_dd,max_loss_stopped_trading`), so it is not evidence of strategy viability or performance. No report-split performance values are reported here. Evidence: [recovery report](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/p3_smokes/job_21040072/p3_recovery_evidence.json), [repeatability report](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/p3_smokes/job_21040072/p3_hashseed_repeatability.json), and [pass marker](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/p3_smokes/passed.json).

### Initial 18-variant development screen (before the FSM repair)

The first seed-50 screen targeted 20 million steps per variant and predates the baseline/FSM repair. Eight variants recorded at least one training trade and ten recorded none; both predeclared FSM arms recorded zero training trades. The terminal summary reports 15 controller successes and three failures of the configured training risk-survival gate (`ladder_a2_tp`, `ladder_a2_tp_po3`, `ladder_a3_multi_tp_dist`). This screen's 2026 evaluation role was `selection_acknowledged` on a previously inspected interval. These results are descriptive development evidence from an implementation later found to have a baseline/FSM candidate-generation defect; they neither estimate the effect of state visibility nor provide a confirmatory comparison or strategy-efficacy result. The screen is not treated as an unexposed holdout or as a basis for a new selection claim.

**Training outcomes**

| Variant | Controller | Train gate | Full-year episodes | Environments reaching year end | Train return | Train PnL (USD) | Train trades |
|---|---|---|---:|---:|---:|---:|---:|
| `ladder_a0_flat` | succeeded | PASS | 64 | 64 | -3.61% | -3,606.28 | 251 |
| `ladder_a1_sl` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a2_tp` | failed | FAIL | unavailable | unavailable | -10.10% | -10,102.11 | 571 |
| `ladder_a3_multi_tp` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a4b_entry_fsm` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a4a_entry_fsm_hidden` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a0_flat_po3` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a1_sl_po3` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a2_tp_po3` | failed | FAIL | 0 | 0 | -10.13% | -10,127.90 | 295 |
| `ladder_a3_multi_tp_po3` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a4b_po3_ifvg` | succeeded | PASS | 0 | 0 | -0.89% | -886.36 | 11 |
| `ladder_a4a_entry_fsm_hidden_po3` | succeeded | PASS | 0 | 0 | -1.69% | -1,692.44 | 11 |
| `ladder_a0_flat_dist` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a1_sl_dist` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a2_tp_dist` | succeeded | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a3_multi_tp_dist` | failed | FAIL | 0 | 0 | -10.40% | -10,403.69 | 498 |
| `ladder_a4b_distribution` | succeeded | PASS | 0 | 0 | -0.44% | -444.65 | 1 |
| `ladder_a4a_entry_fsm_hidden_dist` | succeeded | PASS | 0 | 0 | -0.00% | -0.16 | 1 |

**Exposed report-period outcomes**

| Variant | Return | PnL (USD) | Max DD | Sharpe | Profit factor | Trades | Survived / sessions |
|---|---:|---:|---:|---:|---:|---:|---|
| `ladder_a0_flat` | -1.73% | -1,731.17 | 3.41% | -1.005 | 0.870 | 197 | yes / 115/205 |
| `ladder_a1_sl` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a2_tp` | -10.28% | -10,279.84 | 12.51% | -1.142 | 0.834 | 166 | no / 44/205 |
| `ladder_a3_multi_tp` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a4b_entry_fsm` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a4a_entry_fsm_hidden` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a0_flat_po3` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a1_sl_po3` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a2_tp_po3` | -10.32% | -10,319.65 | 18.30% | -0.959 | 0.799 | 99 | no / 23/205 |
| `ladder_a3_multi_tp_po3` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a4b_po3_ifvg` | -0.62% | -623.56 | 0.64% | -1.252 | 0.023 | 4 | yes / 4/205 |
| `ladder_a4a_entry_fsm_hidden_po3` | -1.39% | -1,393.74 | 1.40% | -2.095 | 0.000 | 4 | yes / 4/205 |
| `ladder_a0_flat_dist` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a1_sl_dist` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a2_tp_dist` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a3_multi_tp_dist` | -10.48% | -10,480.14 | 15.07% | -1.008 | 0.764 | 82 | no / 20/205 |
| `ladder_a4b_distribution` | +0.49% | +491.32 | 0.20% | 1.306 | — | 1 | yes / 1/205 |
| `ladder_a4a_entry_fsm_hidden_dist` | +0.00% | +0.28 | 0.00% | 1.306 | — | 1 | yes / 1/205 |

The controller report reconciled 36/36 train/test PnL values against final equity; three runs failed both named train-survival checks (`max_loss_stopped_trading` and `peak_trailing_dd`). The full per-run evidence and resource samples are in the [terminal screen summary](../../outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/reports/campaign_results_summary_terminal_20261019.md). Campaign fingerprint: `b8001ab6430bb2e23d5e044b1dba7bfe2b407323c5d1e13065f459db335701cb`; campaign manifest SHA-256: `5f472210a2409b6116b28dc09f60635244a09a1833460785887b73fdb7529eb8`. The matching terminal JSON has SHA-256 `1f66593cf8cc7a9a2308861588f12f37d69405f81c368d0143589a3b345c98e1`; the summary Markdown SHA-256 is `001e9ced9eeffb6235b6a050141dfd99f169ac283419b5d4f27317b42724245d`.

### Corrective development iteration (warm-start, exploratory)

The terminal P4 iteration continued inherited checkpoints after the baseline/FSM repair; it was not a fresh matched RQ1 comparison. In the selected pair's 2025 training evaluation, the visible-state control recorded 6 trades, +$1,002.70 total PnL, and a reconciled Differential-Sharpe reward sum of 0.2483. The hidden-state treatment recorded 7 trades, +$3.18 total PnL, and reward sum 0. Both completed the training period with full-year survival and tick-sourced fills. The treatment's zero reward fails the P6 learning-viability criterion despite its trades and positive PnL. These exploratory warm-start figures are not an estimate of the state-visibility effect. The paired training-only diagnostics are available for the [control](../../outputs/gpu_campaigns/replacement18_iter1_seed50_train2025_test2026_20261020_reorg1/runs/attempts/ladder_a4b_entry_fsm__seed50/attempt2/ladder_a4b_entry_fsm__seed50/20261021_050914_357392_rl_train_seed50_tcn/training/evaluation_diagnostics.json) and [treatment](../../outputs/gpu_campaigns/replacement18_iter1_seed50_train2025_test2026_20261020_reorg1/runs/attempts/ladder_a4a_entry_fsm_hidden__seed50/attempt2/ladder_a4a_entry_fsm_hidden__seed50/20261021_050902_218959_rl_train_seed50_tcn/training/evaluation_diagnostics.json); both reconcile reward attribution with zero error and no unattributed reward.

The full terminal P4 variant outcomes are summarized below. Training metrics are in-sample; the 2026 results are exposed report-only outcomes from a previously inspected interval. `PASS` and `FAIL` refer only to the configured training survival gate. The `ladder_a3_multi_tp` controller success (†) only re-evaluated an inherited 20-million-step checkpoint: its attempt provenance points to the prior attempt's final checkpoint, SHA-256 `18e61ad8e956df114bd1031f20f90f8a4cd178c3506cf5f1e1a37ad03e3e674a`, and its final manifest records zero requested new learning steps. Thus a controller success is not necessarily a fresh training update. These warm-start results are exploratory and were not used as comparative evidence or to select the frozen pair. The tables transcribe the terminal summary; `unavailable` indicates the source summary did not have the value, and `—` indicates that the source reported no profit factor for that row.

**Training outcomes**

| Variant | Gate | Full-year episodes | Environments reaching year end | Train return | Train PnL (USD) | Train trades |
|---|---|---:|---:|---:|---:|---:|
| `ladder_a0_flat` | PASS | 64 | 64 | -3.61% | -3,606.28 | 251 |
| `ladder_a1_sl` | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a2_tp` | FAIL | unavailable | unavailable | -10.10% | -10,102.11 | 571 |
| `ladder_a3_multi_tp` | PASS† | unavailable | unavailable | +0.00% | +0.00 | 0 |
| `ladder_a4b_entry_fsm` | PASS | 0 | 0 | +1.00% | +1,002.70 | 6 |
| `ladder_a4a_entry_fsm_hidden` | PASS | 0 | 0 | +0.00% | +3.18 | 7 |
| `ladder_a0_flat_po3` | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a1_sl_po3` | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a2_tp_po3` | FAIL | unavailable | unavailable | -10.69% | -10,688.57 | 314 |
| `ladder_a3_multi_tp_po3` | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a4b_po3_ifvg` | PASS | 0 | 0 | -0.89% | -886.36 | 11 |
| `ladder_a4a_entry_fsm_hidden_po3` | PASS | 0 | 0 | -1.69% | -1,692.44 | 11 |
| `ladder_a0_flat_dist` | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a1_sl_dist` | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a2_tp_dist` | PASS | 0 | 0 | +0.00% | +0.00 | 0 |
| `ladder_a3_multi_tp_dist` | FAIL | unavailable | unavailable | -10.48% | -10,477.67 | 318 |
| `ladder_a4b_distribution` | PASS | 0 | 0 | -0.44% | -444.65 | 1 |
| `ladder_a4a_entry_fsm_hidden_dist` | PASS | 0 | 0 | -0.00% | -0.16 | 1 |

**Exposed report-period outcomes**

| Variant | Return | PnL (USD) | Max DD | Sharpe | Profit factor | Trades | Survived / sessions |
|---|---:|---:|---:|---:|---:|---:|---|
| `ladder_a0_flat` | -1.73% | -1,731.17 | 3.41% | -1.005 | 0.870 | 197 | yes / 115/205 |
| `ladder_a1_sl` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a2_tp` | -10.28% | -10,279.84 | 12.51% | -1.142 | 0.834 | 166 | no / 44/205 |
| `ladder_a3_multi_tp` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a4b_entry_fsm` | +5.39% | +5,387.81 | 1.14% | 2.099 | 3.749 | 7 | yes / 7/205 |
| `ladder_a4a_entry_fsm_hidden` | +0.00% | +1.56 | 0.00% | 1.130 | 3.977 | 8 | yes / 8/205 |
| `ladder_a0_flat_po3` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a1_sl_po3` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a2_tp_po3` | -10.12% | -10,115.57 | 12.12% | -1.051 | 0.571 | 51 | no / 10/205 |
| `ladder_a3_multi_tp_po3` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a4b_po3_ifvg` | -0.62% | -623.56 | 0.64% | -1.252 | 0.023 | 4 | yes / 4/205 |
| `ladder_a4a_entry_fsm_hidden_po3` | -1.39% | -1,393.74 | 1.40% | -2.095 | 0.000 | 4 | yes / 4/205 |
| `ladder_a0_flat_dist` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a1_sl_dist` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a2_tp_dist` | +0.00% | +0.00 | 0.00% | 0.000 | 0.000 | 0 | yes / 0/205 |
| `ladder_a3_multi_tp_dist` | -10.21% | -10,211.90 | 13.89% | -0.915 | 0.763 | 67 | no / 20/205 |
| `ladder_a4b_distribution` | +0.49% | +491.32 | 0.20% | 1.306 | — | 1 | yes / 1/205 |
| `ladder_a4a_entry_fsm_hidden_dist` | +0.00% | +0.28 | 0.00% | 1.306 | — | 1 | yes / 1/205 |

The terminal campaign report records 15 controller successes, three training-gate failures, and 36/36 train/test PnL reconciliations. The P4 campaign identity is `replacement18_iter1_seed50_train2025_test2026_20261020_reorg1`, canonical fingerprint `0b64554caf4e147731df892495f0f76a30e43794b816530ae172a20dcd491597`. Its campaign record preserves the explicit resume override: canonical fingerprint `0b64554caf4e147731df892495f0f76a30e43794b816530ae172a20dcd491597`, effective fingerprint `a1f50da0fc2fac4d4bd9a3c777c1f4505d5c7f2f95d8c76223ef37bfa94d1f82`, with results labeled exploratory after the user-approved continuation. The full per-run evidence links and resource samples are in the [terminal campaign summary](../../outputs/gpu_campaigns/replacement18_iter1_seed50_train2025_test2026_20261020_reorg1/reports/campaign_results_summary_terminal_20261021.md); its SHA-256 is `14c04a1b1dda8f7c93eb4ad8d343b52951380282b289778abc63aacdf8f9b924`, and the campaign manifest SHA-256 is `cb266e6bfbf3838454a15a74598c20bbb84fa274bdfe5c5b6e2880ec9e8a1b6d`. P4 is a development screen and supplies no causal estimate for the visible-versus-hidden intervention.

The P4 corrective iteration is terminal. Its three training risk-gate failures and the no-update checkpoint re-evaluation are retained; the resumed outcomes are exploratory. Under the prior entry rules, the P6 scratch pilot and P7 paired analysis passed their historical run-level checks; neither supports claims about the corrected entry logic. The 2026 interval was previously inspected, so the old P7 analysis was exploratory and was not fresh confirmatory out-of-sample evidence.

### Corrected 2M seed-50 pilot (P6 preflight; survival gate failed)

The corrected pilot tested startup, learning progress, and entry-lifecycle operation using the frozen visible-state control and hidden-state treatment. Each arm trained from scratch for the requested 2,000,000 steps and completed 2,007,040 steps. The full 2025 training-split replay produced active trading and fully reconciled reward attribution in both arms, but each hit the unchanged survival gate. Reported training win rates were 41.73% in the visible arm and 39.38% in the hidden arm; neither arm therefore had a 50% win rate. Their mean PnL per completed position was -$72.23 and -$7.37, with profit factors 0.712 and 0.972. A win-rate percentage alone does not control drawdown: returns also depend on win/loss size and sequence. This pilot is not a P6 viability pass and provides no comparative-effect estimate.

| Arm | Completed steps | Training trades | Training PnL | Return | Max peak-to-trough drawdown | Breach | Cumulative DSR reward | Reward reconciliation error | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Visible-state control (`ladder_a4b_entry_fsm`) | 2,007,040 | 139 | -$10,040.58 | -10.04% | 10.41% | 1 | -21.2300 | 2.1e-14 | FAIL |
| Hidden-state treatment (`ladder_a4a_entry_fsm_hidden`) | 2,007,040 | 1,356 | -$9,991.11 | -9.99% | 39.50% | 1 | -16.8194 | 7.8e-14 | FAIL |

The visible arm first breached on 2025-02-07; the hidden arm's recorded breach time was 2025-11-12. Activity gates passed. The visible arm recorded 325 tick-sourced fill-quote events; the hidden arm recorded 2,712. These activity differences are descriptive for this single seed and short pilot only. The hidden arm's equity peaked at $148,785 on March 20, 2025, then fell to $90,009 on November 12, 2025: 39.50% from peak, despite the final loss being about 10% of initial equity. The saved configs set `env.peak_trailing_dd_limit=0`; the evaluation path disables the PPO-only trailing guardrail and applies the $10,000 loss limit from initial equity. Thus the report's 10% peak-to-trough threshold is a post-run gate, not an active evaluation stop. The visible arm peaked at $100,413 and ended at $89,959, giving 10.41% peak drawdown. The configured drawdown metric and breach checks are consistent; the mismatch is that the acceptance threshold was not an active behavior limit in this evaluation. Automatic 2026 report evaluation is excluded from P6 because that interval is exposed exploratory evidence. Both final checkpoint SHA-256 values and each run/config manifest are recorded in the campaign run manifests. Campaign `rq1_entry_fsm_pair_seed50_train2025_20261024_entry_levels_pilot2m_v3` used fingerprint `9fafe8eccf1e38c2bc2863bc5e77661385e24e4ce8da47e38d8d8e5d12fd278e`, source commit `cbf7825170dee7f68ad578a19023e0847b11393f`, and source-snapshot SHA-256 `a4b3737fcf0ae9ee9329de50f461302c965d3cc97317be07fa97e2e8639910f1`. Before the 20M viability run, test a consistent 10% peak-drawdown behavior from scratch in a bounded matched pilot; no current pilot checkpoint is eligible for warm start.

### Historical seed-50 viability pilot (P6; superseded entry rules)

After the FSM repair, each selected arm was trained from scratch for the requested 20 million steps with seed 50 on the 2025 training interval. Both completed 20,004,864 steps, survived the full training year without a risk-gate breach, and generated seven trades. The outcome is a minimum viability check for the repaired training path; the single seed does not estimate the effect of exposing the lifecycle state.

| Arm | Completed steps | Trades | Net PnL | Return | Max drawdown | Full-year survival | Cumulative DSR reward | Reward error / unattributed | Candidates; entries / closes |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| Visible-state control (`ladder_a4b_entry_fsm`) | 20,004,864 | 7 | $6,349.80 | 6.35% | 2.76% | yes | -4.0182 | 0.0 / 0.0 | 4,530; 7 / 7 |
| Hidden-state treatment (`ladder_a4a_entry_fsm_hidden`) | 20,004,864 | 7 | $4,579.86 | 4.58% | 1.70% | yes | -5.6819 | 0.0 / 0.0 | 4,577; 7 / 7 |

Both cumulative Differential-Sharpe reward sums are negative. Therefore the gate establishes only that both repaired policies completed the training year with trades and reconciled reward attribution; it does not establish positive risk-adjusted learning or comparative efficacy. Each arm recorded fourteen tick-sourced fill-quote events. The automatically generated 2026 evaluation is excluded. These P6 runs used selected raw tick quotes without the additional configured 5-point slippage. The scratch campaign was `p6_entry_fsm_viability_seed50_train2025_20261021_scratch1`, fingerprint `3ca5c8e0cd3df51b29a5cdcb8f3d1c0f602a6c6721bf5aed3f6986a851b769b7`; its campaign-manifest SHA-256 is `df4ae5c024e0e44e5673cbd53b644b2ef3c6d5a523cee1c831a653540f62ca71`. See the [provenance-complete P6 report](evidence/p6_seed50_training_viability_20261021.md) (SHA-256 `9c16acfe28016a2b8e5e7cee358ac6541831f4d44b29984fccec6a59ea00cb0b`) and [machine-readable evidence](evidence/p6_seed50_training_viability_20261021.json) (SHA-256 `4dfa4e10f8c5d44f3e254ec3e961922b7d12fc7cf85ae3466ca5a279968aacc2`).

### Historical P7 exploratory paired results (superseded entry rules)

The primary outcome is net return, calculated as `100 × total_pnl / initial_balance`. The paired difference is hidden treatment minus visible control in percentage points. All five pairs completed at least the requested 20 million training steps, and each pair shares the same data, feature, execution, reward, cost, model, and evaluation identities. The full source paths and hashes are recorded in the machine-readable payload.

Each arm requested 20,000,000 training steps. Completed step counts below were checked against all ten terminal run manifests.

| Seed | Visible control steps | Hidden treatment steps |
|---:|---:|---:|
| 101 | 20,004,864 | 20,005,312 |
| 102 | 20,004,864 | 20,004,864 |
| 103 | 20,001,280 | 20,004,864 |
| 104 | 20,001,280 | 20,005,440 |
| 105 | 20,005,312 | 20,005,440 |

| Seed | Visible control PnL | Trades | Return | Hidden treatment PnL | Trades | Return | Difference |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | $531.55 | 8 | 0.5316% | $1,950.49 | 8 | 1.9505% | +1.4189 pp |
| 102 | $543.49 | 8 | 0.5435% | $76.97 | 8 | 0.0770% | -0.4665 pp |
| 103 | $7.96 | 8 | 0.0080% | $3,288.64 | 7 | 3.2886% | +3.2807 pp |
| 104 | $1.72 | 7 | 0.0017% | $1,904.46 | 7 | 1.9045% | +1.9027 pp |
| 105 | $1,714.26 | 7 | 1.7143% | $2.09 | 8 | 0.0021% | -1.7122 pp |

The mean paired difference is **+0.8847 pp** (sample SD 1.9769 pp). Its two-sided 95% Student-t interval is **[-1.5699, +3.3393] pp** (df=4), which includes zero. The supplemental paired-seed bootstrap interval is [-0.7401, +2.3572] pp using 10,000 paired-seed resamples and RNG seed 42; the exact two-sided paired sign-flip test gives p=0.4375. With five seeds, this test has coarse attainable p-values and assumes symmetry of paired differences under its null. This comparison does not establish that hiding the lifecycle state improves returns. Because the 2026 interval was inspected previously, these results are exploratory and cannot support a fresh confirmatory or locked-holdout claim. See the [provenance-complete P7 report](../../outputs/gpu_campaigns/rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1/reports/p7_paired_seed_results_exploratory_20261022.md) (SHA-256 `d200b77b9db83fcde1a76714587a01dace5f79382850a98b5d52a2f26abcdd59`) and [statistics JSON](../../outputs/gpu_campaigns/rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1/reports/p7_paired_seed_statistics_exploratory_20261022_v2.json) (SHA-256 `50b8f6b19b85abc0e417bd6ebde0ada9d1261d8233cfdb547bb92477657c6f29`). The corresponding metrics payload SHA-256 is `2c206a0f9c8de691bbe902ed4c6767699a512c5551e2af22f19bb26efeaac136`.

### Historical selected-pair execution-cost sensitivity (P7; superseded entry rules)

The same five checkpoints per arm were re-evaluated at effective tick-fill slippage of 5.0 points and 10.0 points. Only the adverse executed side changes after tick selection; tick/time/latency selection, opposite quotes, action mode, data, checkpoint, and position-sizing reserve remain fixed. The 5-point baseline here is applied to evaluation fills even though P6/P7 training used raw selected tick quotes. The 2026 report interval was previously inspected, so these are exploratory sensitivity results, not fresh out-of-sample evidence.

| Arm | Mean 10-point minus 5-point return | Paired-seed Student-t 95% CI (df=4) | Paired-seed bootstrap 95% CI | Exact sign-flip p (Holm) |
|---|---:|---:|---:|---:|
| Visible-state control | -0.7830 pp | [-1.7623, +0.1963] pp | [-1.4028, -0.1632] pp | 0.0625 (0.0625) |
| Hidden-state treatment | -0.7353 pp | [-1.5275, +0.0570] pp | [-1.2375, -0.2474] pp | 0.0625 (0.0625) |

The mean cost-stress difference is negative in both arms, and all five seed-level differences are negative in each arm. With only five paired seeds, the exact test is discrete; p=0.0625 does not cross a 0.05 threshold. The Student-t intervals include zero, while the paired-seed bootstrap intervals do not. These intervals resample training seeds and do not measure uncertainty across market periods or temporal paths. The divergence between interval methods, the small seed count, and low trade counts counsel against calling either arm robust or declaring a statistically established cost effect.

| Seed | Control 5-point return | Control 10-point return | Difference | Trades 5 / 10 | Treatment 5-point return | Treatment 10-point return | Difference | Trades 5 / 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | +0.131244% | -0.627920% | -0.759165 pp | 8 / 8 | +1.175005% | -0.354295% | -1.529299 pp | 8 / 8 |
| 102 | +0.112123% | -1.658986% | -1.771108 pp | 8 / 8 | -0.006110% | -0.350668% | -0.344558 pp | 8 / 8 |
| 103 | +0.002044% | -0.024893% | -0.026938 pp | 8 / 8 | +2.738891% | +2.191543% | -0.547348 pp | 7 / 7 |
| 104 | +0.001048% | -0.000331% | -0.001378 pp | 7 / 7 | +1.690109% | +0.435291% | -1.254818 pp | 7 / 7 |
| 105 | +1.055341% | -0.301011% | -1.356353 pp | 7 / 7 | +0.001588% | +0.001257% | -0.000331 pp | 8 / 8 |

The replay ledger contains 24/24 expected records and 240 artifact hashes. Exact repeated seed-101 runs matched for each arm at each cost level. Campaign fingerprint: `10e75ab7a25a53b3d4444098185fe45603811f09f2d556265c9f80ca402388ab`; source paired-payload SHA-256: `2c206a0f9c8de691bbe902ed4c6767699a512c5551e2af22f19bb26efeaac136`; replay-ledger SHA-256: `f1c0d2344494de8f1f76f69140c9214d2c6c027bfc033af5cda894c0dbf78549`. The [numerical replay report](../../outputs/gpu_campaigns/rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1/reports/p7_tick_cost_stress_20261022/p7_tick_cost_stress_results_20261022.md) SHA-256 is `badbbb68220a7b2316b643f064d4d73364e945fe83286ce10cd1e25ffe0f0fb3`; the [control metrics JSON](../../outputs/gpu_campaigns/rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1/reports/p7_tick_cost_stress_20261022/p7_cost_stress_control_seed101_105_20261022_metrics.json) SHA-256 is `0a5fd7bf999442f838b0308dc72302de29d9c5cf9c5ab0e29c6109c346e9aae2`, and the [treatment metrics JSON](../../outputs/gpu_campaigns/rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1/reports/p7_tick_cost_stress_20261022/p7_cost_stress_treatment_seed101_105_20261022_metrics.json) SHA-256 is `94ebe67585612913bf67bda397edd77da316299bbeb34e82d098383cdd2b632b`. The control statistics JSON SHA-256 is `22678da665a2cb86e73641abb4bc34d5495bda2aa402d86e72c08b495342cf02`, and the treatment statistics JSON SHA-256 is `b995552f4e3b88c495f679209049f1fe1dce34e45128346248c941aa8fc9f489`. The exact-repeat and artifact-hash audit passed. Full per-run provenance and effective cost identities are in the linked report and its machine-readable inputs.


## 7. Limitations

The available 2026 interval is exposed and cannot establish untouched out-of-sample performance. The configured date-level schedule reaches policy inputs from the first bar. In the audited 2025 training interval, all materially shorter observed NY sessions overlap configured dates; several near-full sessions with a few fewer than 391 minute rows remain unclassified, and the 2026 report interval has no separate per-date shape audit. Calendar completeness for every provider-specific reduced-hours or data-gap date is therefore not claimed. The selected-pair evaluation cost stress is complete: both arm means declined at 10 points, but each five-seed Student-t interval includes zero and the exact p-value is 0.0625. The paired-seed bootstrap intervals exclude zero, so the method-dependent uncertainty result is reported without treating it as a definitive effect. Low trade counts and prior inspection of the report interval further limit interpretation. P6/P7 training used raw tick quotes without the extra slippage; the stress therefore measures evaluation sensitivity rather than training-time cost exposure. The simulator does not model queue position, market impact, every broker execution effect, or live trading performance. The data feed and contract details also limit transfer to other markets.

## 8. Discussion and current conclusion

**Historical conclusion only:** the interpretation below summarizes the prior implementation and must not be treated as the conclusion for the corrected entry rules. Replace it after P6/P7 evidence is refreshed.

The experiment has not shown a reliable return advantage from hiding the entry-lifecycle state. The five paired differences have mixed signs, and the positive mean of +0.8847 percentage points is imprecise: its 95% Student-t interval spans negative and positive effects, and the exact paired sign-flip p-value is 0.4375. Report-period activity was low (seven or eight trades per arm and seed), so the result is sensitive to a small number of simulated trades. It remains descriptive evidence conditional on this market interval and simulator, not a causal or general profitability claim.

The seed-50 scratch pilot shows that both selected policies can complete the training year with trades and reconciled reward attribution, but their negative cumulative Differential-Sharpe rewards do not indicate strong risk-adjusted learning. The 18-variant screen records a development history and retained gate failures; its warm-start outcomes do not estimate the selected intervention. Cost-stress replays show lower returns at 10 points in every seed for both arms, but five pairs, low activity, exposed data, and interval-method disagreement leave the strength and generality of that sensitivity uncertain. The evidence identities for these claims are frozen in this report and its linked provenance artifacts.

## References

- Allen, C., Kirtland, A., Tao, R. Y., Lobel, S., Scott, D., Petrocelli, N., Gottesman, O., Parr, R., Littman, M. L., & Konidaris, G. (2024). [Mitigating partial observability in sequential decision processes via the λ-discrepancy](https://doi.org/10.52202/079017-2011). *Advances in Neural Information Processing Systems, 37*.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). [An empirical evaluation of generic convolutional and recurrent networks for sequence modeling](https://arxiv.org/abs/1803.01271). arXiv:1803.01271.
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). [The probability of backtest overfitting](https://doi.org/10.21314/jcf.2016.322). *Journal of Computational Finance, 20*(4), 39–69.
- Moody, J. E., & Saffell, M. (1998). [Reinforcement learning for trading](https://proceedings.neurips.cc/paper_files/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf). *Advances in Neural Information Processing Systems, 11*.
- Nguyen, H. H., Baisero, A., Wang, D., Amato, C., & Platt, R. (2023). [Leveraging fully observable policies for learning under partial observability](https://proceedings.mlr.press/v205/nguyen23a.html). *Proceedings of the 6th Conference on Robot Learning*, 205, 1673–1683.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal policy optimization algorithms](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
- Yang, H., Liu, X.-Y., Zhong, S., & Walid, A. (2020). [Deep reinforcement learning for automated stock trading: An ensemble strategy](https://doi.org/10.1145/3383455.3422540). In *Proceedings of the First ACM International Conference on AI in Finance*.
- Zhang, Z., Zohren, S., & Roberts, S. (2020). [Deep reinforcement learning for trading](https://doi.org/10.3905/jfds.2020.1.030). *The Journal of Financial Data Science, 2*(2), 25–40.

## 9. Historical report audit and remaining work

- Historical selected-pair cost-stress ledger, 240 artifact hashes, four exact repeats, numerical report, and statistics JSON identities are retained above; rerun selected-pair stress only after corrected P7 checkpoints exist.
- Historical 18-variant, P6, and P7 audit counts and hashes apply only to the prior implementation. Reconcile corrected P6/P7 artifacts against their manifests and evidence after those runs finish.
- Keep the report self-contained in its explanation and tables; link to exact machine-readable evidence identities and hashes for reproducibility.
- P9 evidence freeze is reopened. Freeze corrected campaign manifests, checkpoints, metrics, reports, source/data/config provenance, and claim limits only after the corrected P6/P7 gates pass.
- This is not yet the final results deliverable. Refresh the P6/P7 sections and conclusion, then record the final report SHA-256 in the [execution plan](master_project_execution_plan.md).
