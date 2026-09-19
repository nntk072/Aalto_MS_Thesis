# Thesis Audit 2026 — Claim-Affecting Findings

Re-verification of Startup plan 01 findings **F3–F8** and **F10** against
`Aalto_MS_Thesis` source on branch `feature/thesis-plan-validity`. Status values:
`CONFIRMED` | `REFUTED` | `PARTIAL`.

Optional context (not T02 blockers): **F1** short history, **F9** multi-simulator.

| # | Status | Claim impact | Citation |
|---|--------|--------------|----------|
| **F3** | CONFIRMED | Session / PD labels use fixed `Etc/GMT-3`. A ±1 h broker DST shift would relabel Asia/London/NY windows and poison Idea-1 / PD-context features. CT session-OHLC path is DST-aware separately. | `quant_rl/config/default.yaml` L7, L24–28 (`dst_flag` only); `quant_rl/data/session.py` (default tz); `quant_rl/features/pd_context.py` / `build.py` PD tz fallback |
| **F4** | CONFIRMED | Ablation and OOS scripts rescore the same locked holdout (`train_end` 2025-12-31 / `test_start` 2026-01-01). Using that slice for architecture selection is selection bias. | `quant_rl/config/default.yaml` L12–14; `config/experiments.yaml`; `scripts/ablation_runner.py`; `scripts/test_oos.py` |
| **F5** | CONFIRMED (pre-fix) | Legacy `purged_walk_forward` used `step < test_len`, so adjacent test folds overlapped (~40 bars for n=1000, 5 splits, test_size=0.2). Purge was a fixed bar count, not label-horizon-aware. Mitigated by T-03.1 on this branch. | `quant_rl/evaluation/walkforward.py`; historical behaviour retained as `purged_walk_forward_legacy` |
| **F6** | CONFIRMED (pre-fix) | Cache early-return loaded parquet if the path existed; key was `{symbol}_features_{FEATURE_CACHE_VERSION}.parquet` with hand-bumped version only — feature flag flips could reuse stale files. Mitigated by T-02.1 content hash. | `quant_rl/features/build.py` (~L313–314 pre-fix); `train_rl.py` cache path construction |
| **F7** | CONFIRMED (pre-fix) | `rolling_zscore(..., train_mask=)` existed but `train_rl` called `build_features` without `train_mask`, so z-score stats used the full series. Adequate for one causal rolling window on a single split; unsafe for honest WF refits. Mitigated by T-02.4. | `quant_rl/features/normalize.py`; `quant_rl/train/train_rl.py` (~L290); `baseline/A2_split_audit.md` |
| **F8** | PARTIAL | Truncation causality exists for RSI/MACD/EMA/returns and for several structure/PO3/PD/HTF **blocks**, but there was no harness over the **assembled** `build_features` matrix. Mitigated by T-02.2 assembled-matrix test. | `tests/test_causal_features.py`; `tests/test_features/test_variants.py`, `test_pd_context.py`, `test_htf_alignment.py`; new assembled harness |
| **F10** | CONFIRMED | Defaults are optimistic (`slippage_points` often 0; constant spreads). Thesis tables must report cost stress, not default-cost Sharpe alone. | `quant_rl/config/default.yaml` `costs:`; `scripts/test_oos.py` grid |

## F1 / F9 (limitations)

| # | Status | Note |
|---|--------|------|
| **F1** | CONFIRMED | ~18 months, US100/US500 MT5 CSV — no multi-regime / crisis coverage. Accepted MSc limitation. |
| **F9** | CONFIRMED | `TradingEnv`, `run_backtest`, backtrader CV, and MT5 live can diverge. Thesis eval pack must name which engine produced each table. |

## Verification method

Code citations from repository read of the paths above; Startup plan 01 used as the finding catalogue only.
