# Implementation audit ledger — 2026-10-12

Repository: `/scratch/work/nguyenl37/Aalto_MS_Thesis` on `code.triton.aalto.fi`.
Branch: `main`; HEAD: `b00b5e85622ec1410c8a11cc063cd52c87d3976d`.
This document records the **pre-fix** state. The authoritative plan is `/scratch/work/nguyenl37/Agent_Implementation_Plan.md` and remains unchanged.

## Baseline and preservation

- Before agent edits: **22 pre-existing modified tracked paths**, **five pre-existing untracked paths** (`config/campaigns/18_variant_seed42_20m.yaml`, `docs/operations/triton_environment.md`, `docs/operations/triton_mypy_lustre.md`, `docs/training/18_variant_training_readiness_plan.md`, `docs/training/triton_evaluation_acceleration_agent_plan.md`); index clean. These changes must be preserved. The subsequent `tests/test_wp_correctness.py` is agent-created.
- Instructions examined: `AGENTS.md`, relevant `.agents/rules/` instructions and `.agents/domain_maps/` for environment, evaluation, training, features, backtest, models. `.venv-x86/bin/python` is the compatible x86 login-node interpreter; `.venv` is ARM and unsuitable here. No GPU allocation was opened.
- Observed historical artifacts: existing `outputs/defense_figures`, `outputs/feature_verification`, `outputs/gpu_matrix*`, `outputs/gpu_campaigns` and seed-50 evaluation/training artifacts. Their presence is **not** proof of valid reproduced metrics or an untouched OOS holdout. Exact dataset hashes and checkpoint-provenance checks remain pending.
- Optional code-review-graph MCP unavailable in the connected tool inventory; `rg` unavailable on remote; fallback targeted `grep`/`sed` source inspection. No graph coverage claimed.
- Pre-fix scoped baseline: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_envs/test_tp_mode.py tests/test_envs/test_trading_env_golden_traces.py tests/test_evaluation/test_metrics.py tests/test_evaluation/test_calibration_ci.py -q --disable-warnings --maxfail=1`: **62 passed, one warning**, ~2.98 s.
- New full-runtime focused regressions, **pre-fix**: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-x86/bin/python -m pytest tests/test_wp_correctness.py -q --disable-warnings --maxfail=8`: **7 failed, 2 passed** (~3.61 s). Failures demonstrate F01 (2 cases), F02 (2), F06 (2), F08 (1). These targeted tests use stubs to call actual methods for F01/F02 and live Python imports; they are not execution-engine end-to-end parity tests.

## F01–F15 finding dispositions at initial HEAD

Classification reflects **pre-fix evidence**, and is intentionally separate from subsequent remediation status. Confidence H=high, M=medium, L=low.

| Finding | Classification/confidence | Source and observable evidence | Affected behaviour / required regression / limitation |
|---|---|---|---|
| F01 | **REPRODUCED/H** | `quant_rl/envs/trading_env.py:_apply_trailing_stop` (around line 3418) selects swing **high** for long, **low** for short; two mirrored full-import regressions fail. | SL protection direction; test delayed confirmation, price geometry, monotonicity, gaps, partial fills. Stubs did not reproduce time-series causality. |
| F02 | **REPRODUCED/H** | `trading_env.py:_reorder_tp_levels_safe` (around 3440) writes `p_ai` rather than modified `prices[ai]`; both long/short duplicate-price tests fail. `_apply_tp_trailing` can merge slots. | Target order and quantities; test tick/bar multi-slot and enabled/hit invariants. Stub reproduction only. |
| F03 | **SOURCE-CONFIRMED/H** | `trading_env.py:_check_sl_tp` (around 1270) tick branch closes whole size before TP-slot handling. | Premature full closure; require tickbook partial TP fill and SL-first cases. Tickbook fixture not yet executed. |
| F04 | **SOURCE-CONFIRMED/H** | `trading_env.py:_check_sl_tp` partial branch logs fills but omits `_note_close`; `_note_close` replaces scalar `_last_realized_close_pnl` and `_stamp_exit` overwrites `_pending_realized_r`. | Missing reward for partials/multiple fills; require end-to-end incremental reconciliation on skipped-bar replay. |
| F05 | **SOURCE-CONFIRMED/H** | `quant_rl/evaluation/runner.py:run_episode` counts every PnL exit row as a trade. | Position vs fill metrics; require three-leg single-position win-rate test. |
| F06 | **REPRODUCED/H** | `quant_rl/models/ppo_policy.py:_layout_key` (around 91) rejects width 9; `_MODE_INDEX['8']` assumes direction even when absent; two full-import regression failures. `models/agent.py:build_agent` uses flags. | Wrong prior coordinates and model layout; table-driven full-schema action and checkpoint tests. |
| F07 | **SOURCE-CONFIRMED/H** | `trading_env.py:_maybe_ema_exit` (around 1589) sends `(bar.close,bar.close)` to spread-aware broker; `_is_ny_session_boundary` (around 3322) misses final NY index. | Short spread/costs and terminal valuation; require both sides, session transition and terminal replay. Fill latency remains unverified. |
| F08 | **REPRODUCED/H** | `quant_rl/evaluation/bootstrap_ci.py:metrics_with_ci` (around 115) bootstraps equity **levels**, yielding positive max-drawdown CI for strictly increasing equity. | Invalid MDD uncertainty; require return-block reconstruction and first-step-loss cases. |
| F09 | **SOURCE-CONFIRMED/M** | `bootstrap_ci.py:sortino_stat` vs `evaluation/metrics.py:calculate_metrics` downside conventions differ; annualization defaults require full call-site audit. | Inconsistent reported metrics; paired/zero-trade/sharpe-sortino contract tests needed. |
| F10 | **UNVERIFIED/M** | `quant_rl/train/train_rl.py` (walk-forward assembly) and `features/build.py` have pre-existing user edits; the asserted original fold-fit flow has not been executed or conclusively traced on current code. | Potential fold-fit leakage; require disjoint-fold perturbation and fitted-state provenance tests. |
| F11 | **SOURCE-CONFIRMED/M** | `quant_rl/features/normalize.py:rolling_zscore` uses masked rolling followed by forward fill; it will drift as masked training rows leave the window. | Freeze-vs-rolling contract needs explicit scope; test fit-only state and causal warmup. Current-bar information available at decision time is not automatically leakage. |
| F12 | **SOURCE-CONFIRMED/H** | `trading_env.py:_check_entry_gate` (around 1099) permissively handles missing gate inputs, `_resolve_ny_indices` (around 3311) falls back to all bars. | Mislabelled eligible runs; add opt-in strict research-mode exceptions with explicit compatibility default. |
| F13 | **SOURCE-CONFIRMED/M (semantic decision)** | `trading_env.py:_decode_action` (around 2381) permits entry with intensity threshold zero; `_apply_tp_breakeven` (around 3469) moves nearest TP to `entry ± alpha*R`. | Legacy TP/hold meanings are **not approved for reinterpretation**; record exact design decision, avoid modifying TP modes. |
| F14 | **SOURCE-CONFIRMED/M (protocol)** | `docs/thesis/eval_pack.md`, `validation_protocol.md` and historical comparison settings describe differences beyond PO3 restriction and previously viewed OOS evidence. | Requires matched experiment and used-holdout provenance; do not present historical OOS as pristine. |
| F15 | **SOURCE-CONFIRMED/H** | `.agents/domain_maps/env.md` describes 5 account values and 4-D strategy actions, inconsistent with current environment widths. | Misleading agent guidance; update domain map only after schema verified. |

## Dependency order and validation boundary

WP01 schema → WP02 stable fill ledger/position summaries → WP03 swing SL and WP04 slot fills → WP05 clocks/costs → WP06 position metrics/return-block bootstrap → WP07 fold-scoped fit → WP08 eligibility → WP09 causal reference targets (TP mode semantics held) → WP10 matched controls → WP11 manifests/holdout gate → WP12 bounded CPU parity → WP13 documentation/handoff. This sequence is a dependency map of required work, not evidence it was executed.

Historical reported performance and any seed-50/OOS plots remain **reported artifacts only**. No training, Triton allocations, commits, pushes, or historical artifact overwrites were authorized or performed for this audit.
