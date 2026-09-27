# Dynamic Market Adaptation & RL Agent Overhaul Plan

## 1. Problem Diagnosis & Motivation

Recent 20M training rollouts across all model backbones (TCN, GRU, Transformer) have produced consistent underperformance:
- **In-sample Sharpe:** $\approx -0.71$ to $-1.86$ (Total Return: $-4.7\%$ to $-10.6\%$)
- **Out-of-sample Sharpe:** $\approx -1.53$ to $-2.57$ (Total Return: $-8.5\%$ to $-10.0\%$)
- **Win Rates:** $30.8\% - 37.8\%$
- **Equity Gate:** Fails with `end_not_above_start,slope_not_positive,peak_trailing_dd`.

### Root Causes Identified
1. **The Risk / EOD Guard Contradiction:** Sizing via `compute_lots` sizes trades to risk up to $1,000 (1% of $100k balance) at the structural stop. However, `_eod_max_loss_usd` has been hardcoded to `$500.0`. In recent runs, over **80% of total losses** were triggered prematurely by `eod_max_loss` ($-\$12,653.90$ in the test set alone) during normal adverse excursions before the structural thesis could play out.
2. **Deterministic Heuristic Chokepoint:** When `strategy_actions: true`, the RL agent is deprived of directional control ($+1/-1/0$). Direction is hardwired to `context_direction(feat_row)`. When market dynamics change (e.g. from range-bound distribution to strong momentum trends), the heuristic generates false reversals, and the agent's only recourse is to suppress trading entirely (`hold_low_intensity`).
3. **Sparse Step Noise vs. Credit Assignment:** In a 1-minute bar environment with only 60–150 trades per year, $>99.8\%$ of steps have zero PnL. The per-bar differential Sharpe ratio (`DSRReward`) experiences exponential decay artifacts ($\eta = 0.01$), rewarding inaction over good risk-adjusted setups.
4. **Time & Volatility Blindness:** The agent lacks awareness of intraday session time elapsed (e.g., minutes remaining until NY close) and local volatility regimes (Garman-Klass / Parkinson realized volatility).

---

## 2. Architecture & Design Specifications

### Component A: Unified Risk Calibration & Dynamic Breakeven
- **Harmonize Sizing and Emergency Stops:** Set `eod_risk.max_loss_usd` $\ge$ planned trade risk ($1,100, above the $1,000 max loss cap) or dynamically evaluate `max_loss_usd = max(position.planned_risk_usd * 1.1, eod_max_loss_usd)`. Sizing and cutoffs must agree so the structural stop loss remains the primary risk boundary.
- **Dynamic Breakeven / Profit Protection:** When favorable excursion reaches $+1.0 R$ (one stop distance), move the internal stop to breakeven + buffer. This prevents trades that almost reached $+3R$ from rotating into full losses or EOD cutoffs.

### Component B: Agent Direction Autonomy & Action Space
- Action space option: Allow the agent to output an intent or confirmation of direction:
  - When `direction_bias` in action is enabled:
    - Dim 0: Direction conviction $[-1, 1]$
    - Dim 1: Structural SL anchor $[0, 1]$
    - Dim 2: Risk fraction $[0, 1] \to [\text{min\_risk}, \text{max\_risk}]$
    - Dim 3: Reachable reward target fraction $[0, 1]$sa
  - When the agent direction agrees or passes a threshold, the trade is taken; if it disagrees strongly, the agent can veto bad heuristic signals.

### Component C: Regime-Aware Observation Features
- **Intraday Session Clock:** Normalized progress through the active session $[0, 1]$ (bars since session open / total session bars) so the policy knows whether there is time for a multi-hour swing before the overnight close.
- **Microstructure Volatility Estimator:** Add Parkinson / Garman-Klass realized volatility ratios over rolling windows to distinguish trending breakouts from chop.

### Component D: Hybrid Trade-Level & Step-Level Reward Shaping
- Complement the step-level DSR with trade-completion credit assignment:
  $$R_t = R_{\text{step}} + \mathbb{I}_{[\text{trade closed at } t]} \cdot \phi(\text{PnL}_{\text{trade}}, R_{\text{realized}})$$
- Avoid penalizing normal in-trade adverse excursion within the planned structural stop distance.

---

## 3. Implementation Roadmap

1. **Phase 1: Risk & Execution Realignment**
   - Update `quant_rl/config/default.yaml` `eod_risk.max_loss_usd` to 1100.0.
   - Update `quant_rl/envs/trading_env.py` `_apply_position_guards` to ensure adverse excursion threshold respects the position's planned dollar risk.
   - Add dynamic breakeven option to `trading_env.py`.

2. **Phase 2: Observation & Feature Enrichment**
   - Add session progress (`session_progress`) and Parkinson / realized volatility metrics in `quant_rl/features/`.
   - Wire them into observation vectors in `trading_env.py`.

3. **Phase 3: Policy Direction Autonomy**
   - Allow configurable directional control in `TradingEnv._decode_action`.
   - Update tests to ensure backward compatibility and new action mechanics.

4. **Phase 4: Reward Calibration**
   - Enhance reward function with trade-completion PnL feedback.

5. **Phase 5: Verification**
   - Run unit test suite and smoke training.

---

## Implementation Status

| Phase | Item | State | Where |
|-------|------|-------|-------|
| 1 | `eod_risk.max_loss_usd` 500 → 1100 | **done** | `quant_rl/config/default.yaml` |
| 1 | Adverse cut floored by the trade's own planned risk | **done** | `TradingEnv._eod_adverse_cut_usd` |
| 1 | `Position.planned_risk_usd` recorded at entry | **done** | `broker.py`, `TradingEnv._try_enter_position` |
| 1 | Breakeven stop rewrite at +1R | **done** | `TradingEnv._apply_breakeven` |
| 2 | `session_progress` / `po3_phase` (already built) | **already existed** | `features/build.py::build_po3_phase_features` |
| 2 | `realized_vol` (already in base indicators) | **already existed** | `features/indicators.py` |
| 2 | Bar spacing read from the index, not assumed | **done** | `TradingEnv._minutes_per_step` |
| 2 | `ny_session_start_idx` defaults to the first NY bar | **done** | `TradingEnv.__init__` |
| 3 | Opt-in 5-D signed-direction action space | **done** | `TradingEnv._apply_direction_control` |
| 3 | Config plumbed to train / eval / walk-forward | **done** | `train_rl.py`, `eval_run.py`, `eval/rollout.py` |
| 4 | Trade-level reward credit assignment | **designed, not implemented** | see below |

### Phase 1 — what changed and why

The `$500` intraday cut and the `$1,000` per-trade sizing budget were the same
risk expressed twice, in conflict. `compute_lots` solves lot size so the
*structural stop* costs `risk_per_trade_limit`; `_eod_max_loss_usd` then flattened
the position at half that. The cut is now a **floor, not the primary boundary**:

```python
cut = max(configured_max_loss_usd, planned_risk_mult * planned_risk_usd)
```

so a position can never be closed by this guard *inside* its own stop. The
structural stop remains the real risk boundary. `respect_planned_risk: false`
restores the old behaviour for ablation.

Breakeven then protects the runner: once favorable excursion reaches
`breakeven_trigger_r` stop distances, the stop tightens to entry +/- buffer and
only ever moves toward entry, never back out.

### Phase 2 — a real bug found on the way

`minutes_since_open` was hard-coded as `steps * 5 / 60`, but the environment
steps the **M1** spine. The sweep reward's time-decay term is linear in that
value (`t_t = max(0, minutes - 20)`), so at 20 minutes into the session the
agent already saw a penalty sized as if 100 minutes had passed, and by the
close the term outweighed the PnL component entirely. Bar spacing is now read
from the DatetimeIndex, and the session-start index defaults to the first real
NY bar rather than `obs_window`.

### Phase 3 — agent direction control

`env.agent_direction_control: true` switches the overlay from 4-D to 5-D. Dim 0
becomes signed conviction, so the agent can veto or invert a heuristic side
that is failing in the current regime. `direction_override_threshold` sets the
band around neutral within which the heuristic still governs; at exactly
neutral (PPO's deterministic mean) the agent defers by design. **Default
`false`** — the 4-D layout and the frozen 20M campaign configs are unchanged.

### Phase 4 — reward, still open

`DSRReward` is computed per M1 step where >99% of steps have zero PnL, so the
EMA estimates (`eta=0.01`) decay and the gradient is dominated by flat regions.
The intended fix is explicit credit assignment at trade close:

```
R_t = R_step + I[trade closed at t] * phi(realized PnL, realized R, close reason)
```

with `close_reason` already available on the step `info` dict (added in
`reward_parts` / `close_reason`). Not implemented: it changes every arm's
reward and must be re-baselined before it can be attributed.

---

## Guardrails for the Next Stage

- **Do not read locked OOS Sharpe while selecting.** Per
  `docs/thesis/validation_protocol.md`, `test_start: 2026-01-01` is final
  report only; use `train_rl --walk-forward` for selection.
- **Ablate one change at a time.** `respect_planned_risk`, `breakeven_trigger_r`,
  and `agent_direction_control` are all independent switches. Ranking them on
  one 20M run is exactly the multiple-trials problem the deflated Sharpe in
  `outputs/rescore_sl_fill/deflated_sharpe.json` already penalises.
- **The `eod_max_loss` share is the acceptance metric.** Before: ~80% of test
  losses came from that guard. After the fix, a correct run should show the
  mass moving to `structure_sl` and `structure_tp`, and the mean trade PnL
  should stop being dominated by a fixed `-$450` cluster.

---

## Verification Record

CI bar mirrors `scripts/run_full_tests.sh` — `ruff format --check .`,
`ruff check .`, `mypy .`, and **`pytest tests/` with no marker filter**
(the `slow`-marked training tests included). Run on Triton in the reused
`quant-rl-train:gpu-shell` H200 allocation (x86_64, `.venv-x86`, CUDA True):

| Stage | Result |
|-------|--------|
| `ruff format --check .` | 0 |
| `ruff check .` | 0 |
| `mypy .` | 0 (301 files) |
| `pytest tests/` | **913 passed**, 0 failed |

### Two bugs found and fixed while closing the bar

**1. `_extract_window` leaked the overnight gap.** The trailing `context`
was clamped to the *calendar* day, so an `eod_close` at 23:00 pulled in 59
dead candles from 23:01–23:59 where the market is closed. Now clamped to the
*session* end (`_session_end_pos`), with `max(..., i_c)` so an overnight-hold
exit keeps its own exit bar. Post-exit context for mid-session trades is
unchanged — locked in by two new tests.

**2. The sweep reward's time decay was unbounded (this is what made PPO NaN).**
`T_t = max(0, minutes_since_open - 20)` grew without limit with the episode,
while `ΔPnL_t` is a few dollars. On the `test_scripts_smoke` fixture
(3000 tz-naive **hourly** bars, 60 min/step) a single step reached
`β·T_t ≈ −1324`, giving a **one-step reward of −445** and driving the value
function to NaN — `Categorical(logits)` all-NaN, `test_train_rl` failing.
`T_t` now saturates at `decay_horizon_min` (default `alpha/beta`), so the decay
can at most cancel the confirmation bonus it penalises. Same fixture: the
one-step reward is **−0.07**.

> This is a genuine reward-design defect, not just a test artefact: any run
> whose episode is long relative to `alpha/beta` gets a reward dominated by a
> clock term. It interacts with the Phase 2 `minutes_since_open` fix — bar
> spacing multiplies the same unbounded term. **Re-baseline the sweep arm.**

Tests added: 19 in `test_risk_alignment.py`, 5 in `test_ny_steps_eod.py`,
3 in `test_sweep_reward.py`, 2 in `test_plots_export.py`.



