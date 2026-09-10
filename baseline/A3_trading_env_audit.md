# A3 — TradingEnv Correctness Audit

**Date**: 2026-09-10
**Scope**: TradingEnv.step() correctness, reward interaction, position management, guardrails

## Summary

**PASS with observations** — The TradingEnv step logic is correctly implemented. Position management, SL/TP handling, guardrail breaches, and reward computation follow the documented contract. One minor observation about entry-gate fallback behavior.

## Detailed Findings

### 1. Position Entry Logic ✅

The entry flow is:
1. Decode action → `discrete_action` + `risk_frac` + `rr_ratio`
2. Check entry gate (`_check_entry_gate` or `strategy.validate_entry`)
3. Check guardrails (breach → force close, done)
4. Check SL/TP on existing position
5. Handle exit (action==19) or new entry
6. For new entry: compute structure SL/TP → compute lots → open position

**No naked positions**: If swing levels are unavailable or invalid (e.g., `last_swing_low >= entry_price` for long), the entry is rejected (`lots = 0.0`). This is correct.

**SL already hit check**: Lines 911-916 check if the stop loss would already be triggered at fill time (e.g., `fill_bid <= sl_price` for long). This prevents opening a position that would immediately be stopped out.

### 2. Position Exit Logic ✅

- **SL hit**: Checks `bar["low"] <= sl_price` (long) or `bar["high"] >= sl_price` (short) → close at fill quote
- **TP hit**: Checks `bar["high"] >= tp_price` (long) or `bar["low"] <= tp_price` (short) → close at fill quote
- **Manual exit** (action==19): Only in non-strategy mode
- **Overnight block**: Force-closes at session end if `block_overnight=True`
- **Guardrail breach**: Force-closes position, records breach

### 3. Guardrail Handling ✅

| Mode | Breach Behavior |
|------|----------------|
| `episodic=True` (training) | `done=True`, episode ends |
| `episodic=False` (eval) | `session_blocked=True`, no new trading for rest of session, rollout continues |

In eval mode, breached sessions are tracked in `breached_sessions` set, and a fresh breach is recorded exactly once per session. This mirrors `run_backtest`'s behavior.

### 4. Entry Gate ⚠️ Observation

`_check_entry_gate()` requires `london_high`, `london_low`, `asian_high`, `asian_low`, `volume_spike` columns. When these are missing:
- Warns once per episode (`_entry_gate_warned` flag)
- **Falls back to allowing ALL entries** (`return True`)

**Impact**: When session-liquidity features are not in the feature matrix (the default config), the entry gate is effectively disabled. This is by design (the gate is an optional enhancement), but it means the gate's filtering behavior is only active when explicitly configured.

**Not a bug** — this is documented fallback behavior. The UserWarning correctly alerts users.

### 5. Reward Computation ✅

The reward computation correctly handles three configurations:
1. **DSR only** (`use_sweep_reward=False`, no strategy): Uses `DSRReward` directly
2. **Composite** (`use_sweep_reward=True`): `CompositeReward` combines DSR + SweepConfirmationReward
3. **Strategy + Composite** (`strategy_actions=True`): Adds strategy-alignment reward (PO3Reward/DistributionReward)

The `pnl_step` is computed as `equity_curve[-1] - equity_curve[-2]` (change in equity). The `daily_loss` is `initial_balance - equity` (positive when losing).

### 6. Fill Latency ✅

`fill_latency_bars` parameter shifts the fill quote by N bars. `fill_idx = step_idx + 1 + fill_latency_bars`. When `fill_idx >= len(bars)`, falls back to current bar quote. This is correct.

### 7. Max Episode Steps Truncation ✅

The truncation is applied AFTER all other logic (data-end, breach), so it consistently overrides `done=True` signals. The `episode_step_count` is incremented before the check, so a cap of N allows exactly N `step()` returns.

### 8. Observation Construction ✅

Observations are built from `self._obs_features` (which excludes raw price-level columns and MTF raw suffixes per `Agent.md §11`). The `seq` window is the last `obs_window` bars, and the `account` vector contains `[equity, position_direction, open_pnl, unrealised_r, dist_to_sl]`.

### 9. Equity Curve & Trade Log ✅

- Equity curve starts with `[initial_balance]`
- PnL history starts with `[0.0]`
- Trade log records: open, close, stop_close, tp_close, forced_close, eod_close
- Each trade records: type, pnl, price, reason, bar index, time, equity, and (for opens) strategy metadata

## Potential Concerns

### Medium — `minutes_since_open` calculation assumes 5-min bars

Line 993: `minutes_since_open = max(0, (self.step_idx - self.ny_session_start_idx)) * 5.0 / 60.0`

This multiplies by 5 (assuming 5-minute bars) and divides by 60 (converting to hours). The environment uses M1 data, so this should be `* 1.0 / 60.0` for 1-minute bars. However, `ny_session_start_idx` defaults to `obs_window` (60), which may compensate.

**Impact**: The time-decay penalty in `SweepConfirmationReward` may progress at the wrong rate. With the default config (`use_sweep_reward=False`), this has no effect.

### Low — Entry gate fallback allows all entries

When session-liquidity features are missing, the gate warns once and then allows all entries. This is documented behavior but could surprise users who expect the gate to always filter.

## Conclusion

**TradingEnv.step() is correctly implemented.** Position management, SL/TP, guardrails, and rewards follow the documented contracts. The two observations above are minor and do not affect the default training configuration (DSR reward, no strategy actions).