# A4 — Reward Source + Reward Audit

**Date**: 2026-09-10
**Scope**: Identify authoritative reward implementation, verify reward correctness

## Summary

**PASS** — The reward system is correctly implemented with a clear hierarchy. The authoritative reward for the baseline configuration is `DSRReward` (Differential Sharpe Ratio). Alternative rewards (SweepConfirmationReward, PO3Reward, DistributionReward) are opt-in via CLI flags.

## Reward Architecture

### Reward Selection Logic (`trading_env.py` lines 212-227)

```python
if use_sweep_reward or (strategy_actions and strategy_reward is not None):
    # CompositeReward combining DSR + SweepConfirmationReward + optional strategy reward
else:
    self.reward_fn = DSRReward(eta=dsr_eta)  # Default
```

### Reward Implementations

| Reward | File | When Used | Type |
|--------|------|-----------|------|
| DSRReward | `envs/reward.py` | Default (`--reward dsr`) | Continuous (per-step) |
| SweepConfirmationReward | `envs/sweep_reward.py` | `--reward sweep` | Continuous (per-step) |
| CompositeReward | `envs/sweep_reward.py` | `--reward sweep` or strategy actions | Combination |
| PO3Reward | `envs/po3_reward.py` | `--strategy po3_ifvg` | Event-based (entry only) |
| DistributionReward | `envs/distribution_reward.py` | `--strategy distribution` | Event-based (entry only) |

### Authoritative Reward: DSRReward

The default training configuration (`--reward dsr`, no strategy) uses `DSRReward`:

```python
# Differential Sharpe Ratio with FTMO soft penalties
# Reward = dS_t / dF_t · ΔF_t  (first-order approximation)
# Plus: soft penalty for FTMO daily-loss proximity
# Plus: hard breach → terminal negative reward (-1.0)
```

**Implementation correctness**:
- EMA estimates A (mean return) and B (mean squared return) are updated online
- DSR formula: `(B*(r - A_prev) - 0.5*A*(r² - B_prev)) / (denom^1.5)` where `denom = B - A²`
- Clipping to [-10, 10] prevents extreme values
- Soft FTMO penalty linearly ramps in the last 20% of the daily loss limit
- Breach returns -1.0 (terminal negative reward)

### CompositeReward Correctness

The `CompositeReward` correctly:
1. Maintains persistent DSR state (doesn't reset per step)
2. Computes DSR component via `self._dsr_fn(pnl_step, ...)`
3. Adds sweep component when sweep parameters are provided
4. Adds strategy-alignment component when `strategy_reward` is configured
5. Weights: `total = dsr_weight * dsr + sweep_weight * sweep + strategy_weight * strategy`

### Strategy Rewards (Event-Based)

**PO3Reward**: Rewards entering inside a confirmed IFVG zone during distribution, penalizes entering during manipulation or outside IFVG.

**DistributionReward**: Rewards entering after a directionally consistent sweep→BOS chain (long: sweep_low + BOS_up; short: sweep_high + BOS_down).

Both are stateless and only fire on position entry transitions (`position_changed=True, direction≠0`).

## Configuration Defaults

From `config/default.yaml`:
- `env.reward_dsr_eta: 0.01` (DSR damping)
- `env.use_sweep_reward: false` (sweep reward disabled by default)
- `env.strategy_actions: false` (strategy rewards disabled by default)

## Reward Source Resolution

**Answer to Open Question #2**: The authoritative reward for the baseline (default) configuration is `DSRReward` from `quant_rl/envs/reward.py`. `SweepConfirmationReward` is only used when `--reward sweep` is explicitly passed. `PO3Reward` and `DistributionReward` are strategy-specific alignment rewards.

## Conclusion

The reward system is correctly implemented with a clear selection mechanism. The default DSRReward follows the standard differential Sharpe ratio formulation with FTMO penalties. The composite and strategy rewards are correctly layered on top when enabled.