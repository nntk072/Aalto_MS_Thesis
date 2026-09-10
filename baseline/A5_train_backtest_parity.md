# A5 — Train/Backtest Parity Audit

**Date**: 2026-09-10
**Scope**: Verify training environment and evaluation/backtest produce consistent results

## Summary

**PASS with documented differences** — Training and evaluation use the **same** `TradingEnv` class (differing only in `episodic` mode), ensuring parity. The `run_backtest` engine is a **separate implementation** for rule-based policies and is NOT used for RL evaluation — this is by design.

## Architecture

### Training → Evaluation Parity ✅

| Component | Training | Evaluation | Match? |
|-----------|----------|------------|--------|
| Environment class | `TradingEnv` | `TradingEnv` | ✅ Same |
| `episodic` mode | `True` | `False` | ⚠️ Different |
| Observation format | Dict{"seq", "account"} | Dict{"seq", "account"} | ✅ Same |
| Action space | Discrete(20) or Box(-1,1) | Same | ✅ Same |
| SL/TP logic | Structure-based | Structure-based | ✅ Same |
| Guardrail logic | Ends episode on breach | Blocks session on breach | ⚠️ Different semantics |
| Entry gate | Active (if features present) | Active (if features present) | ✅ Same |
| Lot sizing | `compute_lots` with risk_frac | `compute_lots` with risk_frac | ✅ Same |

**Conclusion**: Training and evaluation ARE parity. The only difference is `episodic` mode, which affects breach handling but not trade execution logic.

### Training → Backtest Differences (By Design)

`run_backtest()` is a **separate, lightweight engine** for rule-based policies (e.g., BuyAndHold, EMACross). It is NOT used for RL evaluation. This is explicitly documented in `rollout.py` (lines 1-19):

> "quant_rl.backtest.engine.run_backtest drives a rule-based policy callable (plain np.ndarray in, int action out) through a lightweight event loop. The RL model, however, was trained against TradingEnv's own Dict observation and Discrete(20) action space — those formats don't match run_backtest's interface at all."

| Feature | TradingEnv (RL) | run_backtest (Rule-based) |
|---------|-----------------|---------------------------|
| Observation | Dict{"seq", "account"} | Raw ndarray window |
| Action format | Discrete(20) or Box(-1,1) | {-1, 0, 1} |
| Entry gate | Yes (if features present) | No |
| Lot sizing | compute_lots with risk_frac | Fixed `lots` parameter |
| Position sizing | 3×3 risk/rr grid (actions 1-18) | Single risk_frac |
| SL/TP | Structure-based with buffer | Structure-based OR global USD cap |
| Breach handling | Episodic or session-block | Session-block only |

**Conclusion**: The differences are by design. `run_backtest` is for baselines; `TradingEnv` is for RL.

## Verification: evaluate_model() Uses TradingEnv

From `train_rl.py` (lines 324-340):
```python
test_result = evaluate_model(
    model,
    bars=test_bars,
    features=test_feat,
    obs_window=cfg.env.obs_window,
    ...
)
```

And `evaluate_model()` in `rollout.py` (lines 99-100):
```python
env = TradingEnv(
    bars=bars,
    features=features,
    ...
    episodic=False,
)
```

This confirms: **RL evaluation uses TradingEnv, not run_backtest**.

## Potential Concerns

### Medium — Breach semantics differ between train and eval

- **Training** (`episodic=True`): Breach → `done=True`, episode ends immediately
- **Evaluation** (`episodic=False`): Breach → blocks trading for session, continues to next session

**Impact**: The agent trains to avoid breaches (since they end the episode), but evaluation allows continued trading after a breach. This is the intended behavior — eval measures performance over the full test set regardless of breaches.

**Not a bug**: The `episodic=False` mode is specifically designed so a single breach doesn't terminate the entire evaluation rollout (see `rollout.py` docstring).

### Low — Entry gate not in run_backtest

`run_backtest` doesn't implement the entry gate logic. When comparing RL agents (which train with the gate) against rule-based baselines (which don't have it), the baselines have an advantage of being able to enter without gate constraints.

**Impact**: Minor asymmetry in baseline comparison. The gate is disabled by default (features missing → fallback allows all entries), so this only affects configurations where session-liquidity features are explicitly enabled.

## Conclusion

Training and evaluation maintain parity through the shared `TradingEnv` class. The `run_backtest` engine is intentionally separate and used only for rule-based baselines. No parity bugs detected.