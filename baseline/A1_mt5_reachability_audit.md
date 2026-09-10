# A1 — mt5_trading Reachability Audit

**Date**: 2026-09-10
**Scope**: Determine if mt5_trading is used by external tooling or only by live trading scripts

## Summary

**IGNORE for refactoring** — `mt5_trading` is only used for live/paper trading with the MT5 broker. It is NOT used by the training, evaluation, or backtesting pipeline. It is correctly isolated as an optional external interface.

## Findings

### Usage Map

| Caller | File | Usage |
|--------|------|-------|
| Live trading | `live_trading_rl.py` | Imports RLRobot, MT5Trader, RiskManager for live execution |
| RL adapter | `quant_rl/live/rl_strategy.py` | Lazy import of mt5_trading adapters (line 17: "imports mt5_trading lazily so the RL stack remains importable without MT5") |
| Live trading | `live_trading.py` | Imports MT5Trader, MT5Data, MultiSymbolManager |
| Tests | `tests/test_live/test_rl_robot.py` | Tests RLRobot (mt5_trading.robot.rl_robot) |
| Tests | `tests/test_live/test_rl_bridge.py` | Tests mt5_trading.domain.signal integration |

### Isolation

The `mt5_trading` package is correctly isolated:
- **Training** (`quant_rl/train/train_rl.py`): Does NOT import mt5_trading
- **Evaluation** (`quant_rl/eval/rollout.py`): Does NOT import mt5_trading
- **Backtest** (`quant_rl/backtest/engine.py`): Does NOT import mt5_trading
- **Features** (`quant_rl/features/build.py`): Does NOT import mt5_trading
- **Environment** (`quant_rl/envs/trading_env.py`): Does NOT import mt5_trading

The `quant_rl/live/rl_strategy.py` module uses lazy imports:
```python
# This module imports ``mt5_trading`` lazily so the RL stack remains
# importable without MT5.
```

### Conclusion

`mt5_trading` is a self-contained package for live/paper MT5 broker interaction. It has no coupling to the RL training/evaluation stack beyond the lazy adapter in `quant_rl/live/rl_strategy.py`. Refactoring `quant_rl` does not affect `mt5_trading`, and vice versa.

**Recommendation**: No refactoring needed for `mt5_trading`. It is correctly architected as an optional external interface.