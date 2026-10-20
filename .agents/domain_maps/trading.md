# Trading Domain

## Ownership

`trading/` owns account state, risk policies, order intent/geometry, execution
cost and fill contracts, position lifecycle, and strategy semantics. The current
FTMO and structural SL/TP behavior are implementations of those concepts, not
the package boundary.

## Primary Files

| File | Role |
|---|---|
| `trading/account/state.py` | Account and position accounting state |
| `trading/risk/guardrails.py` | FTMO guardrail policy |
| `trading/risk/sizing.py` | Position sizing and stop/target math |
| `trading/orders/entries.py` | Structural stop/target construction |
| `trading/orders/stops.py` | Structural stop candidates, filtering, and rejection provenance |
| `trading/orders/targets.py` | Structural target candidates and market-level selection |
| `trading/orders/selection.py` | TP-mode, menu selection and fraction decoding |
| `trading/execution/broker.py` | Broker-neutral fill and position transitions |
| `trading/execution/costs.py` | Spread, slippage and commission semantics |
| `trading/positions/multi_tp.py` | Multi-target position lifecycle |
| `trading/policy/config.py` | Resolved policy configuration |
| `trading/strategies/` | Entry strategy contracts and implementations |

## Dependency Direction

- **May depend on:** configuration, artifact contracts where needed, market-data
  values, and stable simulation contracts.
- **Must not depend on:** `simulation/environment/` runtime modules, model
  implementations, training workflows, or broker SDKs.
- Simulation may compose trading account/order/risk contracts into a state
  transition; trading must remain usable without a particular simulator.

## Change Map

| Task | Axis | Start file | Contract and focused test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add an order target policy | target construction | `trading/orders/targets.py` | order geometry; `tests/test_backtest/test_entry_levels.py` | target policy and execution regression | simulation action decoding internals |
| Change risk policy | risk decision | `trading/risk/contracts.py` | `RiskPolicy`; `tests/test_risk_policy_contract.py` | policy and contract tests | model and broker implementation |

## Axis of Change

- **Axis:** account policy, risk policy, order intent/target selection, execution
  costs/fills, and position lifecycle.
- **Stable boundary:** account snapshots, risk decisions, order intent/geometry,
  and execution/fill records. The current code is still migrating these into
  explicit contract modules.
- **Expected edit radius:** one policy or execution implementation plus its
  focused order/fill tests.
- **Must stay independent from:** the learner/provider family and environment
  runtime details.

## Focused Tests

- `tests/test_backtest/test_entry_levels.py`
- `tests/test_envs/test_tp_action_contracts.py`
- `tests/test_envs/test_tp_mode_state.py`
- `tests/test_envs/test_tp_execution.py`
- `tests/test_envs/test_tp_action_decode.py`
- `tests/test_envs/test_sl_tp_fill.py`
- `tests/test_wp02_accounting.py`
- `tests/test_wp_correctness.py`
## Verified Source Paths

- `trading/orders/entries.py`
- `trading/orders/stops.py`
- `trading/orders/targets.py`
- `trading/orders/selection.py`
- `trading/risk/contracts.py`
- `trading/execution/broker.py`
- `trading/orders/targets.py`
