# Integrations Domain Map

## Ownership

Own external SDK and venue adapters, protocol conversion, and connectivity
details. Broker-neutral lifecycle and strategy semantics belong to `live/`.

## Boundary

- MT5 implementation: `integrations/mt5/`.
- Depends inward on live-facing ports and domain contracts.
- Vendor SDK objects must not escape the adapter boundary.

## Invariants and Tests

- Preserve order, fill, account, and timestamp conversion semantics.
- Keep credentials and SDK state at the integration boundary.
- Focused tests: `tests/test_live/` and MT5 adapter tests.

## Change Map

| Task | Axis | Start file | Contract and test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add or replace a broker adapter | external adapter | `integrations/mt5/domain/trader.py` | live broker ports; adapter contract tests | adapter and conversion tests | live strategy and model packages |

## Axis of Change

- **Axis:** external broker or vendor API adapter.
- **Owner:** `integrations/` converts SDK objects to broker-neutral live contracts.
- **Stable boundary:** live market-data, strategy, and trader ports.
- **Expected edit radius:** one adapter plus conversion and contract tests.
- **Must not depend on:** SDK types escaping into live or scientific packages.

## Verified Source Paths

- `integrations/mt5/domain/trader.py`
- `live/runtime/strategy.py`
- `tests/test_live/test_rl_bridge.py`
