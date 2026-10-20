# Decisioning Domain

`decisioning/` is the backend-neutral inference boundary used by evaluation and
live runtime consumers.

## Ownership

- Versioned observation and action schema references.
- Stable action schema builders and account/entry-state observation coordinates.
- Semantic discrete, scalar-continuous and named-coordinate decisions.
- Provider metadata and persisted artifact references.
- Adapters that translate concrete model outputs into the stable decision contract.

## Non-ownership

- Training workflows, losses, fitting, feature construction or execution policy.
- Simulation environment internals and broker-specific APIs.
- Model construction or checkpoint discovery.

## Public API

- `DecisionInput`, `Decision`, and `DecisionProvider` from `decisioning`.
- `ActionSchema`, `ObservationSchema`, and their versioned builders from `decisioning`.
- `SB3DecisionProvider` from `decisioning.adapters.sb3` for current saved SB3 models.
- `to_policy_action()` only at the environment adapter boundary.

## Dependency direction

`models` and future `learning` backends may implement providers. `evaluation` and
`live` consume providers. The contract package must not import those workflows,
simulation runtime internals, or model implementations. Adapters may depend on
their concrete backend and the generic contract.

## Invariants and tests

- Observation schema version is checked before inference.
- Action coordinates come from persisted schema metadata; width alone is not
  treated as semantic identity.
- Existing SB3 output coordinates are preserved when converted to the
  environment representation.
- Contract and adapter tests live under `tests/test_decisioning/`; consumer
  adaptation tests live with the consumer, currently evaluation rollout tests.

## Current source paths

- `decisioning/contracts.py`
- `decisioning/actions.py`
- `decisioning/observations.py`
- `decisioning/observation_state.py`
- `decisioning/adapters/sb3.py`

## Axis of change

- **Axis:** inference provider or semantic action schema.
- **Owner:** `decisioning/` owns the provider boundary; the backend adapter owns
  conversion from backend outputs.
- **Expected edit radius:** one contract/adapter and its focused contract tests.
- **Must not depend on:** training workflows, simulation internals or execution
  behavior.
## Change Map

| Task | Axis | Start file | Contract and focused test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add a decision backend adapter | inference adapter | `decisioning/adapters/` | `DecisionProvider`; `tests/test_eval/test_decision_provider_rollout.py` | one adapter and schema checks | fitting workflow and execution policy |
| Change action meaning | contract evolution | `decisioning/contracts.py` | versioned `ActionSchemaRef`; action schema tests | schema plus compatibility adapter | PPO/SAC implementation details |

## Axis of Change

- **Axis:** inference backend and semantic action representation.
- **Owner:** `decisioning/` owns provider metadata, schemas, and model-output adapters.
- **Stable boundary:** `DecisionInput`, `Decision`, and `DecisionProvider`.
- **Expected edit radius:** one provider adapter and provider contract tests.
- **Must not depend on:** learner implementation, trading execution, or simulation internals.

## Verified Source Paths

- `decisioning/contracts.py`
- `decisioning/actions.py`
- `decisioning/observations.py`
- `decisioning/adapters/sb3.py`
- `decisioning/adapters/rule_based.py`
- `learning/supervised/reference.py` implements a non-RL provider against this contract.
