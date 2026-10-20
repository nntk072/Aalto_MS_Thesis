# Configuration Domain Map

## Ownership

Own configuration loading, merge order, validation, canonical paths, and stable
configuration identity. Domain packages resolve semantic settings from it.

## Boundary

- Public loader: `configuration/loader.py`.
- Canonical defaults: `config/base/default.yaml`.
- Depends on configuration formats and filesystem mechanics only.
- Must not import training, model, or experiment workflows.

## Invariants and Tests

- YAML remains the single source of truth for configured runtime values.
- CLI overrides apply last; resolved config and its identity are persisted.
- Focused tests: `tests/test_config_typo_safety.py`.

## Change Map

| Task | Axis | Start file | Contract and test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add or migrate a config field | schema evolution | `config/base/default.yaml` | loader/schema tests | default, owning domain resolver, focused tests | unrelated domain defaults |

## Axis of Change

- **Axis:** configuration format, merge policy, schema validation, and identity hashing.
- **Owner:** `configuration/` owns parsing and resolved configuration identity; each domain owns its semantic settings.
- **Stable boundary:** resolved config values and configuration identity helpers.
- **Expected edit radius:** one schema/loader change and the owning domain resolver with focused tests.
- **Must not depend on:** training, model, experiment, or scientific workflow implementations.

## Verified Source Paths

- `configuration/loader.py`
- `config/base/default.yaml`
- `tests/test_config_typo_safety.py`
