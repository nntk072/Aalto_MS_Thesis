# Experiments Domain Map

## Ownership

Own variant definitions, campaign lifecycle, resource-aware scheduling, resume,
artifact verification, and campaign provenance. Scientific algorithms belong to
their domain packages.

## Boundary

- Campaign lifecycle: `experiments/campaigns/`.
- `execution.py` composes focused state, resource, source-snapshot, artifact, and checkpoint services.
- Variant workflows: `experiments/variants/`.
- May compose training, evaluation, compute, and artifacts at explicit workflow roots.
- Must not import simulation runtime internals or own feature/trading semantics.

## Invariants and Tests

- Preserve campaign identity, source snapshots, checkpoint lineage, and resume state.
- Never create duplicate campaigns or short/fan-out Slurm steps.
- Focused tests: `tests/test_campaign*`, `tests/test_triton_campaign*`.

## Change Map

| Task | Axis | Start file | Contract and test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Change campaign lifecycle | workflow state | `experiments/campaigns/execution.py` | campaign state/provenance; campaign tests | one lifecycle service and tests | trading and simulation semantics |

## Axis of Change

- **Axis:** campaign lifecycle, resource scheduling, and resume state.
- **Owner:** `experiments/` composes training, evaluation, compute, and artifact services.
- **Stable boundary:** campaign state, source identity, and checkpoint lineage records.
- **Expected edit radius:** one lifecycle service and campaign regression tests.
- **Must not depend on:** simulation runtime internals or feature/trading semantics.

## Verified Source Paths

- `experiments/campaigns/execution.py`
- `experiments/campaigns/state_store.py`
- `experiments/campaigns/resources.py`
- `experiments/campaigns/source_snapshot.py`
- `experiments/campaigns/artifact_validation.py`
- `experiments/campaigns/checkpoint_lineage.py`
- `experiments/campaigns/constants.py`
- `experiments/variants/runner.py`
- `tests/test_campaign_results_summary.py`
