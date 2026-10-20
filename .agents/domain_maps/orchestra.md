# Orchestra Domain Map

## Ownership

Own engineering-task orchestration, phase transitions, agent invocation, and
verification policy. It does not own thesis experiment or trading semantics.

## Boundary

- Composition root: `orchestra/pipeline.py`.
- Phase bodies are grouped by lifecycle stage in `orchestra/phase_*.py`.
- Task routing and handoff live in `orchestra/` modules.
- Infrastructure may call repository checks but production packages must not import it.

## Invariants and Tests

- Preserve explicit phase state, bounded retries, and captured verification results.
- Keep orchestration metadata out of scientific artifacts unless provenance requires it.
- Focused tests: `tests/test_orchestra/`.

## Change Map

| Task | Axis | Start file | Contract and test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add or change a pipeline phase | workflow state | `orchestra/pipeline.py` | phase/state contract; orchestra tests | one phase and routing tests | thesis training semantics |

## Phase Ownership

- `orchestra/phase_triage.py`: task triage and complexity selection.
- `orchestra/phase_planning.py`: plan generation and planner coordination.
- `orchestra/phase_critique.py`: plan critique and synthesis.
- `orchestra/phase_implementation.py`: implementation handoff.
- `orchestra/phase_review.py`: review and review synthesis.
- `orchestra/phase_fixing.py`: bounded repair cycle.
- `orchestra/phase_verification.py`: verification result handling.
- `orchestra/phase_reporting.py`: final task report.

## Axis of Change

- **Axis:** engineering workflow phase, routing, and verification policy.
- **Owner:** `orchestra/` owns task orchestration, not scientific workflows.
- **Stable boundary:** pipeline state and phase result contracts.
- **Expected edit radius:** one phase and routing/verification tests.
- **Must not depend on:** thesis trading or model semantics.

## Verified Source Paths

- `orchestra/pipeline.py`
- `orchestra/handoff.py`
- `tests/test_orchestra/test_handoff.py`
