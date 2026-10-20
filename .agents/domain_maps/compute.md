# Compute Domain Map

## Ownership

Own hardware capability discovery, accelerator precision selection, resource sizing,
and scheduler adapters. Learning composes the selected precision with model-specific
encoder behavior; models do not discover hardware.

## Boundary

- Public entry points: `compute/environment/capabilities.py`, `compute/environment/device.py`, `compute/environment/precision.py`, `compute/slurm/resources.py`.
- Device, resource, job, and backend contracts live in `compute/contracts.py`.
- Device discovery may use Torch; compute must not import model or learner classes.
- Slurm is an adapter. Scientific result identity must not depend on job IDs.

## Invariants and Tests

- Respect the live allocation and CPU/RAM/GPU inventory before a real request.
- Never fan out short Slurm steps; use an existing allocation when available.
- Focused tests: `tests/test_utils/test_device.py`, `tests/test_triton_campaign_worker_dispatch.py`.

## Change Map

| Task | Axis | Start file | Contract and test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add a compute backend | execution adapter | `compute/slurm/resources.py` | resource request semantics; scheduler tests | one backend and contract tests | model constructors and scientific behavior |
| Change accelerator precision | device runtime policy | `compute/environment/precision.py` | capability-driven dtype; `tests/test_utils/test_device.py` | hardware selection and runtime setup | model-specific encoder behavior |

## Axis of Change

- **Axis:** hardware capability discovery and workload execution backend.
- **Owner:** `compute/` owns resource facts and scheduler adapters.
- **Stable boundary:** `DeviceCapabilities`, `ResourceProfile`, `JobSpec`, and `ExecutionBackend`.
- **Expected edit radius:** one backend adapter and its contract tests.
- **Must not depend on:** model classes or scientific policy.

## Verified Source Paths

- `compute/environment/device.py`
- `compute/environment/capabilities.py`
- `compute/environment/precision.py`
- `compute/slurm/resources.py`
- `tests/test_utils/test_device.py`
- `tests/test_triton_campaign_worker_dispatch.py`
- `compute/contracts.py`
