# Artifacts Domain Map

## Ownership

Own logical artifact identity, manifests, atomic persistence mechanics, validation,
retention, and promotion. Producing domains own the meaning and schemas of their
artifacts.

## Boundary

- Public contracts: `artifacts/contracts.py` (`ArtifactRef`, `ArtifactStore`).
- Storage adapter: `artifacts/storage.py` (`FileSystemArtifactStore`).
- Atomic byte/text/JSON writes: `artifacts/atomic.py`.
- Depends only on the standard library. Producers may depend on `artifacts/`.
- Must not infer scientific meaning from directory names or import workflow packages.

## Invariants and Tests

- Logical keys and content digests are independent of physical layout.
- Reads verify the declared digest; writes are atomic.
- Focused tests: `tests/test_artifact_store_contract.py`, `tests/test_artifacts.py`.

## Change Map

| Task | Axis | Start file | Contract and test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add artifact storage backend | storage adapter | `artifacts/storage.py` | `ArtifactStore`; `tests/test_artifact_store_contract.py` | backend plus contract tests | producer-owned schemas and artifact meaning |

## Axis of Change

- **Axis:** artifact identity, storage layout, and retention mechanics.
- **Owner:** `artifacts/` owns opaque references and persistence mechanics; producer domains own schemas and meaning.
- **Stable boundary:** `ArtifactRef` and `ArtifactStore`.
- **Expected edit radius:** one storage or lifecycle adapter plus its contract test.
- **Must not depend on:** training, evaluation, model, trading, or campaign semantics.

## Verified Source Paths

- `artifacts/contracts.py`
- `artifacts/storage.py`
- `artifacts/atomic.py`
- `tests/test_artifact_store_contract.py`
