# Security hardening status

**Updated:** 2026-10-23
**Scope:** Current `main` checkout and the Linux runtime-plus-dev dependency set.

## Current state

- The locked dependency audit now reports zero finding records, zero unique package/advisory pairs, and zero affected packages. The fail-closed gate has no active reviewed exceptions.
- The audit skips the exact CUDA build `torch==2.14.0+cu130`, which is not available from PyPI. Its security status remains unverified by this audit; do not describe the full environment as vulnerability-free.
- The reviewed dependency changes are recorded in `pyproject.toml`, `uv.lock`, `requirements.txt`, and `requirements-dev.txt`. `uv lock --check` and `make deps-check` pass.
- The secret scanner passes with only its configured Base64 and Hex entropy detectors disabled. The audit scripts, exception file, and tests are wired into required CI jobs.
- Full-tree Ruff format, Ruff lint, and Mypy pass (739 files). The scoped compatibility and evidence suite passes (110 tests), including checkpoint migration, campaign artifacts, numerical reports, and VAE loading.
- `load_vae_from_checkpoint` uses `weights_only=True` for tensor state dictionaries.

## Historical baseline

The frozen Linux audit at baseline `f504c82` found 121 finding records, 78 unique package/advisory pairs across 14 packages. Its original report is [`evidence/pip_audit_locked_linux_dev_20261021.json`](evidence/pip_audit_locked_linux_dev_20261021.json), SHA-256 `30e2afb42a421c05bb822b640e75c78182fa22c8b50b242d33df05d570032afb`. That report is retained as the pre-remediation snapshot and does not describe the current lock.

See [`dependency_security_ci_hardening_plan.md`](dependency_security_ci_hardening_plan.md) for the remediation rationale and scan limitations.
