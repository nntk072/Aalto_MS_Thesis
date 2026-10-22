# Dependency Security and CI Hardening Implementation Plan

## 1. Purpose

This plan addresses the GitHub Dependabot vulnerability backlog currently reported for the project and adds a durable dependency-security gate to CI without destabilizing the scientific/RL training stack.

The current problem is dependency security, not a lack of Ruff, mypy, or pytest. Ruff, mypy, tests, architecture checks, graph-quality checks, contracts, parity tests, and training smoke tests cover source quality and behavioral correctness. Dependabot alerts are generated because versions present in the dependency graph match published security advisories.

The implementation therefore focuses on four outcomes:

1. Inventory and classify all current dependency vulnerabilities.
2. Remediate Critical and High vulnerabilities safely and incrementally.
3. Add a repeatable CI dependency-audit gate so vulnerable dependency changes cannot enter silently.
4. Add a small set of complementary security checks that have high value for this thesis project.

This is intentionally scoped as thesis/research hardening rather than enterprise security engineering.

## 2. Current Baseline

### Repository state

Baseline Git commit at plan creation:

- `f504c82` — graph-quality CI hardening already present.

Existing CI already includes:

- dependency lock validation
- Ruff format/lint
- mypy
- architecture tests
- graph-quality verification
- contract tests
- unit tests
- build/import smoke
- integration tests
- execution/parity tests
- training smoke tests
- aggregate `ci-success`

Existing quality tooling is sufficient for general code quality. Additional generic linters are not required for the dependency-alert problem.

### Known GitHub security signal

GitHub has reported approximately 179 Dependabot alerts on the default branch, including a large Critical/High subset. The exact live count may change as GitHub advisories or the dependency graph changes, so the first implementation step must capture the current alert set rather than relying permanently on the historical count.

### Frozen-lock audit of the active main checkout (2026-10-21)

At baseline commit `f504c82`, a Linux runtime-plus-dev export from `uv.lock` was scanned with `pip-audit==2.10.1` using `--no-deps --disable-pip`. It returned 121 finding records, 78 unique package/advisory pairs, across 14 package names. Repeated records mean the raw total is not a unique-advisory count and is not directly comparable with GitHub's Dependabot total. `torch==2.14.0+cu130` was skipped because that CUDA-local version is not available from PyPI; the scan does not establish that Torch is vulnerability-free.

The scan used `pyproject.toml` SHA-256 `690024bc52cabd78ca1dfcb746ebacbdad4c44cc2c2b9fd21d5e00e134ace105` and `uv.lock` SHA-256 `1d4c93c7c1f391e58bf9971d6cb48eb775a4c5485418665601e25f98aafc3fcd`. Machine-readable output is in [`evidence/pip_audit_locked_linux_dev_20261021.json`](evidence/pip_audit_locked_linux_dev_20261021.json), SHA-256 `30e2afb42a421c05bb822b640e75c78182fa22c8b50b242d33df05d570032afb`. This is a snapshot of the scanner's advisory database as observed on that date, not a live security status.

### Main-checkout implementation status (2026-10-21)

The main checkout now has `./tools/check security-deps`, an isolated `pip-audit==2.10.1` run against the frozen Linux runtime+dev export, a fail-closed advisory/package exception policy in `vulnerability_exceptions.toml`, and a required GitHub Actions `security-deps` job. Its first gate run correctly failed on the 78 unresolved advisory/package pairs above; dependency remediation is still required before CI can pass.

The separate credential scan is also wired as `./tools/check security-secrets` and a required CI job. It uses `detect-secrets==1.5.0` with only the Base64 and Hex entropy detectors disabled. A prior note says the scan passed after allowlisting one non-verified test fixture, but the completed Slurm job's command/output evidence was not located during the 2026-10-21 status review; treat that result as unverified. Active P7 records a 539-file source identity under fingerprint `10e75ab7a25a53b3d4444098185fe45603811f09f2d556265c9f80ca402388ab`. The security scripts/tests, VAE loader, and `docs/security/` are outside it; `pyproject.toml` and `uv.lock` are included and must stay unchanged until P7 is terminal. Do not infer Mypy or pytest completion for these security changes from the P7 pretraining gate.

### Remediation triage from the locked scan

The 14 affected packages divide into four direct dependencies and ten transitive dependencies. This is a candidate order for remediation after P7 is terminal; it is not authorization to change the lock during training. The versions below are the scanner's fix versions in the 2026-10-21 snapshot, not a live advisory check.

| Scope | Locked packages | Initial action / risk |
|---|---|---|
| Direct | `bokeh` 3.8.0, `python-dotenv` 1.0.1, `tornado` 6.5.2 | Test compatible patched releases (`3.8.2`, `1.2.2`, `6.5.9`) in one small batch. |
| Direct, broad API surface | `pillow` 11.3.0 | The scan reports 26 advisories with fixes spanning 12.1.1–13.0.0; validate the highest required fixed version against image/report exports before locking. |
| Transitive, routine patch candidates | `certifi`, `fonttools`, `idna`, `requests`, `soupsieve`, `urllib3` | Prefer compatible patched versions (`2026.10.19`, `4.60.2`, `3.15`, `2.33.0`, `2.9.0`, `2.8.0`) and inspect the resolved parent changes. |
| Transitive, runtime family changes | `curl-cffi`, `protobuf` | Upgrade through their parents (`yfinance`) where possible; retain protobuf major version 6 and validate data-provider imports. |
| Transitive, development tooling | `fastmcp`, `pyjwt` | Trace `code-review-graph`/MCP parents and test the tool stack. `fastmcp` fixes span major versions; one `pyjwt` advisory has no fixed version in this scan, so do not blanket-exempt it. |

After the campaign is terminal, apply updates in small dependency-family batches, regenerate the lock, and run the existing scoped/full gates required for the resulting code and dependency changes. Keep the known-vulnerability gate fail-closed; do not add bulk exceptions merely to restore a green CI result.

### Existing security-tool gap at baseline

The repository currently has no explicit dependency-vulnerability scanner wired into local checks/CI, such as `pip-audit` or `safety`.

It also has no dedicated repository-local source-security or secret-scanning gate such as Bandit/Semgrep or Gitleaks/detect-secrets.

For the current objective, only the dependency audit is P0. Source-security and secret-scanning are separate workstreams.

## 3. Design Principles

### 3.1 Keep security tooling isolated from the thesis runtime

Do not add `pip-audit` to the project runtime dependencies merely to make CI work. Prefer installing it as an isolated tool in CI, similar to the existing isolated `code-review-graph` installation.

Reason:

- security tooling has its own dependency tree;
- it should not constrain the training environment;
- it should not perturb Torch/SB3/NumPy/Numba compatibility;
- it should be easy to upgrade independently.

### 3.2 Do not bulk-upgrade the scientific stack

Do not run a blind global dependency upgrade to make the Dependabot number smaller.

Upgrades must be grouped by vulnerability and dependency family, with full checks between batches. This is especially important for PyTorch, Stable-Baselines3, Gymnasium, NumPy, Pandas, Numba, llvmlite, PyArrow, and plotting/backtesting packages.

The thesis training environment must remain reproducible and behaviorally stable.

### 3.3 Treat direct and transitive vulnerabilities differently

For every alert, identify whether the vulnerable package is:

- a direct dependency declared in `pyproject.toml`;
- a transitive dependency pulled by another package;
- runtime/training relevant;
- development-only;
- platform-specific;
- potentially unreachable in this project.

Prefer upgrading the parent dependency when a vulnerable package is transitive.

### 3.4 Fail closed for new unreviewed vulnerabilities

Once the existing backlog is triaged, the CI gate should fail when an unreviewed vulnerability appears.

Any temporary exception must be explicit, documented, and reviewable.

## 4. Phase 0 — Establish the Security Baseline

### Tasks

1. Capture the current Git commit and lockfile hash.
2. Export or query the full current Dependabot alert list.
3. Record the current number of alerts by severity.
4. Create a machine-readable vulnerability ledger.
5. Establish ownership and remediation state for each advisory.

### Recommended ledger fields

Create a file such as `docs/security/dependency_vulnerability_ledger.csv` or a generated JSON/Markdown equivalent with these fields:

- advisory ID (`GHSA-*` / `CVE-*`)
- package name
- installed/resolved version
- patched/fixed version
- severity
- CVSS score if available
- direct/transitive
- parent dependency if transitive
- runtime/dev/platform scope
- reachable/relevant/unknown
- remediation action
- status
- justification for exception if any
- review/expiry date

### Acceptance criteria

- Every current Dependabot alert appears in the ledger.
- Alert totals in the ledger reconcile with GitHub at capture time.
- Critical and High alerts are explicitly identifiable.
- No dependency is upgraded yet during this inventory step.

## 5. Phase 1 — Add `pip-audit` as the Dependency Security Scanner

### Objective

Provide a reproducible command that audits the project dependency set locally and in GitHub Actions.

### Tooling decision

Use `pip-audit` as the first dependency-security scanner.

Install it as an isolated tool in CI, conceptually:

```bash
uv tool install pip-audit
```

Avoid adding it to `[project.dependencies]`.

If version reproducibility is desired, pin the scanner version explicitly in CI after validating a compatible release.

### Audit target

The audit must reflect the frozen project dependency set rather than an unrelated environment.

Implementation must determine the most reliable method for the current `uv` setup, for example:

- audit an exported frozen requirements representation derived from `uv.lock`;
- audit the synchronized CI environment;
- use another `pip-audit` input mode that accurately matches the locked dependency resolution.

The chosen method must be tested against this repository rather than assumed.

### Required behavior

The audit command must:

- return nonzero for unaccepted known vulnerabilities;
- print package/advisory/fixed-version information;
- work on GitHub Actions Ubuntu runners;
- avoid modifying `uv.lock`;
- avoid installing/upgrading project runtime packages as part of the audit;
- support documented exceptions when remediation is temporarily blocked.

### Acceptance criteria

- `pip-audit` runs successfully against the repo dependency set.
- The report can be correlated with GitHub Dependabot findings.
- Running the audit does not mutate `pyproject.toml` or `uv.lock`.

## 6. Phase 2 — Add `./tools/check security-deps`

### Objective

Make dependency security part of the project's standard check interface.

### Planned change

Extend `tools/check` with:

```bash
./tools/check security-deps
```

### Gate responsibilities

The gate should:

1. Verify the expected dependency/lock state is present.
2. Run the dependency vulnerability audit.
3. Apply only reviewed advisory exceptions.
4. Fail on any unreviewed vulnerability covered by the policy.
5. Produce readable output for CI debugging.

### Integration with existing commands

Update the usage text to include `security-deps`.

Decide whether `full` should include `security-deps` after the gate is stable. It should eventually be included if runtime cost is reasonable.

`fast` does not need to include the security audit unless execution is sufficiently cheap and reliable.

### Acceptance criteria

- `./tools/check security-deps` works locally on Triton/standard Linux where supported.
- The check is deterministic from the locked dependency state.
- A controlled negative test proves the gate can fail.
- The command does not alter the environment unexpectedly.

## 7. Phase 3 — Add the GitHub Actions `security-deps` Job

### Objective

Run dependency vulnerability auditing on every push/PR covered by the existing CI workflow.

### Planned CI dependency graph

```text
dependency-lock
      |
      +--> quality
      +--> architecture
      +--> graph-quality
      +--> contracts
      +--> unit
      +--> build-smoke
      +--> security-deps   [NEW]
```

The new job should start after `dependency-lock` succeeds.

### Proposed job behavior

Conceptually:

```yaml
security-deps:
  needs: dependency-lock
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v5
    - name: Install dependency auditor
      run: uv tool install pip-audit==<validated-version>
    - run: ./tools/check security-deps
```

Exact syntax/version must be validated during implementation.

### Aggregate CI

Add `security-deps` to the `ci-success` `needs` list.

The aggregate gate must fail if the security job fails or is unexpectedly skipped.

### Acceptance criteria

- Security audit runs independently from Ruff/mypy.
- Failure is clearly attributable to dependency vulnerabilities.
- `ci-success` requires `security-deps`.
- A clean dependency state can pass the job.

## 8. Phase 4 — Define a Controlled Vulnerability Exception Policy

### Objective

Allow narrowly reviewed exceptions for vulnerabilities that cannot be immediately fixed without destabilizing the scientific stack.

### Recommended configuration

Add a repository-owned configuration file, for example `config/security/vulnerability_exceptions.toml`.

Each exception should contain at least:

```toml
[[exception]]
advisory = "GHSA-..."
package = "example-package"
reason = "Upstream fixed version conflicts with validated training stack"
impact = "Feature is not used in training/runtime path"
owner = "thesis project"
review_after = "2026-11-01"
upstream = "https://..."
```

### Policy

An exception is valid only when:

- the advisory ID is explicit;
- the affected package is explicit;
- the project relevance is reviewed;
- the reason is recorded;
- an expiry/review date exists.

Avoid global ignores, severity-wide ignores, permanent undocumented suppressions, and suppressing an advisory merely to make CI green.

### Acceptance criteria

- Every ignored advisory has a documented entry.
- Unknown/unlisted advisories still fail CI.
- Expired exceptions are detectable.

## 9. Phase 5 — Remediate Critical Vulnerabilities

### Priority

Critical alerts are P0 and should be resolved before broad High-severity cleanup.

### Per-alert workflow

For each Critical advisory:

1. Identify the vulnerable package/version.
2. Determine direct vs transitive origin.
3. Identify the minimum fixed version.
4. Determine whether the affected code path is used by this project.
5. Evaluate compatibility with the scientific stack.
6. Upgrade the smallest possible dependency set.
7. Regenerate `uv.lock` intentionally.
8. Review the lockfile diff.
9. Run the validation matrix.
10. Re-run dependency audit.
11. Confirm GitHub/Dependabot recognizes remediation after push.

### Dependency-family caution

Use extra care if remediation touches:

- Torch/CUDA package selection
- NumPy ABI compatibility
- Numba/llvmlite pairing
- Pandas/PyArrow integration
- SB3/Gymnasium APIs
- model checkpoint loading
- backtesting execution semantics

### Target state

- Zero unreviewed Critical dependency vulnerabilities.
- Any unavoidable Critical exception must have a strong documented justification and explicit expiry.

## 10. Phase 6 — Remediate High Vulnerabilities

### Classification

Classify every High advisory into one of these buckets:

1. `fix-now-direct`
2. `fix-now-transitive`
3. `dev-only`
4. `platform-specific`
5. `not-reachable/research-only`
6. `blocked-by-upstream`
7. `blocked-by-training-stack-compatibility`

### Upgrade strategy

Remediate in small coherent batches, for example:

- web/HTTP dependencies
- plotting/rendering stack
- data stack
- development tooling
- ML/RL stack
- platform-specific packages

Do not mix unrelated major upgrades in one change.

### Acceptance criteria

- Every High alert is classified.
- Every fixable High alert has been remediated.
- Deferred High alerts have explicit rationale and review dates.
- Scientific-stack upgrades have training-specific validation.

## 11. Phase 7 — Validation Matrix for Every Dependency Upgrade Batch

### Always run

```bash
./tools/check deps
./tools/check quality
./tools/check architecture
./tools/check graph-quality
./tools/check contracts
./tools/check unit
./tools/check build
./tools/check integration
./tools/check parity
./tools/check training-smoke
./tools/check security-deps
```

### Run for ML/scientific dependency changes

Also run the Triton/GPU validation path when the change can affect training execution, especially for Torch, CUDA-related resolution, Stable-Baselines3, Gymnasium, NumPy, Numba, llvmlite, and PyArrow.

### Reproducibility checks

For changes affecting final training, verify:

- Git SHA recorded
- clean/expected working tree for launch
- `uv.lock` committed
- Python version unchanged or intentionally changed
- dependency fingerprint/hash recorded
- seeds unchanged
- dataset fingerprints unchanged
- config hashes unchanged unless intentionally modified
- checkpoint compatibility verified

### Acceptance criteria

No dependency remediation batch is accepted if it breaks type checking, core tests, environment contracts, execution/parity behavior, training smoke, or required GPU launch/runtime checks.

## 12. Phase 8 — Add Secret Scanning as a Separate Security Layer

### Why separate

Secret scanning solves credential exposure, not dependency CVEs. Keep it independent from `security-deps` so failures have clear meaning.

### Priority

P1 after the dependency audit is operational.

### Preferred implementation order

1. Use GitHub Secret Scanning / Push Protection if available for the repository.
2. Optionally add Gitleaks as a CI/local repository check if GitHub-only coverage is insufficient.

### Scope

Pay particular attention to MT5 login/password variables, API tokens, GitHub tokens, cloud credentials, experiment logs, run manifests, archived output/config bundles, and `.env` files.

### Acceptance criteria

- No known real credentials are committed.
- CI/log output does not echo credentials.
- Experiment provenance captures configuration without secret values.

## 13. Phase 9 — Small Source-Security Hardening Pass

This phase is complementary to Dependabot remediation.

### 13.1 PyTorch checkpoint loading

Review `learning/self_supervised/vae_pre_ny.py`.

Current relevant behavior uses unrestricted:

```python
torch.load(path, map_location="cpu")
```

If the checkpoint contains only tensors/state dictionaries and the project format permits it, migrate to:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

Add a meaningful compatibility test covering the actual VAE checkpoint format.

### 13.2 Subprocess trust boundaries

Review orchestration code where argv/env/cwd are constructed from campaign payloads, especially:

- `compute/slurm/`
- `experiments/campaigns/`
- `experiments/orchestration/`
- `orchestra/`

Current use of argv lists rather than `shell=True` is good. The review should focus on validating the source and allowed shape of payload fields.

### 13.3 Credentials and manifests

Verify MT5 credentials and other secrets are never copied into command manifests, environment snapshots, experiment metadata, stdout/stderr archives, or thesis evidence artifacts.

### Acceptance criteria

- Trusted checkpoint format enforced where practical.
- No new shell-string execution paths.
- No secret values written to provenance artifacts.

## 14. Phase 10 — Decide Whether to Add Bandit or Semgrep

Do not add Bandit/Semgrep merely to resolve the Dependabot backlog. They scan source patterns, while Dependabot/pip-audit scan vulnerable dependency versions.

Potential P2 use:

- Bandit for Python-specific security checks.
- Semgrep for targeted project rules.

If added, start with a small curated ruleset and baseline findings before making it required. Avoid introducing a noisy CI gate immediately before final thesis training.

## 15. Phase 11 — Optional Supply-Chain Hardening

P2, after Critical/High dependency work is stable.

Potential improvements:

- pin GitHub Actions to immutable commit SHAs;
- generate SBOM for release/container artifacts;
- add container image vulnerability scanning;
- pin Docker base images by digest where appropriate;
- generate provenance/attestations for GHCR images;
- audit build dependencies separately from runtime dependencies.

These are useful, but they have lower thesis ROI than dependency remediation and reproducibility checks.

## 16. Proposed Commit Structure

### Commit 1 — Add dependency-security CI gate

Scope:

- `./tools/check security-deps`
- isolated `pip-audit` setup
- GitHub Actions `security-deps` job
- `ci-success` integration
- vulnerability-exception configuration format
- focused tests for wrapper/config behavior if needed

No broad dependency upgrades in this commit.

### Commit 2 — Remediate Critical vulnerabilities

Scope:

- minimum required dependency updates
- lockfile update
- compatibility fixes directly required by those updates
- documentation/ledger update

### Commit 3+ — Remediate High vulnerabilities in coherent batches

Possible batches:

- High runtime/direct
- High runtime/transitive
- High dev/tooling
- scientific stack changes requiring GPU verification

Do not bundle unrelated thesis/source work into these commits.

## 17. Agent Execution Order

An implementation agent should execute in this order:

1. Preserve and inspect existing dirty working-tree state.
2. Continue only in the existing `/scratch/work/nguyenl37/Aalto_MS_Thesis` main checkout; do not create or switch branches or worktrees.
3. Capture current Dependabot/security baseline.
4. Prototype `pip-audit` against the actual locked environment.
5. Select and pin a validated audit-tool version.
6. Implement `./tools/check security-deps`.
7. Add exception configuration and parser/validation if required.
8. Add GitHub Actions `security-deps` job.
9. Wire it into `ci-success`.
10. Run negative tests proving the gate fails appropriately.
11. Run current CI-equivalent checks.
12. Commit the CI/security-gate change separately.
13. Triage Critical alerts.
14. Remediate Critical dependencies in minimum-change batches.
15. Run full validation after each batch.
16. Triage and remediate High alerts.
17. Add secret scanning.
18. Harden unsafe checkpoint deserialization if compatible.
19. Review subprocess/credential trust boundaries.
20. Update the vulnerability ledger and final security status report.

## 18. Definition of Done

The dependency-security work is complete when all of the following are true:

### Dependency inventory

- All current Dependabot alerts are represented in the vulnerability ledger.
- Direct/transitive origin is known for Critical and High alerts.
- Runtime/dev/platform relevance is classified.

### Critical severity

- Zero unreviewed Critical vulnerabilities remain.
- Any unavoidable Critical exception is explicit, justified, and time-bounded.

### High severity

- Every High alert has a disposition.
- Safe fixes are applied.
- Deferred findings are documented and time-bounded.

### CI

- `./tools/check security-deps` exists and passes in the accepted dependency state.
- GitHub Actions runs `security-deps`.
- `ci-success` depends on `security-deps`.
- CI fails on newly introduced unreviewed vulnerabilities.

### Quality/regression safety

- Ruff passes.
- mypy passes.
- architecture checks pass.
- graph-quality passes.
- contract tests pass.
- unit tests pass.
- build smoke passes.
- integration tests pass.
- parity tests pass.
- training smoke passes.
- Triton/GPU validation passes for changes affecting the scientific/ML stack.

### Security hygiene

- Secret scanning is enabled or an equivalent repository check is present.
- No credentials are present in tracked source, logs, or run manifests.
- VAE checkpoint loading uses the safest compatible loading mode.
- subprocess entrypoints have an explicit trusted-input model.

### Reproducibility

- Dependency changes are committed in `uv.lock`.
- Final training runs record repository SHA and dependency state.
- Security remediation does not silently change experiment semantics.

## 19. Priority Summary

### P0 — Do before final training freeze

- Capture all current Dependabot alerts.
- Add `pip-audit`-based `security-deps` check.
- Add the CI security job and aggregate gate.
- Triage and remediate all Critical alerts.
- Classify every High alert.
- Validate scientific-stack changes on Triton/GPU.

### P1 — Strongly recommended

- Remediate safe High vulnerabilities.
- Add time-bounded vulnerability exceptions.
- Add secret scanning.
- Harden VAE `torch.load` if checkpoint format permits.
- Review subprocess and credential logging trust boundaries.

### P2 — Later / optional

- Bandit or Semgrep with curated rules.
- SBOM generation.
- Container scanning/provenance.
- GitHub Action SHA pinning.
- Docker base-image digest pinning.

## 20. Final Recommendation

For this thesis repository, the highest-return next security change is a dependency-security feedback loop:

```text
Dependabot/GitHub advisory signal
          ↓
locked dependency inventory
          ↓
pip-audit / security-deps CI gate
          ↓
Critical/High triage
          ↓
small dependency upgrade batches
          ↓
full correctness + parity + training validation
          ↓
documented exceptions only when necessary
```

This gives the project a defensible security posture while protecting the stability and reproducibility of the final RL training environment.
