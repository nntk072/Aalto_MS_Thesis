# P2 Hash-Seed Runtime Provenance — 2026-10-18

**Status:** bounded implementation and contract tests passed; no clean-checkpoint
cross-seed evaluation was run.

## Contract

The evaluation runtime record captures `PYTHONHASHSEED` when
`evaluation.analysis.run_manifest` loads. It records a string when the variable is
present or JSON `null` when it is absent. The numerical-report verifier and
deterministic-reanalysis comparator require the field and reject values other
than string or null. The comparator treats this one field as an explicit
comparison axis: two runs may declare different hash seeds, but all other
provenance fields and the required numerical artifacts must still match.

## Validation

The captured log is
[`p2_hashseed_runtime_provenance_20261018.log`](../../../outputs/thesis_validation/p2_hashseed_runtime_provenance_20261018.log),
SHA-256 `112906284b738f91af89b9eae978df4a21f6e2e5925862dfa338442671561f87`.
It records scoped Ruff format and lint passes and **87 passing tests** across
run-manifest creation, evaluation provenance, numerical-report verification,
and deterministic-reanalysis comparison (42 pre-existing pandas warnings).
Path-scoped Mypy also reported no issues in the seven touched Python source and
test files; full-tree gates were not rerun for this change.
The comparator regression accepts identical artifacts under explicit seeds 1
and 12; the verifier/comparator regressions reject absent and malformed
records. The test process itself started without `PYTHONHASHSEED`, as recorded
by the shell environment; one unit test sets a captured startup value and then
mutates the later environment to verify the captured record does not change.

## Source identity

The tests ran at repository HEAD
`d0183cd34f5707eaadc5dad8af9ce0abf7d42dd1` on branch `main`. Relevant file
SHA-256 values at validation time:

| File | SHA-256 |
|---|---|
| `quant_rl/eval/run_manifest.py` | `02e46463d6c074caa5e7904cf758e627a845d35ec5dd03157a6c11c1205b5d81` |
| `scripts/thesis/compare_deterministic_reanalysis.py` | `e9024b7ff21c35cb1e5a1288879dea19adf11fa29210d3b1a17ad54c832793fd` |
| `scripts/thesis/verify_numerical_report.py` | `34fba97739084d9f14a2d890c085e19611c074a7c7f58e655c47aa25771ae063` |
| `tests/test_eval/test_run_manifest.py` | `18ae9a85ddee42f33e5be062ec01500f62d4d10cbfbbe85b1c1e0f1737cc8d6c` |
| `tests/test_eval/test_eval_run_provenance.py` | `3ad85767a1c443dc06c447f6c6abd9bf6a4f5a6af83b83496962863f8ba16263` |
| `tests/test_thesis_numerical_report_verifier.py` | `5e6b31b53dc053940da02647bb8dbd1e847f6a325f67d26ecf7ec163ddadc85e` |
| `tests/test_thesis_deterministic_reanalysis_comparator.py` | `2e70346954359a64619cbafdb62fe57343ef526a0045087d1a2994309daeefb1` |

## Limits and next acceptance

The tests show that the new manifests expose hash-seed environment metadata
and that the comparator can assess a crossed-seed artifact pair. They do not
show that evaluation is byte-stable across actual hash seeds, prove the
historical seed for archived replays, or accept a contaminated/legacy
checkpoint. P2 remains open until deterministic train and report-only test
evaluation, including raw trades/fills/equity and source/data/schema lineage,
passes on an admissible completed checkpoint.

## Interpreter-startup follow-up

A subprocess regression now starts real interpreters with `PYTHONHASHSEED=73`
and with the variable unset, then reads the captured runtime field. Both cases
pass. The refreshed validation log is
[`p2_hashseed_runtime_provenance_subprocess_20261018.log`](../../../outputs/thesis_validation/p2_hashseed_runtime_provenance_subprocess_20261018.log),
SHA-256 `247db301d819a7015cf118690931376b0bb987c5a49c116ee04b7610d760d372`.
It records Ruff format/lint, path-scoped Mypy (seven files, no issues), and
**89 passing tests** across the manifest, evaluator-provenance, verifier, and
comparator suites (42 pre-existing pandas warnings). The updated
`tests/test_eval/test_run_manifest.py` SHA-256 is
`9d214155459d2e152418d76e524828aa84090a9a4ad3856322035018606dda06` at this
validation. This improves confidence in provenance capture; it still does not
test cross-seed policy output equality.
