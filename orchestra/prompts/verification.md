## CI verification (required before done)

Run from `{{workspace}}` in this order — same as GitHub CI, **not** nightly:

{{ci_commands}}

All four commands must pass. Do not use `pytest -m "not slow"` or scoped `mypy quant_rl/` as a substitute.
