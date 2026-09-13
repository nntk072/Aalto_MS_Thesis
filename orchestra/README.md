# Orchestra — Multi-Model Coding Pipeline

Orchestra coordinates multiple CLI-based coding agents through a structured
workflow: triage → plan → implement → verify. The verification phase runs the
same CI gate as GitHub Actions (`orchestra/ci_gate.py`).

## Quick start

```bash
# Tier-0 health check (no inference)
orchestra doctor

# Full pipeline on a task
orchestra run "add unit tests for purged_walk_forward"

# Dry run (show plan without spawning agents)
orchestra run "refactor risk module" --dry-run
```

## Pipeline phases

| Phase | Purpose |
|-------|---------|
| 1. Triage | Classify task complexity and tier |
| 2. Planning | Parallel planners produce implementation plans |
| 3. Critique | Critics review plans |
| 4. Synthesis | Merge plans into a single plan |
| 5. Implementation | Implementer applies changes |
| 6. Review | Reviewers check the diff |
| 7. Review synthesis | Merge review feedback |
| 8. Fixing | Address review findings (up to `max_fixes`) |
| 9. Verification | Run CI gate (`CI_CHECKS`) |

State is persisted under `orchestra/state/` per task.

## CI verification gate

Defined in `orchestra/ci_gate.py` (`CI_CHECKS`). Mirrors
`.github/workflows/ci.yml` jobs `code-formatting` + `ut-venv`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest tests/ -v
```

Orchestra phase 9 (verification) and the implementer prompt run this gate.
Nightly workflows (`pytest -m slow`) are **not** part of the merge bar.

## Package layout

```
orchestra/
  cli.py           CLI entry point (orchestra doctor / run)
  pipeline.py      Phase orchestrator
  ci_gate.py       CI_CHECKS definition
  router.py        Model selection and routing
  health.py        Tier-0/Tier-1 health probes
  agent_runner.py  Spawn and capture agent CLI sessions
  prompts/         Phase prompt templates
  docs/
    probe-matrix.md  CLI probe evidence and tier gates
```

## Configuration

- Model definitions: `orchestra/models.yaml`
- Probe matrix and tier gates: [docs/probe-matrix.md](docs/probe-matrix.md)
- Agent rules: `.agents/rules/ci-verification.md`

## References

- [AGENTS.md](../AGENTS.md) — agent guide (orchestra section)
- [docs/operations/RUNNING_COMMANDS.md](../docs/operations/RUNNING_COMMANDS.md) — orchestra commands
- [docs/architecture.md](../docs/architecture.md) — quant_rl architecture
