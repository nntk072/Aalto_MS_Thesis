# Development workflow

The full feature development process that happens before git operations: research, planning, TDD, code review, then committing.

## Feature implementation pipeline

0. **Research & reuse first** (mandatory before new code)
   - Search GitHub (`gh search code`) and package registries (PyPI) for existing implementations before writing utilities.
   - Prefer a battle-tested library over a hand-rolled solution when it meets the requirement.
   - Check this repo first: many indicators/rewards/env pieces already exist under `quant_rl/` and `backtest/`.

1. **Plan before execute**
   - For complex features: write a short plan (goal, phases, risks, affected modules) before coding.
   - Identify dependencies and blast radius using [code-review-graph.md](code-review-graph.md).

2. **TDD** — follow [testing-rules.md](testing-rules.md): RED → GREEN → IMPROVE.

3. **Review** — run [code-review-graph.md](code-review-graph.md) on your own diff; fix CRITICAL/HIGH findings.

4. **Commit & push** — follow [git-commit-rules.md](git-commit-rules.md).

5. **Pre-review checks**
   - All automated checks pass (ruff, mypy, pytest).
   - Branch up to date with main; no merge conflicts.
   - Only request review after these pass.

## Knowledge capture

- Project knowledge (architecture decisions, API changes, runbooks) goes into the project's docs structure.
- Do not duplicate information that already exists in docs or code comments.
- Ask before creating any new top-level documentation file.

## Relations

- Activates: any non-trivial feature or refactor. Orchestrates [testing-rules.md](testing-rules.md), [code-review-graph.md](code-review-graph.md), [git-commit-rules.md](git-commit-rules.md), and [security-rules.md](security-rules.md).
