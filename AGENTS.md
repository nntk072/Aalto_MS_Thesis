# Aalto_MS_Thesis — Agent Guide

Quantitative RL trading system. PPO/SAC agents learn entry/exit timing on US100/US500
using structure-aware features (SMT divergence, PO3 state, liquidity sweeps, FVG zones).

## Build & Test

```bash
uv sync                              # install deps
python scripts/prepare_data.py       # raw CSV → parquet → features
python -m quant_rl.train.train_rl --mvp   # smoke test (30 days)

# Local (implementation): scoped checks only
uv run ruff format --check <touched paths>
uv run ruff check <touched paths>
uv run pytest <relevant test files> -q

# Agent commit/push gate — REQUIRED (do not skip)
uv run ruff format --check .    # if fails: uv run ruff format . then re-check
uv run ruff check .
uv run mypy .
uv run pytest <relevant test paths> -q   # NOT tests/ -v — keep scoped

# Full suite: GitHub CI / Orchestra phase 9 only (not agent default)
# uv run pytest tests/ -v
```

Full rule: `.agents/rules/ci-verification.md` (all agents). Commit checklist:
`.agents/rules/git-commit-rules.md`.

## Orchestra (`orchestra/`)

Multi-model CLI pipeline (triage → plan → implement → verify). Implementer agents run
scoped tests only. Phase 9 (verification) runs the full CI gate (`orchestra/ci_gate.py`).
Agent rules: `.agents/rules/ci-verification.md`.

```bash
orchestra doctor          # Tier-0 health (no inference)
orchestra run "task"      # full pipeline
```

## Architecture

```
data → features → envs → models → train → evaluation
              ↕           ↕
         backtest ←───────┘
```

| Domain | Entry File | Role |
|--------|-----------|------|
| data | `quant_rl/data/pipeline.py` | load → resample → clean → session → split |
| features | `quant_rl/features/build.py` | indicators + SMT + structure + PO3 → matrix |
| envs | `quant_rl/envs/trading_env.py` | Gymnasium env (obs, act, reward) |
| backtest | `quant_rl/backtest/engine.py` | event-driven engine (fill, SL/TP, guardrails) |
| models | `quant_rl/models/agent.py` | encoder + PPO/SAC wiring |
| train | `quant_rl/train/train_rl.py` | full training loop |
| evaluation | `quant_rl/evaluation/runner.py` | metrics, walk-forward, bootstrap CIs |
| eval | `quant_rl/eval/rollout.py` | rollout, plots, export, `eval_run` |
| live | `quant_rl/live/rl_strategy.py` | MT5 bridge |
| orchestra | `orchestra/pipeline.py` | multi-model task pipeline + CI verification |

Full architecture map: `.agents/architecture.md`
Domain-level context: `.agents/domain_maps/*.md`

## Critical Invariants (DO NOT VIOLATE)

1. **Causal features only** — no future data in any indicator. All HTF features
   use `align_timeframes` with forward-fill (never backward).
2. **Train-only normalization** — `rolling_zscore` fits on train_mask only.
   Test rows use last known training statistics.
3. **Purged walk-forward** — `purged_walk_forward()` yields **non-overlapping**
   test folds; purge = `purge_bars + label_horizon`, embargo before each test.
   Legacy overlapping stepper is `purged_walk_forward_legacy` (deprecated).
4. **Session labels are eligibility flags** — never drop bars from feature
   dataset based on session. `filter_session` is mask-only.
5. **Config is single source of truth** — `quant_rl/config/default.yaml`.
   All YAML keys are read via `cfg.<path>`. Adding a feature flag? Add YAML key first.
6. **Cache versioning** — `FEATURE_CACHE_VERSION` plus **content hash**
   (`feature_cache_path` / `.hash` sidecar). Hand-bump alone is not enough.
7. **Live risk ≈ training risk** — `live_risk_overrides` must stay aligned with
   `ftmo` dollar limits. Silent divergence is the failure mode.
8. **Locked OOS is final report** — do not select architectures on the locked
   holdout; see `docs/thesis/validation_protocol.md`.

## Context Routing

Task → read the matching domain map in `.agents/domain_maps/` first.
Use code-review-graph MCP tools (`query_graph_tool`, `get_impact_radius_tool`)
to narrow scope before reading source.

## Rules

All process rules: `.agents/rules/README.md`

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**This project has a knowledge graph. Start with the code-review-graph
MCP tools to narrow scope, then read the source.** The graph is cheaper than scanning files and
gives you structural context (callers, dependents, test coverage) that file search cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

### Verify in the source

- Narrow scope with the graph, then read the source. Do not change code from graph output alone.
- For any non-trivial change, read the implementation and the relevant tests before concluding.
- Verify the exact source when touching behavior, database logic, migrations, retries, fallbacks,
  recovery, or compatibility code.
- When the graph and the source disagree, the source wins. The graph may be stale or may not
  model that relationship.
- An empty graph result can mean "not indexed" or "not statically visible", not "does not exist".

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
<!-- /code-review-graph MCP tools -->
