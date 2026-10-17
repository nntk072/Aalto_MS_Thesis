# Aalto_MS_Thesis — Agent Guide

Quantitative RL trading system. PPO/SAC agents learn entry/exit timing on US100/US500
using structure-aware features (SMT divergence, PO3 state, liquidity sweeps, FVG zones).

## Active research roadmap

Use [`docs/thesis/master_project_execution_plan.md`](docs/thesis/master_project_execution_plan.md)
as the single operational queue for thesis readiness, the active campaign,
throughput work, and Q0–Q7. Older plans are technical references and evidence,
not parallel work queues. Preserve the roadmap's stage gates and current
campaign safety constraints.

## Build & Test

```bash
uv sync                              # install deps
python scripts/data/prepare_data.py       # raw CSV → parquet → features
python -m quant_rl.train.train_rl --mvp   # smoke test (30 days)

# Local (implementation): scoped checks only
uv run ruff format --check <touched paths>
uv run ruff check <touched paths>
uv run pytest <relevant test files> -q

# Agent commit/push gate — REQUIRED (do not skip)
# Format/lint/types MUST be full-tree (`.`). Never substitute path-scoped mypy/ruff.
uv run ruff format --check .    # if fails: uv run ruff format . then re-check
uv run ruff check .
uv run mypy .                   # REQUIRED full tree — not mypy <touched paths>
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
For node architecture, virtual environments, Lustre versus local storage,
installation and diagnostics, read
[`docs/operations/triton_environment.md`](docs/operations/triton_environment.md).
For intermittent Mypy startup stalls on Lustre (ARM64 or x86_64), follow
[`docs/operations/triton_mypy_lustre.md`](docs/operations/triton_mypy_lustre.md);
use the full-tree Mypy gate even when staging the executable on node-local storage.

## Triton (Aalto HPC)

GPU training on Triton: see [`.agents/rules/triton-slurm.md`](.agents/rules/triton-slurm.md).
Helpers: [`scripts/triton/`](scripts/triton/) (`status.sh`, `find_gpu_session.sh`, `attach_or_alloc.sh`).

- **Reuse first:** if a GPU tmux/`srun` already exists, run train/eval/tests
  there (`tmux send-keys`).
- **GPU allocation:** standing user authorization permits necessary thesis
  allocations. Reuse an existing allocation first; before a new one, inspect
  every candidate node's live GPU availability, allocated/free CPUs, and
  allocated/free host RAM. Never use `srun --test-only` or scheduler forecasts;
  when a compatible GPU is idle, make one real bounded request from a named,
  persistent login tmux session with `--immediate=60`. Never launch a new
  `srun` directly from a transient Codex/SSH terminal. Stop polling after 60
  seconds. Leave pending requests untouched and inspect alternatives without
  creating an overlapping request.
  Do not create short diagnostic jobs or fan out Slurm steps.
- **Cancellation:** never run `scancel`. Preserve active and pending
  allocations; never pass `REPLACE=1`.
- **Before alloc:** first check the user's current allocation and reuse it if
  available; otherwise check `sinfo`/`squeue` for free GPU/CPU/RAM and apply the
  partition and GPU-family guidance in `.agents/rules/triton-slurm.md`. The
  user requires a per-node CPU, host-RAM, and GPU inventory before each real
  allocation request. The agent must wait/poll for an eligible allocation for
  no more than 60 seconds. For `srun`/`salloc`, pass Slurm's immediate timeout
  (`--immediate=60`) where supported; never leave a blocking allocation command
  waiting longer. For a submitted job, stop polling after 60 seconds.
  This limits the agent's wait, not an allocation's runtime. Do not cancel a
  pending request at the cutoff. Before considering an alternative, inspect
  every candidate node's allocated and free CPUs, host RAM, and
  configured/allocated GPU inventory. Do not use `srun --test-only`. An approved
  successor behind an explicit
  dependency starts its 60-second resource-wait clock only after the dependency
  clears. MI210 is excluded for the current thesis work. Preserve the
  cancellation approval rules for every existing or unrelated job.
  The current thesis campaign has a user-set hard cap of **2 GH200 GPUs total**
  across all active allocations, and at most 2 GH200s in any one request. This
  is an aggregate cap: do not combine jobs or node requests whose GH200 counts
  sum to more than 2 (for example, 1 on gpuarm1 plus 2 on gpuarm2, or 2 on
  gpuarm1 plus 1 on gpuarm2). Prefer one GH200 on each of two nodes; if that
  layout is unavailable, two GH200s on one eligible node are permitted within
  the same aggregate cap. A
  single 2-GH200 successor may be queued only with a
  Slurm dependency that guarantees it starts after the active allocation ends;
  include the pending successor in the aggregate check. This cap does not
  apply to other GPU families. The
  2026-10-16 V100 probe uses 1 V100 shared by 2 trainers;
  `.venv-x86` cu130 is incompatible with V100, so use `.venv-v100` only after
  checking `sm_70` support and a real CUDA operation. Match venv architecture
  (`.venv` on GH200/aarch64, `.venv-x86` on H100/H200 x86_64). Login has no GPU
  and not enough RAM for `n_envs=64`.
- **Never `scancel`** when a train finishes or for any other reason; keep the
  GPU bash + tmux allocation available for subsequent work.
- Login SSH for `ls` / `squeue` / `sinfo` / `tmux capture-pane` / inspect-only
  `scripts/triton/*.sh`; never `srun` for those checks.
- **Short-step monitor hard gate:** any job/step under 120 seconds counts as
  short, including child steps inside a long allocation. Create zero new short
  Slurm jobs/steps. Before launch/resume, inspect the full job/step topology,
  including retries and fan-out; never use one `srun` per worker, variant,
  retry, test, monitor, or node-local command. Run short work inside an existing
  shell/allocation without another step. If unexpected early exits or steps
  appear, stop submitting/retrying, preserve logs and allocation state, and
  diagnose. Never cancel Slurm jobs or allocations. Follow the
  detailed hard gate in `.agents/rules/triton-slurm.md`.
