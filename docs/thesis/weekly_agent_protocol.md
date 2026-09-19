# Weekly RL Progress Agent Protocol

This is one reusable agent plan for the weekly Aalto MS thesis progress cycle.
It is not a week-by-week research schedule. The agent repeats the same
end-to-end process every week for the three current strategy variants:

- `baseline`
- `po3_ifvg` (PO3 + IFVG)
- `distribution`

The weekly objective may change, but every report must trace:

```text
dataset -> cleaning/resampling -> features -> split/masks -> environment
-> model -> training -> checkpoint -> evaluation -> metrics/plots -> report
```

The professor must be able to understand the current status without reading
source code or raw logs: what was attempted, what is correct, what is wrong,
where the first problem occurred, which agent is leading, and what should be
improved next.

## Agent workflow

### 1. Establish the weekly objective

Read the previous report, open issues, failed runs, accepted fixes, and
supervisor decisions. State:

- the question this week should answer;
- expected results and acceptance criteria;
- unresolved issues carried from the previous week;
- exact experiments for all three agents.

Do not repeat an old experiment unless the report explains why.

### 2. Audit the input chain

Before training, record and verify:

- raw paths, symbols, timezone, date coverage, and row counts;
- duplicates, missing bars, invalid OHLCV values, and cleaning actions;
- resampling outputs and row counts;
- feature names, shapes, NaNs, cache version, and content hash;
- train, validation, and locked-OOS dates and masks;
- train-only normalization and other leakage checks;
- every difference from the previous week's data or configuration.

If an input is missing, unexpectedly changed, or violates an invariant, mark
the affected run invalid and report the blocker before interpreting it.

### 3. Register the experiment matrix

For every run, record strategy, command, git commit, config hash/diff, seed,
timestep budget, algorithm, encoder, reward, costs, risk limits,
hardware/environment, output directory, expected artifacts, and run role.

Keep data, splits, evaluation, and budgets matched across agents. Change only
the strategy overlay unless the experiment explicitly tests another factor.
Use roles such as `smoke`, `development`, `selection`, `robustness`, and
`final_report_only`.

### 4. Execute and validate each stage

For each agent, record:

1. input discovery;
2. preprocessing and feature loading/building;
3. split and mask creation;
4. environment construction;
5. model construction;
6. training and progress logging;
7. checkpoint save and reload;
8. IS/validation evaluation;
9. metric calculation, plots, and export.

At every stage compare expected and actual dates, shapes, columns, action
semantics, reward ranges, episode lengths, trade counts, and artifacts. Save
logs and outputs in unique run directories; never overwrite evidence.

### 5. Diagnose before making claims

For every anomaly or failed run, document:

```text
symptom
-> earliest stage that diverged
-> hypothesis
-> diagnostic test
-> evidence
-> confirmed/rejected root cause
-> correction or workaround
-> rerun result
-> effect on the thesis conclusion
```

Separate data, feature/configuration, environment, training, checkpoint, and
evaluation/reporting problems. Failed, incomplete, and invalid runs remain in
the report and are never silently removed.

### 6. Compare the agents fairly

For every agent and seed, report return, Sharpe, Sortino, maximum drawdown,
volatility, win rate, profit factor, trade count, turnover, exposure, costs,
risk-limit breaches, coverage, and confidence intervals when available.
Include learning curves, equity curves, and relevant diagnostic plots.

State the current leader metric by metric, including sample size, uncertainty,
and whether the comparison is fair. Do not declare an overall winner from one
seed or one metric, and do not use locked OOS for selection.

### 7. Recommend improvements

End with evidence-based recommendations, classified as a data/feature fix,
configuration/environment fix, training/seed/budget change, evaluation/report
change, rerun, stop decision, or supervisor approval request.

Explicitly state what is correct, what is wrong until now, what remains
uncertain, the risk to the thesis, and what the next run should resolve.

### 8. Apply validation gates

- Validate the pipeline with smoke runs before increasing the budget.
- Use purged walk-forward results and seed stability for selection.
- Use matched seeds and budgets for ablations and robustness comparisons.
- Freeze code, configs, checkpoints, and manifests before final OOS.
- Reserve locked OOS for the final report; never tune after viewing it.
- Obtain human approval before changing locked dates, feature definitions,
  evaluation rules, selection criteria, or final OOS protocol.

## Weekly supervisor report

Create one report every week with this fixed structure:

1. **Executive status:** where the project is now.
2. **Objective and acceptance criteria:** what the week tested.
3. **Previous issues:** closed, carried forward, blocked, or reopened.
4. **Dataset/input audit:** full lineage, checks, hashes, and changes.
5. **Experiment matrix:** every run for all three agents, including failures.
6. **Pipeline trace:** expected versus actual result at each stage.
7. **Results:** per-agent/per-seed metrics, plots, and artifact links.
8. **What is correct:** verified invariants and trustworthy outputs.
9. **What is wrong:** anomalies, missing outputs, invalid runs, and blockers.
10. **Root-cause register:** symptom-to-rerun chain for every problem.
11. **Current comparison:** leaders, uncertainty, fairness, and caveats.
12. **Suggested improvements:** evidence-based actions and priority.
13. **Risks and decisions:** threats to validity and approvals needed.
14. **Next action:** commands, expected outputs, and acceptance criteria.

All three agents must appear even when one fails. A failed run is a result with
a cause and status, not an empty table cell.

## Required artifacts

```text
outputs/thesis_progress/
  protocol.yaml
  manifests/
    week_<N>.json
    week_<N>_batch_<M>.json
  week_<N>/
    report.md
    dataset_audit.json
    issue_root_cause_register.md
    comparison.csv
    batches/
      batch_<M>/
        manifest.json
        objective.md
        dataset_audit.json
        comparison.csv
        baseline/
          runs/<run_id>/{config.yaml,metrics.json,plots/,logs/}
        po3_ifvg/
          runs/<run_id>/{config.yaml,metrics.json,plots/,logs/}
        distribution/
          runs/<run_id>/{config.yaml,metrics.json,plots/,logs/}
    weekly_summary/
      comparison.csv
      leader_analysis.md
      recommendations.md
  final_oos/
```

`week_<N>` is the reporting boundary, not a single experiment. The agent may
create multiple `batch_<M>` directories in one week for independent questions,
debugging attempts, retries, ablations, or robustness checks. Each batch must
contain its own objective, input audit, experiment matrix, result artifacts,
and issue/root-cause evidence. A retry must receive a new `run_id`; it must
never overwrite the failed run.

The weekly `report.md` consolidates all batches and must state:

- which batches were completed, blocked, retried, or invalid;
- how each batch changed the current understanding;
- which results are comparable and which must not be pooled;
- the current leader across valid comparable batches;
- unresolved problems and the suggested next experiment.

## Orchestrator CLI

The focused orchestrator creates a new immutable run directory for every
strategy/seed attempt. It invokes the canonical training entrypoint and keeps
failed attempts in the manifests; a retry gets a new directory and a
`retry_of` link.

```bash
uv run python scripts/thesis_orchestrator.py \
  --week 3 --batches 2 --seeds 42 43 \
  --strategies baseline po3_ifvg distribution \
  --steps 50000 --output-root outputs/thesis_progress --retries 1
```

The command writes `week_3/manifest.json`, a report for each
`batches/batch_<M>/`, and `week_3/report.md`. Run artifacts are linked from
both Markdown reports. Use a distinct `--week` for a new reporting boundary;
do not delete or reuse a run directory when rerunning a failed attempt.
`dataset_hash`, train/test dates, and evaluation protocol can be supplied by
using `BatchSpec` from `quant_rl.orchestration.weekly` when embedding the
orchestrator in a research script.

Every reported number must link to its batch, run, dataset hash, code commit,
config, command, checkpoint, log, and plot. Negative or inconclusive results
are valid when they are reproducible and honestly explained.

## Repository constraints

Use the canonical `quant_rl.train.train_rl` entrypoint and shared evaluation
and export code. Read [`AGENTS.md`](../../AGENTS.md), the relevant
`.agents/domain_maps/*.md`, and
[`validation_protocol.md`](validation_protocol.md) before experiments.
Preserve the chronological train/OOS split, purged walk-forward protocol, and
locked-OOS rule.
