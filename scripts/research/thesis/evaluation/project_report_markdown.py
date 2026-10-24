"""Render the project and run context as explanatory Markdown."""

from __future__ import annotations

from typing import Any

from scripts.research.thesis.evaluation.project_report_context import GLOSSARY


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _inline(value: Any) -> str:
    return f"{chr(96)}{value}{chr(96)}"


def _plain(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        return ", ".join(_plain(item) for item in value) if value else "None"
    if isinstance(value, dict):
        return (
            "; ".join(f"{key.replace('_', ' ')}: {_plain(item)}" for key, item in value.items())
            or "None recorded"
        )
    return str(value).replace("|", "\\|").replace("\n", " ")


def _hash_prefix(value: Any) -> str:
    return _inline(str(value)[:16]) if value else "Not recorded"


def render_project_context(context: dict[str, Any]) -> str:
    """Render architecture, recorded provenance, and training setup for non-experts."""
    fence = chr(96) * 3
    lines = [
        "## How the repository produced these results",
        "",
        "This section connects recorded input files and caches to the model, training run, evaluation, and plots. It is generated from source paths and saved manifests; missing settings stay visible as Not recorded.",
        "",
        "### Pipeline architecture",
        "",
        f"{fence}mermaid",
        str(context.get("pipeline_diagram", "")).rstrip(),
        fence,
        "",
        f"Standalone architecture diagram: [{context.get('pipeline_diagram_file', 'not recorded')}]({context.get('pipeline_diagram_file', '')}).",
        "",
        "| Stage | What it does | Source code |",
        "|---|---|---|",
    ]
    for stage in context.get("source_map", []):
        sources = ", ".join(
            f"[{item['path']}]({item['report_link']})"
            if item.get("exists")
            else f"{_inline(item['path'])} (not found)"
            for item in stage.get("sources", [])
        )
        lines.append(f"| {stage['name']} | {stage['purpose']} | {sources} |")

    repo = _dict(context.get("current_repository"))
    lines.extend(
        [
            "",
            "### Recorded source and data",
            "",
            f"Report-generator checkout: {_inline(repo.get('path') or 'not recorded')} at {_inline(repo.get('head') or 'not recorded')}.",
            f"Working tree when generated: {repo.get('tracked_changes', 'unknown')} tracked changes and {repo.get('untracked_changes', 'unknown')} untracked files.",
            f"Campaign source revision: {_inline(context.get('campaign_source_revision') or 'not recorded')}. Each run source is listed below; revisions are not assumed to match.",
        ]
    )
    snapshot = _dict(context.get("campaign_source_snapshot"))
    if snapshot:
        lines.append(
            f"Campaign source snapshot: {_inline(snapshot.get('path', 'not recorded'))}; {'present' if snapshot.get('present_at_report_time') else 'missing'}; archive SHA-256 {_inline(snapshot.get('sha256', 'not recorded'))}; {snapshot.get('file_count', 'unknown')} files; file-hash manifest SHA-256 {_inline(snapshot.get('manifest_sha256', 'not recorded'))}; {snapshot.get('source_file_hash_count', 'unknown')} source hashes in the campaign manifest."
        )
    else:
        lines.append("Campaign source snapshot: not recorded in the campaign manifest.")
    lines.append("Report generator source digests:")
    for item in context.get("report_generator_sources", []):
        lines.append(
            f"- {_inline(item.get('path'))}: {_inline(item.get('sha256') or 'not available')}"
        )
    lines.extend(
        [
            "",
            "#### Campaign input files",
            "",
            "Hashes are the values captured in the campaign manifest. File presence is checked separately at report time.",
            "",
            "| Split/source role | File | Timeframe in filename | Present now | SHA-256 prefix |",
            "|---|---|---|---|---|",
        ]
    )
    for item in context.get("raw_inputs", []):
        lines.append(
            f"| {item.get('role', 'unknown')} | {_inline(item.get('path', 'not recorded'))} | {item.get('timeframe_in_filename', 'unknown')} | {'Yes' if item.get('present_at_report_time') else 'No'} | {_hash_prefix(item.get('sha256'))} |"
        )
    if not context.get("raw_inputs"):
        lines.append("| Not recorded | Not recorded | Not recorded | Not checked | Not recorded |")

    cache = _dict(context.get("cache_summary"))
    lines.extend(["", "#### Prepared data/cache inventory", ""])
    if cache:
        lines.append(
            f"Cache directory: {_inline(cache.get('cache_dir', 'not recorded'))}; {cache.get('cache_files', 'unknown')} cache files; {cache.get('tick_cache_files', 'unknown')} tick-cache files."
        )
        discretization = _dict(cache.get("price_discretization"))
        if discretization:
            lines.append(
                f"Price discretization: {'enabled' if discretization.get('enabled') else 'disabled'}; increment {_inline(discretization.get('increment', 'not recorded'))}; rounding {_inline(discretization.get('rounding', 'not recorded'))}."
            )
        lines.extend(["", "| Instrument | Timeframe | Cached rows |", "|---|---|---:|"])
        for symbol, rows in sorted(_dict(cache.get("bars")).items()):
            for timeframe, count in sorted(_dict(rows).items()):
                lines.append(f"| {symbol} | {timeframe} | {count} |")
        for symbol, count in sorted(_dict(cache.get("ticks")).items()):
            lines.append(f"| {symbol} | Tick quotes | {count} |")
    else:
        lines.append("No split_cache_ready.json inventory was saved with this campaign.")

    lines.extend(
        [
            "",
            "#### Run data actually recorded",
            "",
            "Date boundaries are from each run manifest. Observed row timestamps can start later than the configured boundary when no bar is available at that instant.",
            "",
        ]
    )
    for run in context.get("run_evidence", []):
        lines.extend(
            [f"##### {run.get('variant', 'Variant')} (seed {_plain(run.get('seed'))})", ""]
        )
        if not run.get("manifest_available"):
            lines.append(
                "Run manifest missing; dataset, cache, and architecture details are unavailable."
            )
            continue
        split = _dict(run.get("split"))
        source = _dict(run.get("repository_at_run"))
        source_head = source.get("head")
        relation = (
            "matches"
            if source.get("matches_report_checkout_head") is True
            else "differs from"
            if source.get("matches_report_checkout_head") is False and source_head
            else "cannot be compared with"
        )
        lines.append(
            f"Training dates: {_inline(split.get('train_start', 'not recorded'))} to {_inline(split.get('train_end', 'not recorded'))}; configured test boundary: {_inline(split.get('test_start', 'not recorded'))}."
        )
        lines.append(
            f"Run source revision: {_inline(source_head or 'not recorded')}; checkout path: {_inline(source.get('path') or 'not recorded')}; it {relation} the report-generator checkout."
        )
        dirty = source.get("dirty_files", [])
        if dirty:
            lines.append(
                "Recorded dirty files at training time: "
                + "; ".join(
                    f"{_inline(item.get('path', 'unknown'))} (SHA-256 {_inline(str(item.get('sha256', 'not recorded'))[:16])})"
                    for item in dirty
                    if isinstance(item, dict)
                )
                + "."
            )
        lines.extend(
            [
                "",
                "| Saved dataset | Observed start | Observed end | Rows | Columns | SHA-256 prefix |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for dataset in run.get("datasets", []):
            lines.append(
                f"| {dataset['name']} | {dataset.get('start') or 'not recorded'} | {dataset.get('end') or 'not recorded'} | {_plain(dataset.get('rows'))} | {_plain(dataset.get('column_count'))} | {_hash_prefix(dataset.get('sha256'))} |"
            )
        cache_identity = _dict(run.get("feature_cache"))
        if cache_identity:
            lines.append(
                f"Feature cache: {_inline(cache_identity.get('identity', 'not recorded'))}; cache key {_inline(cache_identity.get('sha256', 'not recorded'))}; file digest {_inline(cache_identity.get('file_sha256', 'not recorded'))}; fit scope {_inline(run.get('feature_fit_scope') or 'not recorded')}."
            )
        if run.get("missing_evidence"):
            lines.append(
                "Missing manifest fields: "
                + ", ".join(_inline(key) for key in run["missing_evidence"])
                + "."
            )

    lines.extend(["", "### Model and training setup", ""])
    for run in context.get("run_evidence", []):
        if not run.get("manifest_available"):
            continue
        model = _dict(run.get("model"))
        training = _dict(run.get("training"))
        config_identity = _dict(run.get("config_identity"))
        checkpoint = _dict(run.get("checkpoint"))
        action = _dict(model.get("action_schema"))
        observation = _dict(model.get("observation_schema"))
        lines.extend(
            [
                f"#### {run.get('variant')} (seed {_plain(run.get('seed'))})",
                "",
                f"The recorded learner is **{model.get('algorithm') or 'not recorded'}** with encoder {_inline(model.get('encoder') or 'not recorded')}. The encoder converts ordered feature history into a compact representation; the policy head uses it to choose actions.",
                "",
                "| Setting | Recorded value |",
                "|---|---|",
                f"| Encoder channels / dilation levels | {_plain(model.get('channels'))} / {_plain(model.get('dilations'))} |",
                f"| Kernel / encoded dimension | {_plain(model.get('kernel_size'))} / {_plain(model.get('latent_dimension'))} |",
                f"| Policy head / parameter count | {_plain(model.get('policy_head'))} / {_plain(model.get('parameter_count'))} |",
                f"| Action schema | {_inline(action.get('version', 'not recorded'))}, {_plain(action.get('width'))} coordinates |",
                f"| Observation schema | {_inline(observation.get('version', 'not recorded'))}, {_plain(observation.get('observation_width'))} account coordinates |",
                f"| Requested / completed steps | {_plain(training.get('requested_steps'))} / {_plain(training.get('completed_steps'))} |",
                f"| Environments × rollout bars each | {_plain(training.get('parallel_environments'))} × {_plain(training.get('rollout_steps_per_environment'))} |",
                f"| Batch / epochs / learning rate | {_plain(training.get('batch_size'))} / {_plain(training.get('epochs'))} / {_plain(training.get('learning_rate'))} |",
                f"| Action distribution during training | {_inline(training.get('action_distribution') or 'not recorded')} |",
                f"| Configuration SHA-256 | {_inline(config_identity.get('sha256') or 'not recorded')} |",
                f"| Checkpoint SHA-256 | {_inline(checkpoint.get('sha256') or 'not recorded')} |",
            ]
        )
        if training.get("reward"):
            lines.append(f"| Reward settings | {_plain(training['reward'])} |")
        if training.get("risk"):
            lines.append(f"| Risk settings | {_plain(training['risk'])} |")

    lines.extend(
        [
            "",
            "### Evaluation and result figures",
            "",
            f"Core dataset hashes match across the saved variants: **{'Yes' if context.get('cross_variant_data_hashes_align') else 'No' if context.get('cross_variant_data_hashes_align') is False else 'Not comparable from available manifests'}**.",
            "Each variant section below contains saved training/testing metrics, trade exit, stop, target, and risk tables, learner diagnostics, and generated figures.",
            "A test period labeled previously inspected is historical evidence; it is not an untouched final holdout.",
            "",
            "| Variant | Test period role | Evaluation action mode | Tick execution | Random fill latency | Evaluation seed |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for run in context.get("run_evidence", []):
        evaluation = _dict(run.get("evaluation"))
        lines.append(
            f"| {run.get('variant', 'unknown')} | {evaluation.get('test_role') or 'not recorded'} | {evaluation.get('action_mode') or 'not recorded'} | {'Yes' if evaluation.get('tick_execution') else 'No or not recorded'} | {_plain(evaluation.get('random_fill_latency_ms'))} ms | {_plain(evaluation.get('action_seed'))} |"
        )
        market_context = _dict(evaluation.get("policy_context"))
        if market_context.get("features"):
            features = ", ".join(_inline(name) for name in market_context["features"])
            lines.append(
                f"Recorded schedule-aware context: {features}; schedule SHA-256 {_inline(market_context.get('schedule_sha256', 'not recorded'))}."
            )

    lines.extend(["", "### Plain-language guide", ""])
    lines.extend(f"- **{label}:** {description}" for label, description in GLOSSARY.items())
    return "\n".join(lines).rstrip() + "\n"
