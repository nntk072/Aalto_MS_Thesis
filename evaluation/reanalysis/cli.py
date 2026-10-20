"""Command-line configuration for historical checkpoint reanalysis."""

from __future__ import annotations

import argparse


def build_reanalysis_parser() -> argparse.ArgumentParser:
    """Build the checkpoint-reanalysis CLI with an explicit action mode."""
    parser = argparse.ArgumentParser(
        description="Reanalyse a PPO checkpoint into a separate, new output directory."
    )
    parser.add_argument("--run", required=True, help="Existing source RL run (read-only)")
    parser.add_argument("--output", required=True, help="New, non-existing reanalysis directory")
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Acknowledge that this run's historical test split is already inspected",
    )
    parser.add_argument(
        "--allow-legacy-action-schema",
        action="store_true",
        help="Explicitly opt in to an unversioned checkpoint using the selected legacy layout",
    )
    parser.add_argument(
        "--allow-legacy-observation-schema",
        action="store_true",
        help="Explicitly opt in to an unversioned account-state-v1 checkpoint",
    )
    parser.add_argument(
        "--allow-legacy-sequence-schema",
        action="store_true",
        help=(
            "Explicitly replay a manifest without ordered model-feature coordinates using "
            "the pre-filter sequence columns; the schema remains legacy_unverified"
        ),
    )
    parser.add_argument(
        "--allow-legacy-stop-schema",
        action="store_true",
        help=(
            "Explicitly opt in when an older checkpoint or run manifest has no stop-candidate "
            "schema; this does not assert that it used the expanded multi-HTF menu"
        ),
    )
    parser.add_argument(
        "--allow-legacy-data-provenance",
        action="store_true",
        help=(
            "Explicitly allow reanalysis when the source run lacks feature-cache or dataset "
            "identities; the resulting report will label input parity unverified"
        ),
    )
    parser.add_argument(
        "--source-feature-cache",
        default=None,
        help=(
            "Read-only replay of a preserved training cache; its filename, sidecar content hash, "
            "file SHA-256, saved config, and train/test feature fingerprints must match the run"
        ),
    )
    parser.add_argument(
        "--legacy-action-layout",
        default=None,
        help=(
            "Explicit layout for an unversioned checkpoint; must match the run config "
            "(for example 5, 6_dir, 6_sl, 7, 8, 11, discrete-20). Use legacy4_no_exit_v1 only "
            "for the documented 4D no-exit policy migration."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (relative to --run or absolute); defaults to "
        "model/ppo_final.zip, else the highest-step model/ppo_ckpt_*_steps.zip",
    )
    parser.add_argument(
        "--max-charts",
        type=int,
        default=None,
        help="Cap on per-trade charts per split (default: config output.max_order_charts, else 0).",
    )
    parser.add_argument(
        "--save-html",
        dest="save_html",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render interactive HTML (default: config output.save_html).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Opt in to sampled policy actions; deterministic actions are the default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed NumPy and PyTorch. This makes --stochastic evaluation repeatable; "
            "without --stochastic, policy actions remain deterministic."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Force the data pipeline to rebuild caches"
    )
    parser.add_argument(
        "--train-only-screen",
        action="store_true",
        help=(
            "Evaluate only the training split and write a candidate-screen report; the test "
            "split is not evaluated or reported"
        ),
    )
    return parser
