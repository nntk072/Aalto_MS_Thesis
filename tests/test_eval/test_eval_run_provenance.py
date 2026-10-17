"""Evaluation must preserve input runs and reproduce training fit scope."""

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from quant_rl.eval import eval_run
from quant_rl.eval.reanalysis_provenance import (
    _feature_build_scope,
    _load_run_config,
    _validate_reanalysis_request,
)
from quant_rl.eval.run_manifest import file_sha256
from quant_rl.features.build import feature_cache_path


def test_checkpoint_reanalysis_defaults_to_deterministic_actions() -> None:
    default_args = eval_run._build_parser().parse_args(["--run", "source", "--output", "out"])
    sampled_args = eval_run._build_parser().parse_args(
        ["--run", "source", "--output", "out", "--stochastic", "--seed", "42"]
    )
    assert default_args.stochastic is False
    assert sampled_args.stochastic is True
    assert sampled_args.seed == 42


def test_stop_schema_requires_checkpoint_manifest_and_config_agreement() -> None:
    schema = {"menu": "recent_confirmed_unswept_mtf_swings_v1"}
    assert (
        eval_run._validate_stop_candidate_schema(
            checkpoint_schema=schema,
            manifest_schema=schema,
            expected_schema=schema,
            allow_legacy=False,
        )
        == "versioned"
    )
    with pytest.raises(ValueError, match="disagree"):
        eval_run._validate_stop_candidate_schema(
            checkpoint_schema=schema,
            manifest_schema={"menu": "legacy_last_accepted_swing_v1"},
            expected_schema=schema,
            allow_legacy=False,
        )
    with pytest.raises(ValueError, match="differs from the run config"):
        eval_run._validate_stop_candidate_schema(
            checkpoint_schema=schema,
            manifest_schema=schema,
            expected_schema={"menu": "legacy_last_accepted_swing_v1"},
            allow_legacy=False,
        )


def test_legacy_stop_schema_requires_opt_in_and_cannot_claim_expanded_menu() -> None:
    legacy = {"menu": "legacy_last_accepted_swing_v1"}
    with pytest.raises(ValueError, match="no stop-candidate schema"):
        eval_run._validate_stop_candidate_schema(
            checkpoint_schema=None,
            manifest_schema=None,
            expected_schema=legacy,
            allow_legacy=False,
        )
    assert (
        eval_run._validate_stop_candidate_schema(
            checkpoint_schema=None,
            manifest_schema=None,
            expected_schema=legacy,
            allow_legacy=True,
        )
        == "legacy_opt_in"
    )
    with pytest.raises(ValueError, match="cannot assert that menu was trained"):
        eval_run._validate_stop_candidate_schema(
            checkpoint_schema=None,
            manifest_schema=None,
            expected_schema={"menu": "recent_confirmed_unswept_mtf_swings_v1"},
            allow_legacy=True,
        )


def test_checkpoint_reanalysis_checks_manifest_artifact_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"saved checkpoint")
    expected = file_sha256(checkpoint)

    assert (
        eval_run._validate_checkpoint_manifest_hash(
            checkpoint, {"checkpoint": {"sha256": expected}}
        )
        == "verified"
    )
    with pytest.raises(ValueError, match="does not match"):
        eval_run._validate_checkpoint_manifest_hash(
            checkpoint, {"checkpoint": {"sha256": "0" * 64}}
        )
    assert eval_run._validate_checkpoint_manifest_hash(checkpoint, {}) == (
        "unavailable_legacy_manifest"
    )


@pytest.mark.parametrize(("stochastic", "expected_deterministic"), [(False, True), (True, False)])
def test_checkpoint_split_replay_passes_selected_action_mode(
    monkeypatch: pytest.MonkeyPatch, stochastic: bool, expected_deterministic: bool
) -> None:
    captured: dict[str, bool] = {}

    def fake_evaluate_model(
        _model: object, *, deterministic: bool, **_kwargs: object
    ) -> dict[str, object]:
        captured["deterministic"] = deterministic
        return {}

    monkeypatch.setattr(eval_run, "evaluate_model", fake_evaluate_model)
    result = eval_run._evaluate_checkpoint_split(object(), stochastic=stochastic)

    assert result == {}
    assert captured["deterministic"] is expected_deterministic


def test_reanalysis_requires_explicit_historical_access(tmp_path: Path) -> None:
    source = tmp_path / "historical"
    source.mkdir()
    with pytest.raises(ValueError, match="historical"):
        _validate_reanalysis_request(source, tmp_path / "review", allow_historical=False)


def test_reanalysis_refuses_input_and_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "historical"
    source.mkdir()
    with pytest.raises(ValueError, match="separate"):
        _validate_reanalysis_request(source, source, allow_historical=True)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "metrics.json").write_text("immutable")
    with pytest.raises(FileExistsError, match="exists"):
        _validate_reanalysis_request(source, destination, allow_historical=True)
    assert (destination / "metrics.json").read_text() == "immutable"


def test_reanalysis_allows_new_output_and_keeps_source(tmp_path: Path) -> None:
    source = tmp_path / "historical"
    source.mkdir()
    (source / "config.yaml").write_text("original")
    _validate_reanalysis_request(source, tmp_path / "review", allow_historical=True)
    assert (source / "config.yaml").read_text() == "original"


def test_evaluation_fails_closed_without_run_config_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "historical"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Run config snapshot missing"):
        _load_run_config(run_dir)


@pytest.mark.parametrize("contents", ["", "- 1\n- 2\n"])
def test_evaluation_rejects_nonmapping_run_snapshot(tmp_path: Path, contents: str) -> None:
    run_dir = tmp_path / "historical"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(contents)
    with pytest.raises(ValueError, match="configuration mapping"):
        _load_run_config(run_dir)


def test_evaluation_uses_exact_run_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "historical"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "env:\n  strategy_actions: true\n"
        "data:\n  split:\n    train_end: '2025-12-31'\n"
        "    test_start: '2026-01-01'\n"
    )
    cfg = _load_run_config(run_dir)
    assert cfg.env.strategy_actions is True


@pytest.mark.parametrize(
    "contents",
    [
        "env:\n  strategy_actions: true\n",
        "data:\n  split:\n    train_end: '2025-12-31'\n",
        "data:\n  split:\n    test_start: '2026-01-01'\n",
        "data:\n  split:\n    train_end: null\n    test_start: ''\n",
    ],
)
def test_evaluation_rejects_missing_saved_split_boundaries(tmp_path: Path, contents: str) -> None:
    run_dir = tmp_path / "historical"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(contents)
    with pytest.raises(ValueError, match="split"):
        _load_run_config(run_dir)


def test_eval_feature_scope_matches_training_mask_and_cache(tmp_path: Path) -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    bars = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=index)
    cfg = OmegaConf.create({"features": {"zscore_window": 3}})
    mask, cache = _feature_build_scope(bars, cfg, tmp_path, "US100", "2025-01-03")
    assert mask.tolist() == [True, True, True, False, False, False]
    assert cache == feature_cache_path(tmp_path, "US100", cfg, bars, train_mask=mask)
    assert cache != feature_cache_path(tmp_path, "US100", cfg, bars)
