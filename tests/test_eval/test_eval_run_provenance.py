"""Evaluation must preserve input runs and reproduce training fit scope."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from quant_rl.eval import eval_run, reanalysis_provenance
from quant_rl.eval.reanalysis_provenance import (
    _evaluation_dataset_records,
    _feature_build_scope,
    _load_exact_source_feature_cache,
    _load_run_config,
    _reanalysis_backend_disposition,
    _validate_reanalysis_datasets,
    _validate_reanalysis_feature_cache,
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


def test_execution_seed_uses_explicit_override_then_source_seed() -> None:
    manifest = {
        "seed": 50,
        "training_protocol": {"seed": 51},
        "metrics_protocol": {"execution_seed": 52},
    }

    assert eval_run._resolve_execution_seed(7, manifest) == 7
    assert eval_run._resolve_execution_seed(None, manifest) == 50


def test_execution_seed_rejects_negative_and_reports_missing_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        eval_run._resolve_execution_seed(-1, {})
    assert eval_run._resolve_execution_seed(None, {}) is None


def test_legacy_input_provenance_reanalysis_requires_explicit_opt_in() -> None:
    args = eval_run._build_parser().parse_args(["--run", "source", "--output", "out"])
    assert args.allow_legacy_data_provenance is False
    assert args.source_feature_cache is None
    opted_in = eval_run._build_parser().parse_args(
        [
            "--run",
            "source",
            "--output",
            "out",
            "--allow-legacy-data-provenance",
            "--source-feature-cache",
            "cache/baseline.parquet",
        ]
    )
    assert opted_in.allow_legacy_data_provenance is True
    assert opted_in.source_feature_cache == "cache/baseline.parquet"


def test_legacy_sequence_schema_reanalysis_requires_explicit_opt_in() -> None:
    default_args = eval_run._build_parser().parse_args(["--run", "source", "--output", "out"])
    opted_in = eval_run._build_parser().parse_args(
        ["--run", "source", "--output", "out", "--allow-legacy-sequence-schema"]
    )

    assert default_args.allow_legacy_sequence_schema is False
    assert opted_in.allow_legacy_sequence_schema is True


def test_checkpoint_replay_forwards_legacy_sequence_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate_model(
        _model: object, *, deterministic: bool, **kwargs: object
    ) -> dict[str, str]:
        captured["deterministic"] = deterministic
        captured.update(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr(eval_run, "evaluate_model", fake_evaluate_model)
    result = eval_run._evaluate_checkpoint_split(
        object(),
        stochastic=False,
        bars=pd.DataFrame(),
        features=pd.DataFrame(),
        legacy_sequence_columns=True,
    )

    assert result == {"status": "captured"}
    assert captured["deterministic"] is True
    assert captured["legacy_sequence_columns"] is True


def test_checkpoint_sequence_width_must_match_ordered_feature_schema() -> None:
    model = SimpleNamespace(
        observation_space=SimpleNamespace(
            spaces={"seq": SimpleNamespace(shape=(128, 7)), "account": object()}
        )
    )

    assert eval_run._validate_sequence_input_width(model, 7) == 7
    with pytest.raises(ValueError, match="sequence input width disagrees"):
        eval_run._validate_sequence_input_width(model, 8)


def test_reanalysis_feature_cache_requires_identity_or_legacy_opt_in(tmp_path: Path) -> None:
    cache = tmp_path / "features_v1.parquet"
    cache.write_bytes(b"features")
    (tmp_path / "features_v1.parquet.hash").write_text("abc123\n")

    with pytest.raises(ValueError, match="no feature-cache identity"):
        _validate_reanalysis_feature_cache(None, cache, allow_legacy=False)

    disposition, digest, file_digest = _validate_reanalysis_feature_cache(
        None, cache, allow_legacy=True
    )
    assert disposition == "legacy_unverified"
    assert digest == "abc123"
    assert file_digest == file_sha256(cache)


def test_reanalysis_dataset_parity_requires_exact_train_test_inputs() -> None:
    train = {
        "sha256": "train-digest",
        "rows": 2,
    }
    test = {
        "sha256": "test-digest",
        "rows": 1,
    }
    datasets = {
        "train_bars": train,
        "train_features": train,
        "test_bars": test,
        "test_features": test,
    }

    assert _validate_reanalysis_datasets(datasets, datasets, allow_legacy=False) == "verified"
    with pytest.raises(ValueError, match="dataset differs from training for test_features"):
        _validate_reanalysis_datasets(
            datasets,
            {**datasets, "test_features": {"sha256": "changed"}},
            allow_legacy=False,
        )
    assert _validate_reanalysis_datasets(None, datasets, allow_legacy=True) == "legacy_unverified"
    with pytest.raises(ValueError, match="incomplete train/test dataset identities"):
        _validate_reanalysis_datasets({"train_bars": train}, datasets, allow_legacy=True)


def test_evaluation_dataset_records_include_splits_secondary_and_ticks() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="min", tz="UTC")
    frame = pd.DataFrame({"close": [100.0, 101.0]}, index=index)

    class TickBook:
        _ts = pd.Series([1, 2]).to_numpy(dtype="int64")
        _bid = pd.Series([100.0, 101.0]).to_numpy(dtype="float64")
        _ask = pd.Series([100.1, 101.1]).to_numpy(dtype="float64")

    result = _evaluation_dataset_records(
        train_bars=frame,
        train_features=frame,
        test_bars=frame,
        test_features=frame,
        train_secondary=frame,
        test_secondary=frame,
        primary_ticks=TickBook(),
    )
    assert set(result) == {
        "train_bars",
        "train_features",
        "test_bars",
        "test_features",
        "train_secondary",
        "test_secondary",
        "primary_ticks",
    }
    assert result["primary_ticks"]["rows"] == 2


@pytest.mark.parametrize(
    ("source", "evaluation", "expected"),
    [
        ({"data_source": "frame"}, "frame", ("frame", "same_backend")),
        ({"data_source": "frames"}, "frame", ("frames", "same_backend")),
        (
            {"data_source": "bundle"},
            "frame",
            ("bundle", "cross_backend_requires_checkpoint_parity"),
        ),
        ({"data_source": "bundle"}, "bundle", ("bundle", "same_backend")),
        (None, "frame", ("unknown", "legacy_unverified")),
    ],
)
def test_reanalysis_records_backend_parity_disposition(
    source: object, evaluation: str, expected: tuple[str, str]
) -> None:
    assert _reanalysis_backend_disposition(source, evaluation) == expected


def test_reanalysis_feature_cache_requires_exact_training_identity_and_hash(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "features_v1.parquet"
    cache.write_bytes(b"features")
    (tmp_path / "features_v1.parquet.hash").write_text("abc123\n")
    cache_file_digest = file_sha256(cache)
    source = {
        "identity": cache.name,
        "sha256": "abc123",
        "file_sha256": cache_file_digest,
    }

    assert _validate_reanalysis_feature_cache(source, cache, allow_legacy=False) == (
        "verified",
        "abc123",
        cache_file_digest,
    )
    with pytest.raises(ValueError, match="identity differs from training"):
        _validate_reanalysis_feature_cache(
            {"identity": "other.parquet", "sha256": "abc123"},
            cache,
            allow_legacy=False,
        )
    with pytest.raises(ValueError, match="content identity differs from training"):
        _validate_reanalysis_feature_cache(
            {"identity": cache.name, "sha256": "different"},
            cache,
            allow_legacy=False,
        )
    with pytest.raises(ValueError, match="file SHA-256 differs from training"):
        _validate_reanalysis_feature_cache(
            {**source, "file_sha256": "0" * 64},
            cache,
            allow_legacy=False,
        )
    with pytest.raises(ValueError, match="feature_cache.identity is missing"):
        _validate_reanalysis_feature_cache({}, cache, allow_legacy=True)
    partial_source = {"identity": cache.name, "sha256": "abc123"}
    with pytest.raises(ValueError, match="feature_cache.file_sha256 is missing"):
        _validate_reanalysis_feature_cache(partial_source, cache, allow_legacy=False)
    assert _validate_reanalysis_feature_cache(partial_source, cache, allow_legacy=True) == (
        "legacy_unverified",
        "abc123",
        cache_file_digest,
    )


def test_exact_source_feature_cache_replay_requires_manifest_file_identity(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "baseline.parquet"
    index = pd.date_range("2025-01-01", periods=2, freq="min", tz="UTC")
    expected = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
    expected.to_parquet(cache)
    (tmp_path / "baseline.parquet.hash").write_text("content-hash\n")
    source = {
        "identity": cache.name,
        "sha256": "content-hash",
        "file_sha256": file_sha256(cache),
    }

    loaded, resolved, cache_hash, file_hash = _load_exact_source_feature_cache(source, cache)
    pd.testing.assert_frame_equal(loaded, expected, check_freq=False)
    assert resolved == cache.resolve()
    assert cache_hash == "content-hash"
    assert file_hash == source["file_sha256"]

    with pytest.raises(ValueError, match="file SHA-256 differs from training"):
        _load_exact_source_feature_cache({**source, "file_sha256": "0" * 64}, cache)

    with pytest.raises(ValueError, match="requires feature_cache.file_sha256"):
        _load_exact_source_feature_cache({"identity": cache.name, "sha256": "content-hash"}, cache)


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


def test_eval_feature_scope_honors_and_hash_validates_prebuilt_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    bars = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=index)
    cache = tmp_path / "prebuilt.parquet"
    cache.write_bytes(b"cache")
    (tmp_path / "prebuilt.parquet.hash").write_text("expected\n")
    cfg = OmegaConf.create(
        {
            "data": {"prebuilt_feature_cache": str(cache)},
            "features": {"zscore_window": 3, "po3_reference_mode": "m1_legacy"},
        }
    )
    monkeypatch.setattr(
        reanalysis_provenance,
        "feature_cache_content_hash",
        lambda *_args, **_kwargs: "expected",
    )

    mask, resolved = _feature_build_scope(bars, cfg, tmp_path, "US100", "2025-01-03")
    assert mask.tolist() == [True, True, True, False, False, False]
    assert resolved == cache

    (tmp_path / "prebuilt.parquet.hash").write_text("stale\n")
    with pytest.raises(ValueError, match="does not match the saved config/data hash"):
        _feature_build_scope(bars, cfg, tmp_path, "US100", "2025-01-03")
