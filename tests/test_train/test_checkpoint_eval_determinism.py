from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from learning.rl import callbacks
from training.setup import environments as training_environments


def test_best_checkpoint_callback_uses_deterministic_policy_actions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class Env:
        account = SimpleNamespace(equity=100_000.0)
        trade_log = [{"type": "close"}]
        breach_events: list[dict[str, Any]] = []

        def reset(self) -> tuple[int, dict[str, Any]]:
            return 0, {}

        def step(self, _action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 1, 0.0, True, False, {}

    class Model:
        num_timesteps = 1

        def predict(self, _obs: int, *, deterministic: bool) -> tuple[int, None]:
            assert deterministic is True
            return 0, None

    monkeypatch.setattr(callbacks, "save_ppo_checkpoint", lambda *_args, **_kwargs: None)
    callback = callbacks.BestCheckpointEvalCallback(
        eval_env_factory=Env,
        eval_freq=1,
        best_model_path=tmp_path / "ppo_best",
        n_eval_episodes=1,
    )
    callback.model = cast(Any, Model())

    callback._run_eval()

    assert callback.best_end_equity == 100_000.0
    assert callback.best_checkpoint_status == "ACTIVE_CHECKPOINT_AVAILABLE"
    assert callback.best_n_trades == 1


def test_inactive_checkpoint_is_not_saved(tmp_path: Path, monkeypatch: Any) -> None:
    class Env:
        account = SimpleNamespace(equity=100_000.0)
        trade_log: list[dict[str, Any]] = []
        breach_events: list[dict[str, Any]] = []

        def reset(self) -> tuple[int, dict[str, Any]]:
            return 0, {}

        def step(self, _action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 1, 0.0, True, False, {}

    class Model:
        num_timesteps = 1

        def predict(self, _obs: int, *, deterministic: bool) -> tuple[int, None]:
            assert deterministic is True
            return 0, None

    saved: list[Path] = []
    monkeypatch.setattr(
        callbacks, "save_ppo_checkpoint", lambda _model, path: saved.append(Path(path))
    )
    callback = callbacks.BestCheckpointEvalCallback(
        eval_env_factory=Env,
        eval_freq=1,
        best_model_path=tmp_path / "ppo_best",
        n_eval_episodes=1,
    )
    callback.model = cast(Any, Model())

    callback._run_eval()

    assert saved == []
    assert callback.best_checkpoint_status == "NO_ACTIVE_CHECKPOINT"
    assert callback.best_end_equity == float("-inf")


def test_survival_failed_checkpoint_is_not_saved(tmp_path: Path, monkeypatch: Any) -> None:
    class Env:
        account = SimpleNamespace(equity=100_000.0)
        trade_log = [{"type": "close"}]
        breach_events = [{"time": "2025-01-01"}]

        def reset(self) -> tuple[int, dict[str, Any]]:
            return 0, {}

        def step(self, _action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 1, 0.0, True, False, {}

    class Model:
        num_timesteps = 1

        def predict(self, _obs: int, *, deterministic: bool) -> tuple[int, None]:
            assert deterministic is True
            return 0, None

    saved: list[Path] = []
    monkeypatch.setattr(
        callbacks, "save_ppo_checkpoint", lambda _model, path: saved.append(Path(path))
    )
    callback = callbacks.BestCheckpointEvalCallback(
        eval_env_factory=Env,
        eval_freq=1,
        best_model_path=tmp_path / "ppo_best",
        n_eval_episodes=1,
    )
    callback.model = cast(Any, Model())

    callback._run_eval()

    assert saved == []
    assert callback.best_checkpoint_status == "NO_ACTIVE_CHECKPOINT"


def test_deterministic_inactivity_stops_only_after_two_five_million_checks(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class Env:
        account = SimpleNamespace(equity=100_000.0)
        trade_log: list[dict[str, Any]] = []
        breach_events: list[dict[str, Any]] = []

        def reset(self) -> tuple[int, dict[str, Any]]:
            return 0, {}

        def step(self, _action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 1, 0.0, True, False, {}

    class Model:
        num_timesteps = 1_000_000

        def predict(self, _obs: int, *, deterministic: bool) -> tuple[int, None]:
            assert deterministic is True
            return 0, None

    monkeypatch.setattr(callbacks, "save_ppo_checkpoint", lambda *_args, **_kwargs: None)
    callback = callbacks.BestCheckpointEvalCallback(
        eval_env_factory=Env,
        eval_freq=5_000_000,
        best_model_path=tmp_path / "ppo_best",
        n_eval_episodes=1,
    )
    model = cast(Any, Model())
    callback.model = model

    callback.num_timesteps = model.num_timesteps
    assert callback._on_step() is True
    model.num_timesteps = 5_000_000
    callback.num_timesteps = model.num_timesteps
    assert callback._on_step() is True
    assert callback.viability_failure_reason is None
    model.num_timesteps = 10_000_000
    callback.num_timesteps = model.num_timesteps
    assert callback._on_step() is False
    assert callback.viability_failure_reason is not None
    assert callback.viability_failure_reason.startswith("FAIL_POLICY_COLLAPSE")


def test_final_split_evaluation_forces_deterministic_actions(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate_model(model: object, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"model": model}

    monkeypatch.setattr(training_environments, "evaluate_model", fake_evaluate_model)
    model = object()

    result = training_environments._evaluate_final_split(model, deterministic=False, bars=None)

    assert result == {"model": model}
    assert captured["deterministic"] is True
