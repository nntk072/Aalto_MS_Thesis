from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

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


def test_full_training_viability_uses_its_own_factory_and_crossed_thresholds(
    tmp_path: Path,
) -> None:
    class FullTrainingEnv:
        account = SimpleNamespace(equity=100_000.0)
        trade_log: list[dict[str, Any]] = []
        breach_events: list[dict[str, Any]] = []

        def reset(self) -> tuple[int, dict[str, Any]]:
            return 0, {}

        def step(self, _action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 1, 0.0, True, False, {}

        def entry_state_diagnostics(self) -> dict[str, Any]:
            return {
                "entry_state_machine": True,
                "candidate_created": 1,
                "armed_bars": 1,
            }

    class Model:
        def predict(self, _obs: int, *, deterministic: bool) -> tuple[int, None]:
            assert deterministic is True
            return 0, None

    calls = 0

    def full_training_factory() -> FullTrainingEnv:
        nonlocal calls
        calls += 1
        return FullTrainingEnv()

    callback = callbacks.BestCheckpointEvalCallback(
        eval_env_factory=FullTrainingEnv,
        eval_freq=0,
        best_model_path=tmp_path / "ppo_best",
        viability_eval_env_factory=full_training_factory,
        viability_eval_freq=5_000_000,
    )
    callback.model = cast(Any, Model())

    callback.num_timesteps = 5_000_007
    assert callback._on_step() is True
    assert calls == 1
    assert callback.consecutive_inactive_viability_evals == 1
    assert callback.viability_eval_history[0]["timestep"] == 5_000_007

    callback.num_timesteps = 10_000_009
    assert callback._on_step() is False
    assert calls == 2
    assert callback.viability_failure_reason is not None
    assert callback.viability_failure_reason.startswith("FAIL_POLICY_COLLAPSE")


def test_full_training_viability_records_action_coordinates_and_rejections(
    tmp_path: Path,
) -> None:
    class Schema:
        def to_dict(self) -> dict[str, Any]:
            return {"coordinates": ["direction", "risk"]}

    class FullTrainingEnv:
        action_schema = Schema()
        account = SimpleNamespace(equity=100_000.0)
        trade_log: list[dict[str, Any]] = []
        breach_events: list[dict[str, Any]] = []
        _entry_diag = {"rejected_gate": 3, "rejected_rr": 1}

        def __init__(self) -> None:
            self.steps = 0
            self.closed = False

        def reset(self) -> tuple[int, dict[str, Any]]:
            self.steps = 0
            return 0, {}

        def step(self, _action: Any) -> tuple[int, float, bool, bool, dict[str, Any]]:
            self.steps += 1
            return self.steps, 0.0, self.steps == 2, False, {}

        def close(self) -> None:
            self.closed = True

        def entry_state_diagnostics(self) -> dict[str, Any]:
            return {
                "entry_state_machine": True,
                "candidate_created": 1,
                "armed_bars": 1,
            }

    class Model:
        def predict(self, obs: int, *, deterministic: bool) -> tuple[np.ndarray[Any, Any], None]:
            assert deterministic is True
            return np.asarray([float(obs), float(1 - obs)], dtype=np.float32), None

    envs: list[FullTrainingEnv] = []

    def factory() -> FullTrainingEnv:
        env = FullTrainingEnv()
        envs.append(env)
        return env

    callback = callbacks.BestCheckpointEvalCallback(
        eval_env_factory=factory,
        eval_freq=0,
        best_model_path=tmp_path / "ppo_best",
        viability_eval_env_factory=factory,
        viability_eval_freq=5_000_000,
    )
    callback.model = cast(Any, Model())
    callback.num_timesteps = 5_000_000

    assert callback._on_step() is True
    row = callback.viability_eval_history[0]
    assert row["action_coordinate_schema"] == ["direction", "risk"]
    assert row["action_coordinate_schema_mismatch"] is False
    assert row["action_coordinate_diagnostics"]["direction"]["n"] == 2
    assert row["action_coordinate_diagnostics"]["direction"]["mean"] == 0.5
    assert row["action_coordinate_diagnostics"]["direction"]["std"] == 0.5
    assert row["action_coordinate_diagnostics"]["risk"]["mean"] == 0.5
    assert row["entry_funnel_diagnostics"] == {"rejected_gate": 3, "rejected_rr": 1}
    assert envs[0].closed is True


def test_full_training_viability_stops_on_missing_entry_candidates(
    tmp_path: Path,
) -> None:
    class FullTrainingEnv:
        account = SimpleNamespace(equity=100_000.0)
        trade_log: list[dict[str, Any]] = []
        breach_events: list[dict[str, Any]] = []

        def reset(self) -> tuple[int, dict[str, Any]]:
            return 0, {}

        def step(self, _action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 1, 0.0, True, False, {}

        def entry_state_diagnostics(self) -> dict[str, Any]:
            return {"entry_state_machine": True, "candidate_created": 0}

    class Model:
        def predict(self, _obs: int, *, deterministic: bool) -> tuple[int, None]:
            assert deterministic is True
            return 0, None

    callback = callbacks.BestCheckpointEvalCallback(
        eval_env_factory=FullTrainingEnv,
        eval_freq=0,
        best_model_path=tmp_path / "ppo_best",
        viability_eval_env_factory=FullTrainingEnv,
        viability_eval_freq=5_000_000,
    )
    callback.model = cast(Any, Model())
    callback.num_timesteps = 5_000_000

    assert callback._on_step() is False
    assert callback.viability_failure_reason is not None
    assert callback.viability_failure_reason.startswith("FAIL_ENTRY_PATH")


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
