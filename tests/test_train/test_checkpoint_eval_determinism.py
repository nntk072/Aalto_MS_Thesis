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
