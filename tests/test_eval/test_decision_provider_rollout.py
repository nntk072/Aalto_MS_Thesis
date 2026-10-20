from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from decisioning.adapters.rule_based import RuleBasedDecisionProvider
from decisioning.adapters.sb3 import SB3DecisionProvider
from decisioning.contracts import (
    ActionSchemaRef,
    ContinuousAction,
    Decision,
    DecisionInput,
    DiscreteAction,
    PredictorMetadata,
    StructuredAction,
)
from evaluation.rollout.model import make_action_fn
from evaluation.rollout.provider import make_provider_action_fn


class _Provider:
    def __init__(self, action: Decision) -> None:
        self.action = action
        self.inputs: list[tuple[DecisionInput, bool]] = []
        self.metadata = PredictorMetadata(
            provider_id="test",
            model_family="test-provider",
            training_paradigm="test",
            input_schema_version="obs-v1",
            action_schema=ActionSchemaRef(
                version="test-v1", key="test", coordinates=("action",), kind="test"
            ),
        )

    def predict(self, input: DecisionInput, *, deterministic: bool = True) -> Decision:
        self.inputs.append((input, deterministic))
        return self.action


@pytest.mark.parametrize(
    ("action", "continuous", "strategy", "expected"),
    [
        (
            Decision(
                DiscreteAction(4, "enter_short"),
                ActionSchemaRef("d-v1", "d5", ("hold", "a", "b", "c", "enter_short"), "discrete"),
            ),
            False,
            False,
            4,
        ),
        (
            Decision(
                ContinuousAction(-0.25),
                ActionSchemaRef("c-v1", "scalar", ("position",), "continuous"),
            ),
            True,
            False,
            -0.25,
        ),
        (
            Decision(
                StructuredAction({"direction": 0.75, "risk": -0.5}),
                ActionSchemaRef("a-v1", "pair", ("direction", "risk"), "continuous"),
            ),
            False,
            True,
            np.asarray([0.75, -0.5], dtype=np.float32),
        ),
    ],
)
def test_provider_rollout_adapter_returns_environment_action(
    action: Decision,
    continuous: bool,
    strategy: bool,
    expected: int | float | np.ndarray[Any, Any],
) -> None:
    provider = _Provider(action)
    observation = {"seq": np.ones((2, 3), dtype=np.float32)}
    action_fn = make_provider_action_fn(
        provider,
        input_schema_version="obs-v1",
        deterministic=False,
        continuous_actions=continuous,
        strategy_actions=strategy,
    )

    result = action_fn(observation)

    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    else:
        assert (
            result == pytest.approx(expected) if isinstance(expected, float) else result == expected
        )
    received_input, deterministic = provider.inputs[0]
    assert received_input.schema_version == "obs-v1"
    np.testing.assert_array_equal(received_input.observation["seq"], observation["seq"])
    assert deterministic is False


def test_rule_and_sb3_providers_share_the_same_rollout_boundary() -> None:
    action_schema = ActionSchemaRef(
        version="discrete-v1", key="hold-enter", coordinates=("hold", "enter"), kind="discrete"
    )
    metadata = PredictorMetadata(
        provider_id="rule",
        model_family="rule",
        training_paradigm="rule_based",
        input_schema_version="obs-v1",
        action_schema=action_schema,
    )
    rule_provider = RuleBasedDecisionProvider(
        lambda _: Decision(DiscreteAction(1, "enter"), action_schema), metadata
    )

    class _SB3Policy:
        action_space: spaces.Space[Any] = spaces.Discrete(2)
        _action_schema = {
            "kind": "discrete",
            "version": "discrete-v1",
            "key": "hold-enter",
            "coordinates": ("hold", "enter"),
        }

        def predict(self, _observation, *, deterministic):
            del deterministic
            return 0, None

    sb3_provider = SB3DecisionProvider(_SB3Policy(), input_schema_version="obs-v1")
    observation = {"seq": np.ones((2, 3), dtype=np.float32)}

    actions = tuple(
        make_provider_action_fn(provider, input_schema_version="obs-v1")(observation)
        for provider in (rule_provider, sb3_provider)
    )

    assert actions == (1, 0)


def test_evaluation_rollout_routes_providers_through_semantic_adapter() -> None:
    action_schema = ActionSchemaRef(
        version="discrete-v1", key="hold-enter", coordinates=("hold", "enter"), kind="discrete"
    )
    provider = _Provider(Decision(DiscreteAction(1, "enter"), action_schema))
    action_fn = make_action_fn(provider)

    assert action_fn({"seq": np.ones((2, 3), dtype=np.float32)}) == 1
    assert provider.inputs[0][0].schema_version == "obs-v1"
