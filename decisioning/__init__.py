"""Backend-neutral interfaces for producing trading decisions."""

from decisioning.actions import (
    ActionSchema,
    DiscreteActionSchema,
    build_action_schema,
    build_discrete_trading_action_schema,
)
from decisioning.contracts import (
    ActionSchemaRef,
    ContinuousAction,
    Decision,
    DecisionArtifactRef,
    DecisionInput,
    DecisionProvider,
    DiscreteAction,
    PredictorMetadata,
    StructuredAction,
    to_policy_action,
)
from decisioning.observation_state import (
    BASE_ACCOUNT_DIM,
    ENTRY_STATE_DIM,
    ENTRY_STATES,
    entry_state_one_hot,
)
from decisioning.observations import (
    ObservationSchema,
    build_observation_schema,
    normalized_account_vector,
    pad_observation_window,
)

__all__ = [
    "ActionSchema",
    "ActionSchemaRef",
    "BASE_ACCOUNT_DIM",
    "ContinuousAction",
    "Decision",
    "DecisionArtifactRef",
    "DecisionInput",
    "DecisionProvider",
    "DiscreteActionSchema",
    "DiscreteAction",
    "ENTRY_STATE_DIM",
    "ENTRY_STATES",
    "ObservationSchema",
    "PredictorMetadata",
    "StructuredAction",
    "build_action_schema",
    "build_discrete_trading_action_schema",
    "build_observation_schema",
    "entry_state_one_hot",
    "normalized_account_vector",
    "pad_observation_window",
    "to_policy_action",
]
