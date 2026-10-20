"""Composition root for environment initialization services."""

from __future__ import annotations

from .account_policy_initialization import AccountPolicyInitializationMixin
from .action_episode_initialization import ActionEpisodeInitializationMixin
from .episode_spaces_initialization import EpisodeSpacesInitializationMixin
from .execution_controls_initialization import ExecutionControlsInitializationMixin
from .market_source_initialization import MarketSourceInitializationMixin
from .observation_arrays_initialization import ObservationArraysInitializationMixin
from .reward_initialization import RewardInitializationMixin
from .risk_entry_initialization import RiskEntryInitializationMixin
from .strategy_initialization import StrategyInitializationMixin


class InitializationRuntimeMixin(
    MarketSourceInitializationMixin,
    AccountPolicyInitializationMixin,
    RiskEntryInitializationMixin,
    ExecutionControlsInitializationMixin,
    RewardInitializationMixin,
    StrategyInitializationMixin,
    ObservationArraysInitializationMixin,
    ActionEpisodeInitializationMixin,
    EpisodeSpacesInitializationMixin,
):
    """Compose input, account, policy, reward, strategy, and space setup."""

    pass


__all__ = ["InitializationRuntimeMixin"]
