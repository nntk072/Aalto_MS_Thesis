"""Focused simulation initialization service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from gymnasium import spaces

from .entry_diagnostics import _empty_entry_diag
from .mixin_base import RuntimeMixinBase
from .observations import DEFAULT_HTF_WINDOWS, HTF_BRANCHES
from .timebase import _minutes_per_step


class EpisodeSpacesInitializationMixin(RuntimeMixinBase):
    def _initialize_episode_spaces(
        self,
        ny_session_start_idx: int | None,
        bars: pd.DataFrame | None,
        bundle_data: Any | None,
        use_vae: bool,
        vae: Any,
        obs_window: int,
    ) -> None:
        # Per-step entry diagnostics (strategy_actions); reset each episode.
        self._entry_diag: dict[str, int] = _empty_entry_diag(self.entry_state_machine is not None)
        self._previous_constraint_ratios: dict[str, float] = {
            "daily": 0.0,
            "max_loss": 0.0,
            "trailing": 0.0,
        }

        # NY session start index for time decay penalty. Default to the first
        # tradable NY bar, which is what "minutes since open" measures from.
        self.ny_session_start_idx = (
            int(ny_session_start_idx)
            if ny_session_start_idx is not None
            else int(self._ny_indices[0])
            if len(self._ny_indices)
            else 0
        )
        # Bar spacing in minutes, read from the index rather than assumed. The
        # env steps the M1 spine, so the old hard-coded *5 made
        # "minutes_since_open" five times too large and let the sweep
        # time-decay term dominate the reward.
        time_index = (
            pd.DatetimeIndex(pd.to_datetime(self._bundle_arrays["bar_time_ns"], utc=True))
            if self._bundle_arrays is not None
            else pd.DatetimeIndex(bars.index)
            if bars is not None
            else pd.DatetimeIndex([])
        )
        self._minutes_per_step = _minutes_per_step(time_index)

        # Observation space: market sequence, account state, and optional entry state.
        # features: (obs_window, n_features)
        # account: [equity, position_direction, open_pnl, unrealised_r, dist_to_sl, trailing_dd]
        # vae_z: latent embedding from VAE (if use_vae=True)
        n_features = len(self._obs_columns) if self._obs_columns else 1
        if self.mtf:
            n_features = max(int(self._m1_idx.size), 1)
        vae_latent_dim = vae.encoder.latent_dim if use_vae and vae is not None else 0

        account_dim = self.observation_schema.width
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_window, n_features),
                    dtype=np.float32,
                ),
                "account": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(account_dim,),
                    dtype=np.float32,
                ),
                "seq_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(obs_window,),
                    dtype=np.float32,
                ),
            }
        )
        if self.mtf:
            for key, _, seq_key in HTF_BRANCHES:
                width = max(int(self._htf_idx[key].size), 1)
                length = int(self._mtf_windows.get(key, DEFAULT_HTF_WINDOWS[key]))
                observation_space.spaces[seq_key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(length, width),
                    dtype=np.float32,
                )
                observation_space.spaces[f"mask_{key}"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(length,),
                    dtype=np.float32,
                )
        # Add VAE latent to observation space if enabled
        if use_vae and vae_latent_dim > 0:
            observation_space.spaces["vae_z"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(vae_latent_dim,),
                dtype=np.float32,
            )
        setattr(self, "observation_space", observation_space)


__all__ = ["EpisodeSpacesInitializationMixin"]
