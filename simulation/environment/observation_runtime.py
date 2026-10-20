# ruff: noqa: F403, F405
from __future__ import annotations

from .runtime_dependencies import *


class ObservationRuntimeMixin(RuntimeMixinBase):
    def _session_seq_start(self) -> int:
        """First absolute bar index included in the session-scoped ``seq`` window.

        Only bars belonging to the current ``session_id`` are kept (same-day
        NY context). Left-pads when fewer than ``obs_window`` bars exist.
        """
        window_start = max(0, int(self.step_idx) - int(self.obs_window))
        if self._session_ids_arr is None or not (0 <= self.step_idx < len(self._session_ids_arr)):
            return window_start
        sid = int(self._session_ids_arr[self.step_idx])
        i = int(self.step_idx)
        while i > window_start and int(self._session_ids_arr[i - 1]) == sid:
            i -= 1
        return i

    def _get_observation(self) -> dict[str, np.ndarray[Any, Any]]:
        """Construct observation dict.

        ``seq`` is clipped to the **current** ``session_id`` (same-day NY
        bars only) and left-padded to ``obs_window``. Same-day pre-session
        features live in the current feature row inside that window.
        """
        start_idx = self._session_seq_start()
        seq = self._obs_features_arr[start_idx : self.step_idx]
        if self.mtf and self._m1_idx.size:
            seq = seq[:, self._m1_idx]
        elif self.mtf:
            seq = np.zeros((seq.shape[0], 1), dtype=np.float32)
        seq = cast(np.ndarray[Any, Any], np.nan_to_num(seq, nan=0.0))
        seq, seq_mask = pad_observation_window(seq, self.obs_window)

        # Account state
        pos_dir = float(self.position.direction) if self.position is not None else 0.0
        open_pnl = float(self.account.open_pnl) if self.position is not None else 0.0
        unrealised_r = (open_pnl / self.account.equity * 100) if self.account.equity > 0 else 0.0
        dist_to_sl = 0.0
        if self.position is not None and self.position.sl_price is not None:
            dist_to_sl = (
                self.position.entry_price - self.position.sl_price
                if self.position.direction == 1
                else self.position.sl_price - self.position.entry_price
            )

        if 0 <= self.step_idx < self._n_bars:
            current_close = float(self._close_arr[self.step_idx])
        else:
            current_close = 1.0
        trailing_dd = float(self.account.trailing_drawdown_pct())
        exposed_entry_state = (
            self.entry_state_machine.state
            if self.entry_state_observation and self.entry_state_machine is not None
            else None
        )
        if self.observation_schema.version in (OBSERVATION_SCHEMA_V2, OBSERVATION_SCHEMA_V3):
            if not self.normalize_account:
                raise ValueError("account-state-v2 requires normalize_account=True")
            position = self.position
            position_age_bars = 0.0
            if position is not None and position.entry_timestamp is not None:
                entry_time = pd.Timestamp(cast(Any, position.entry_timestamp))
                now = self._bar_time(self.step_idx)
                minutes = max(float(self._minutes_per_step), 1.0)
                position_age_bars = max((now - entry_time).total_seconds() / 60.0 / minutes, 0.0)
            account_state = account_state_v2(
                equity=float(self.account.equity),
                initial_balance=float(self.initial_balance),
                position_direction=pos_dir,
                open_pnl=open_pnl,
                planned_risk_usd=float(position.planned_risk_usd or 0.0)
                if position is not None
                else 0.0,
                trailing_drawdown_pct=trailing_dd,
                trailing_drawdown_limit=float(self.guardrails.trailing_dd_limit),
                daily_loss=float(self.account.canonical_daily_loss()),
                daily_loss_limit=float(self.guardrails.daily_loss_limit),
                loss_from_initial=float(self.account.loss_from_initial()),
                max_loss_limit=float(self.guardrails.max_loss_limit),
                risk_per_trade_limit=float(self.guardrails.risk_per_trade_limit),
                current_price=current_close,
                entry_price=float(position.entry_price) if position is not None else 0.0,
                initial_sl_price=(position.sl_initial_price or position.sl_price)
                if position is not None
                else None,
                active_sl_price=position.sl_price if position is not None else None,
                remaining_size=float(position.size) if position is not None else 0.0,
                initial_size=float(position.initial_size or position.size)
                if position is not None
                else 0.0,
                position_age_bars=position_age_bars,
                position_age_scale_bars=self.observation_age_scale_bars,
                tp_hit_mask=int(position.tp_hit_mask) if position is not None else 0,
                tp_prices=(
                    (position.tp1_price, position.tp2_price, position.tp3_price)
                    if position is not None
                    else (None, None, None)
                ),
                sl_mode=position.sl_mode if position is not None else "fixed",
                tp_mode=position.tp_mode if position is not None else "fixed",
                exit_mode=position.exit_mode if position is not None else "structural",
                entry_state=exposed_entry_state,
            )
        elif self.normalize_account:
            # Rescale so the account vector matches the ~O(1) magnitude of
            # the z-scored seq features. Without this, equity (1e5) and
            # raw open_pnl dwarf the time-series signal.
            account_state = normalized_account_vector(
                equity=float(self.account.equity),
                initial_balance=float(self.initial_balance),
                pos_dir=pos_dir,
                open_pnl=open_pnl,
                dist_to_sl=dist_to_sl,
                trailing_dd=trailing_dd,
                close=current_close,
                entry_state=exposed_entry_state,
            )
        else:
            raw_account = [
                self.account.equity,
                pos_dir,
                open_pnl,
                unrealised_r,
                dist_to_sl,
                float(self.account.trailing_drawdown_pct()),
            ]
            if exposed_entry_state is not None:
                raw_account.extend(entry_state_one_hot(exposed_entry_state))
            account_state = np.asarray(raw_account, dtype=np.float32)

        obs: dict[str, np.ndarray[Any, Any]] = {
            "seq": seq,
            "seq_mask": seq_mask,
            "account": account_state,
        }
        if self.mtf:
            decision = int(self.step_idx)
            for key, _, seq_key in HTF_BRANCHES:
                cols = self._htf_idx[key]
                block = (
                    self._obs_features_arr[:, cols]
                    if cols.size
                    else np.zeros((self._obs_features_arr.shape[0], 0), dtype=np.float32)
                )
                window, mask = closed_window(
                    np.asarray(block),
                    self._htf_starts[key],
                    decision,
                    int(self._mtf_windows.get(key, DEFAULT_HTF_WINDOWS[key])),
                )
                obs[seq_key] = window
                obs[f"mask_{key}"] = mask

        # Add VAE latent embedding if enabled
        if self.use_vae and self.vae is not None:
            import torch

            latent_dim = int(self.vae.encoder.latent_dim)
            seq_len = int(self.vae.encoder.seq_len)
            n_feat = int(self.vae.encoder.n_features)
            pre_ny_seq: np.ndarray[Any, Any] | None = None

            if self.pre_ny_by_date:
                if self._bundle_arrays is not None:
                    from datetime import date

                    day_key = date.fromordinal(
                        int(self._bundle_arrays["bar_day_ord"][self.step_idx])
                    )
                else:
                    assert self.bars is not None
                    day_key = pd.Timestamp(self.bars.index[self.step_idx]).date()
                pre_ny_seq = self.pre_ny_by_date.get(day_key)
            elif self.pre_ny_data is not None and len(self.pre_ny_data) > 0:
                # Legacy: each row is a flattened (seq_len * n_features) vector
                current_idx = min(self.step_idx, len(self.pre_ny_data) - 1)
                flat = np.asarray(self.pre_ny_data.iloc[current_idx].values, dtype=np.float32)
                flat = np.nan_to_num(flat, nan=0.0)
                if flat.size == seq_len * n_feat:
                    pre_ny_seq = flat.reshape(seq_len, n_feat)
                elif flat.ndim == 2:
                    pre_ny_seq = flat.astype(np.float32, copy=False)

            if pre_ny_seq is None:
                vae_z = np.zeros(latent_dim, dtype=np.float32)
            else:
                pre_ny_seq = np.nan_to_num(np.asarray(pre_ny_seq, dtype=np.float32), nan=0.0)
                if pre_ny_seq.shape != (seq_len, n_feat):
                    # Pad / truncate to the encoder contract
                    fixed = np.zeros((seq_len, n_feat), dtype=np.float32)
                    t = min(seq_len, pre_ny_seq.shape[0])
                    f = min(n_feat, pre_ny_seq.shape[1] if pre_ny_seq.ndim == 2 else n_feat)
                    if pre_ny_seq.ndim == 1:
                        pre_ny_seq = pre_ny_seq.reshape(-1, n_feat)[:t, :f]
                    fixed[-t:, :f] = pre_ny_seq[-t:, :f]
                    pre_ny_seq = fixed
                pre_ny_tensor = torch.from_numpy(pre_ny_seq).unsqueeze(0).float()
                vae_device = next(self.vae.parameters()).device
                pre_ny_tensor = pre_ny_tensor.to(vae_device)
                with torch.no_grad():
                    mu, _ = self.vae.encode(pre_ny_tensor)
                vae_z = mu.detach().cpu().numpy().astype(np.float32).reshape(-1)

            obs["vae_z"] = vae_z

        return obs
