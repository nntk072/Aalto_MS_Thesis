# mypy: ignore-errors
"""Compare MVP vs full seed50 policy open rates and action intensity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from quant_rl.config import load_config
from quant_rl.data.pipeline import prepare_data
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.features.build import build_features, feature_cache_path
from quant_rl.train.train_rl import (
    _guardrail_kwargs,
    _max_loss_per_trade,
    _strategy_from_cfg,
    _strategy_risk_ranges,
)

MVP = Path("outputs/20260920_205105_rl_train_seed50")
FULL = Path("outputs/20260920_210733_rl_train_seed50")


def make_env(cfg, bars, feat):
    strategy, strategy_reward, strategy_weight = _strategy_from_cfg(cfg)
    risk_frac_range, rr_ratio_range = _strategy_risk_ranges(cfg)
    return TradingEnv(
        bars=bars,
        features=feat,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        guardrail_kwargs=_guardrail_kwargs(cfg),
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=_max_loss_per_trade(cfg),
        dsr_eta=cfg.env.reward_dsr_eta,
        episodic=False,
        continuous_actions=False,
        strategy=strategy,
        strategy_actions=bool(cfg.env.get("strategy_actions", False)),
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        sl_buffer_pts=float(cfg.env.get("sl_buffer_pts", 0.0)),
        block_overnight=bool(cfg.env.get("block_overnight", True)),
        eod_risk=dict(cfg.env.get("eod_risk", {})),
        min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
        max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
        entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
        reward_mode=str(cfg.env.get("reward_mode", "dsr")),
        max_episode_steps=None,
    )


def probe(run: Path, bars, feat, n_steps=2000):
    cfg = OmegaConf.load(run / "config.yaml")
    model = PPO.load(str(run / "model" / "ppo_final"), device="cuda")
    env = make_env(cfg, bars, feat)
    obs, _ = env.reset()
    intensities = []
    actions_u0 = []
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        u = 0.5 * (np.clip(a, -1, 1) + 1)
        intensities.append(float(u[0]) if u.size else 0.0)
        actions_u0.append(float(a[0]) if a.size else 0.0)
        obs, r, done, trunc, info = env.step(action)
        if done or trunc:
            break
    diag = dict(env._entry_diag)
    opens = sum(1 for t in env.trade_log if t.get("type") == "open")
    print(f"=== {run.name} ===")
    print("action_space", model.action_space)
    print(
        "raw_a0 mean/std/min/max",
        float(np.mean(actions_u0)),
        float(np.std(actions_u0)),
        float(np.min(actions_u0)),
        float(np.max(actions_u0)),
    )
    print(
        "intensity mean/std/min/max",
        float(np.mean(intensities)),
        float(np.std(intensities)),
        float(np.min(intensities)),
        float(np.max(intensities)),
    )
    print("frac intensity>=0.25", float(np.mean(np.asarray(intensities) >= 0.25)))
    print("entry_diag", diag)
    print("opens_in_probe", opens, "steps", i + 1)
    print()


def main():
    cfg = load_config(
        config_path="config/features_full_po3_mtf.yaml",
        overrides=[
            "features.include_session_ohlc=true",
            "features.liquidity.enabled=true",
            "features.po3_state_mtf.enabled=true",
            "features.ifvg_mtf.enabled=true",
        ],
    )
    # merge idea1 like train
    from omegaconf import OmegaConf

    idea = OmegaConf.load("config/idea1_po3_ifvg.yaml")
    cfg = OmegaConf.merge(cfg, idea)
    frames = prepare_data(cfg, force=False)
    primary_sym = cfg.data.primary
    primary_m1 = frames[primary_sym]
    # secondary
    sec = cfg.data.get("secondary")
    secondary_m1 = frames.get(sec) if sec else None
    train_end = pd.Timestamp(cfg.data.split.train_end)
    if primary_m1.index.tz is not None and train_end.tzinfo is None:
        train_end = train_end.tz_localize(primary_m1.index.tz)
    train_mask = primary_m1.index <= train_end
    cache_dir = Path(cfg.data.cache_dir)
    feat_cache = feature_cache_path(cache_dir, primary_sym, cfg, primary_m1, train_mask=train_mask)
    features = build_features(
        primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache, train_mask=train_mask
    )
    # short NY slice from train
    bars = primary_m1.iloc[:5000]
    feat = features.reindex(bars.index).fillna(0)
    probe(MVP, bars, feat)
    probe(FULL, bars, feat)


if __name__ == "__main__":
    main()
