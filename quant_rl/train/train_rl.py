"""RL training entrypoint (wires SB3 PPO to TradingEnv).

Usage
-----
    cd Aalto_MS_Thesis
    python -m quant_rl.train.train_rl                         # TCN encoder
    python -m quant_rl.train.train_rl arch=transformer        # Transformer
    python -m quant_rl.train.train_rl --stub                  # trivial MLP
    python -m quant_rl.train.train_rl training.mode=mvp \\
        training.use_m1_only=true training.primary_only=true  # MVP mode

Output layout
-------------
    outputs/<ts>_train_<arch>/
        config.yaml
        summary.txt                   # Train vs Test comparison
        training/  (in-sample charts + data)
        testing/   (out-of-sample charts + data)
        model/
            ppo_<arch><mode>.zip
            training_log.csv
            learning_curve.png/html
            losses.png/html
            entropy_explvar.png/html
            kl_clip.png/html
            lr.png/html
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging

import numpy as np

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import split_train_test, get_split_config
from quant_rl.features.build import build_features
from quant_rl.backtest.engine import run_backtest
from quant_rl.backtest.costs import COST_US100
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.eval.metrics import calculate_metrics
from quant_rl.eval.export import save_run, build_run_dir
from quant_rl.eval.training_plots import save_training_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_stub_agent(env, cfg):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    return PPO(
        "MultiInputPolicy", vec_env,
        n_steps=min(cfg.ppo.n_steps, 512),
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        verbose=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    parser.add_argument("--stub",   action="store_true", help="Use stub MLP (no encoder)")
    parser.add_argument("--arch",   default="tcn", choices=["tcn", "transformer"])
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--out",    default="outputs")
    args = parser.parse_args()

    cfg  = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym = cfg.data.primary
    primary_m1  = data[primary_sym]["M1"]

    # MVP mode flags
    training_mode = getattr(getattr(cfg, "training", None), "mode",         "full")
    use_m1_only   = getattr(getattr(cfg, "training", None), "use_m1_only",  False)
    max_days      = getattr(getattr(cfg, "training", None), "max_days",     None)
    is_mvp        = training_mode == "mvp" or use_m1_only

    if is_mvp:
        log.info("MVP mode: M1-only, primary symbol only")
        secondary_m1 = None
    else:
        secondary_sym = cfg.data.secondary
        secondary_m1  = data.get(secondary_sym, {}).get("M1")

    # Build features on full data before splitting
    cache_dir   = Path(cfg.data.cache_dir)
    feat_suffix = "_mvp" if is_mvp else ""
    feat_cache  = cache_dir / f"{primary_sym}{feat_suffix}_features.parquet"
    features    = build_features(primary_m1, secondary=secondary_m1, cfg=cfg,
                                  cache_path=feat_cache, force=args.force)

    # Date-based split
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    log.info("Split: train=%d bars (≤%s)  test=%d bars (≥%s)",
             len(train_bars), train_end, len(test_bars), test_start)

    # Optional: trim TRAIN portion to last N calendar days for fast iteration
    if max_days is not None:
        cutoff = train_bars.index.normalize().unique()[-int(max_days):][0]
        train_bars = train_bars[train_bars.index >= cutoff]
        train_feat = train_feat[train_feat.index >= cutoff]
        log.info("Trimmed train to last %d days: %d bars", max_days, len(train_bars))

    # Create run directory early so the model can be saved inside it
    arch_tag = args.arch if not args.stub else "stub"
    mode_tag = "_mvp" if is_mvp else ""
    run_dir  = build_run_dir(args.out, f"train_{arch_tag}{mode_tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build environment on TRAIN data only
    env = TradingEnv(
        bars=train_bars,
        features=train_feat,
        obs_window=cfg.env.obs_window,
        cost_model=COST_US100,
        initial_balance=cfg.account.initial_balance,
        dsr_eta=cfg.env.reward_dsr_eta,
        max_episode_steps=cfg.env.max_episode_steps,
    )

    if args.stub:
        log.info("Building stub MLP agent …")
        model     = _build_stub_agent(env, cfg)
        timesteps = 2048
    else:
        log.info("Building PPO + %s encoder …", arch_tag.upper())
        from quant_rl.models.agent import build_agent
        model = build_agent(env, cfg, arch=args.arch)
        if is_mvp:
            timesteps = getattr(getattr(cfg, "training", None), "total_timesteps_mvp", 8192)
            log.info("MVP timesteps: %d", timesteps)
        else:
            timesteps = cfg.ppo.total_timesteps

    # Progress logger callback
    log_csv = model_dir / "training_log.csv"
    try:
        from quant_rl.train.callbacks import ProgressLoggerCallback
        callbacks = [ProgressLoggerCallback(log_path=log_csv)]
    except Exception:
        callbacks = []
        log.warning("ProgressLoggerCallback unavailable; training progress will not be logged")

    log.info("Starting training for %d timesteps …", timesteps)
    model.learn(total_timesteps=timesteps, callback=callbacks or None)
    log.info("Training complete.")

    # Save model inside run directory
    model_path = model_dir / f"ppo_{arch_tag}{mode_tag}"
    model.save(str(model_path))
    log.info("Model saved to %s", model_path)

    # Generate training-progress plots from the logged CSV
    save_training_plots(log_csv, out_dir=model_dir,
                        save_html=True, dpi=150)

    if args.no_save:
        return

    # Backtest trained policy on BOTH splits
    obs_window = cfg.env.obs_window

    def _make_policy():
        from quant_rl.backtest.account import AccountState
        _acc = AccountState(initial_balance=cfg.account.initial_balance)

        def policy(obs_seq: np.ndarray) -> int:
            obs_dict = {
                "seq":     obs_seq[np.newaxis].astype(np.float32),
                "account": _acc.to_array()[np.newaxis].astype(np.float32),
            }
            action, _ = model.predict(obs_dict, deterministic=True)
            return {0: 0, 1: 1, 2: -1}.get(int(action), 0)

        return policy

    def _run(bars, feats, label: str) -> tuple[dict, object]:
        log.info("Backtesting [%s] …", label)
        result = run_backtest(
            bars=bars, features=feats,
            policy=_make_policy(),          # fresh account state per split
            obs_window=obs_window,
            initial_balance=cfg.account.initial_balance,
        )
        result["initial_balance"] = cfg.account.initial_balance
        m = calculate_metrics(
            result["equity"], trades=result["trades"],
            n_sessions=result.get("n_sessions", 1),
            n_breach_sessions=result.get("n_breach_sessions", 0),
        )
        log.info(
            "[%s] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%",
            label, m.sharpe, m.max_drawdown * 100, m.total_trades, m.total_return * 100,
        )
        return result, m

    train_result, train_m = _run(train_bars, train_feat, "training")
    test_result,  test_m  = _run(test_bars,  test_feat,  "testing")

    save_run(
        run_dir=run_dir,
        train_result=train_result, train_metrics=train_m, train_bars=train_bars,
        test_result=test_result,   test_metrics=test_m,   test_bars=test_bars,
        cfg=cfg,
    )
    log.info("Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()


def _build_stub_agent(env, cfg):
    """Trivial MlpPolicy PPO – no custom encoder, used only to verify env wiring."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])
    return PPO(
        "MultiInputPolicy",
        vec_env,
        n_steps=min(cfg.ppo.n_steps, 512),
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        verbose=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    parser.add_argument("--stub", action="store_true", help="Use stub MLP (no encoder needed)")
    parser.add_argument("--arch", default="tcn", choices=["tcn", "transformer"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym = cfg.data.primary
    primary_m1 = data[primary_sym]["M1"]

    # --- MVP mode: M1-only, primary symbol only, optional date slice ---
    training_mode = getattr(getattr(cfg, "training", None), "mode", "full")
    use_m1_only   = getattr(getattr(cfg, "training", None), "use_m1_only", False)
    primary_only  = getattr(getattr(cfg, "training", None), "primary_only", False)
    max_days      = getattr(getattr(cfg, "training", None), "max_days", None)

    is_mvp = training_mode == "mvp" or use_m1_only

    if is_mvp:
        log.info("MVP mode: M1-only, primary symbol only, secondary features disabled")
        secondary_m1 = None    # skip SMT / secondary features
    else:
        secondary_sym = cfg.data.secondary
        secondary_m1 = data.get(secondary_sym, {}).get("M1")

    # Optional: trim to last N calendar days for fast iteration
    if max_days is not None:
        cutoff = primary_m1.index.normalize().unique()[-int(max_days):][0]
        primary_m1 = primary_m1[primary_m1.index >= cutoff]
        log.info("Trimmed training data to last %d days: %d bars", max_days, len(primary_m1))

    cache_dir = Path(cfg.data.cache_dir)
    feat_suffix = "_mvp" if is_mvp else ""
    feat_cache = cache_dir / f"{primary_sym}{feat_suffix}_features.parquet"
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg,
                               cache_path=feat_cache, force=args.force)

    env = TradingEnv(
        bars=primary_m1,
        features=features,
        obs_window=cfg.env.obs_window,
        cost_model=COST_US100,
        initial_balance=cfg.account.initial_balance,
        dsr_eta=cfg.env.reward_dsr_eta,
        max_episode_steps=cfg.env.max_episode_steps,
    )

    if args.stub:
        log.info("Building stub MLP agent (env wiring test) …")
        model = _build_stub_agent(env, cfg)
        timesteps = 2048
    else:
        log.info("Building PPO + %s encoder …", args.arch.upper())
        from quant_rl.models.agent import build_agent
        model = build_agent(env, cfg, arch=args.arch)
        if is_mvp:
            timesteps = getattr(getattr(cfg, "training", None), "total_timesteps_mvp", 8192)
            log.info("MVP timesteps: %d", timesteps)
        else:
            timesteps = cfg.ppo.total_timesteps

    log.info("Starting training for %d timesteps …", timesteps)
    model.learn(total_timesteps=timesteps)
    log.info("Training complete.")

    arch_tag = args.arch if not args.stub else "stub"
    mode_tag = "_mvp" if is_mvp else ""
    save_path = Path("models") / f"ppo_{arch_tag}{mode_tag}"
    save_path.parent.mkdir(exist_ok=True)
    model.save(str(save_path))
    log.info("Model saved to %s", save_path)


if __name__ == "__main__":
    main()
