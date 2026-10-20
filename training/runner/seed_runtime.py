"""Per-seed training lifecycle for the RL training composition root."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from configuration.paths import repository_root

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
import time
from functools import partial
from typing import Any, cast

import pandas as pd
from gymnasium import spaces
from omegaconf import OmegaConf

from evaluation import calculate_metrics, evaluation_periods_per_year
from evaluation.reporting.export import save_run
from evaluation.reporting.run_manifest import (
    build_stop_candidate_schema,
    build_training_run_manifest,
    file_sha256,
    write_run_manifest,
)
from evaluation.rollout.model import BAR_CLOSE_EXPOSURE_MEASURE, evaluate_model
from features.context.policy_market import (
    POLICY_MARKET_CONTEXT_VERSION,
)
from market_data.loading.ticks import ticks_covering
from models.factory.agent import build_agent
from models.policies.ppo import install_post_update_std_clip
from trading.policy.config import resolve_trading_policy
from training.artifacts.aggregation import _save_seed_artifacts
from training.artifacts.manifests import (
    _action_schema_manifest,
    _action_space_width,
    _observation_account_dim,
    _observation_schema_manifest,
    _observation_space_shapes,
    _sequence_feature_schema_manifest,
    _training_protocol_manifest,
)
from training.artifacts.metrics import _write_evaluation_artifacts
from training.callbacks.auxiliary import AuxiliaryTrainerCallback
from training.callbacks.core import (
    BestCheckpointEvalCallback,
    ClipLogStdCallback,
    EarlyAbortCallback,
    EpisodeEquityCallback,
    LagrangianRiskCallback,
    LearningDiagnosticsCallback,
    PeriodicCheckpointCallback,
    ProgressLoggerCallback,
    save_ppo_checkpoint,
)
from training.callbacks.dashboard import TrainingDashboardCallback, ppo_log_interval
from training.callbacks.debug_trace import install_flight_recorder
from training.callbacks.equity import assess_train_equity
from training.runner.seed import _direction_summary, _setup_rngs
from training.setup.callbacks import (
    _dashboard_block,
    _early_abort_callback,
    _periodic_checkpoint_callback,
)
from training.setup.environments import (
    _build_eval_common,
    _evaluate_final_split,
    _fill_delay_ms,
    _strategy_from_cfg,
    _training_episodic,
    make_env,
)
from training.setup.models import (
    _initialize_value_head,
    _log_effective_ppo,
    _training_architecture_provenance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass(slots=True)
class SeedRunContext:
    args: Any
    cfg: Any
    device: Any
    seeds: list[int]
    is_multi_seed: bool
    run_dir: Path
    model_dir: Path
    train_env_spec: Any
    train_bars: Any
    train_feat: Any
    env_vae: Any
    pre_ny_by_date: Any
    obs_mmap: str | None
    primary_ticks: Any
    lagrangian_cb: Any
    train_bar_count: int
    best_bars: pd.DataFrame | None
    best_features: pd.DataFrame | None
    bundle_path: Path | None
    cache_dir: Path
    feat_cache: Path
    dashboard_bars: pd.DataFrame | None
    dashboard_features: pd.DataFrame | None
    test_eval_spec: Any
    test_bars: Any
    test_feat: Any
    train_eval_spec: Any
    train_sec: pd.DataFrame | None
    test_sec: pd.DataFrame | None
    test_bundle_path: Path | None
    dataset_records: dict[str, Any]
    train_start: Any
    train_end: Any
    test_start: Any
    data_source: str
    primary_sym: str
    schedule_path: Path
    vae_model: Any


class SeedRunner:
    """Execute one seed through preparation, learning, evaluation, and export."""

    def __init__(self, context: SeedRunContext, seed: int) -> None:
        self.context = context
        self.seed = int(seed)

    def prepare_training(self) -> None:
        ctx = self.context
        args = ctx.args
        cfg = ctx.cfg
        seed = self.seed
        seeds = ctx.seeds
        is_multi_seed = ctx.is_multi_seed
        run_dir = ctx.run_dir
        model_dir = ctx.model_dir
        train_env_spec = ctx.train_env_spec
        train_bars = ctx.train_bars
        train_feat = ctx.train_feat
        env_vae = ctx.env_vae
        pre_ny_by_date = ctx.pre_ny_by_date
        obs_mmap = ctx.obs_mmap
        primary_ticks = ctx.primary_ticks
        device = ctx.device
        vae_model = ctx.vae_model
        lagrangian_cb = ctx.lagrangian_cb
        _setup_rngs(seed)
        phase_timings_seconds: dict[str, float] = {}
        log.info("=== Seed %d / %d ===", seed, len(seeds))

        # Per-seed output dirs
        if is_multi_seed:
            seed_dir = run_dir / "seeds" / f"seed{seed}"
            seed_model_dir = seed_dir / "model"
            seed_model_dir.mkdir(parents=True, exist_ok=True)
        else:
            seed_dir = run_dir
            seed_model_dir = model_dir

        # Create training environment
        log.info("Creating training environment...")
        if train_env_spec is not None:
            from simulation.environment.worker import make_bundle_env

            train_env = make_bundle_env(train_env_spec)
            env_factory = partial(make_bundle_env, train_env_spec)
        else:
            train_env = make_env(
                train_bars,
                train_feat,
                cfg,
                algo=args.algo,
                reward=args.reward,
                episodic=_training_episodic(cfg),
                use_vae=args.use_vae,
                vae=env_vae,
                pre_ny_by_date=pre_ny_by_date,
                obs_features_mmap=obs_mmap,
                tickbook=ticks_covering(primary_ticks, train_bars),
                arch=args.arch,
            )
            env_factory = partial(
                make_env,
                train_bars,
                train_feat,
                cfg,
                algo=args.algo,
                reward=args.reward,
                use_vae=args.use_vae,
                vae=env_vae,
                pre_ny_by_date=pre_ny_by_date,
                obs_features_mmap=obs_mmap,
                tickbook=ticks_covering(primary_ticks, train_bars),
                arch=args.arch,
            )

        checkpoint_callback = _periodic_checkpoint_callback(cfg, seed_model_dir)

        # Train agent
        timesteps = cfg.ppo.total_timesteps if not args.mvp else cfg.training.total_timesteps_mvp
        log.info("Training %s for %d timesteps...", args.algo.upper(), timesteps)

        if args.resume_checkpoint:
            from stable_baselines3 import PPO

            from models.factory.agent import _build_vector_env, validate_model_architecture

            resume_path = Path(args.resume_checkpoint)
            if not resume_path.is_file():
                raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
            resume_env = _build_vector_env(train_env, cfg, env_factory)
            model = PPO.load(resume_path, env=resume_env, device=device)
            validate_model_architecture(model, args.arch, use_vae=args.use_vae)
            from models.factory.agent import validate_model_observation_schema

            validate_model_observation_schema(
                model,
                train_env.observation_schema,
                allow_legacy_v1=bool(args.allow_legacy_observation_schema),
            )
            if int(model.num_timesteps) > int(timesteps):
                raise ValueError(
                    f"resume checkpoint has {model.num_timesteps} steps, exceeding requested {timesteps}"
                )
            learning_timesteps = int(timesteps) - int(model.num_timesteps)
            log.info(
                "Resuming PPO from %s at %d/%d steps; %d remain",
                resume_path,
                model.num_timesteps,
                timesteps,
                learning_timesteps,
            )
        else:
            model = build_agent(
                train_env,
                cfg,
                arch=args.arch,
                algo=args.algo,
                device=device,
                use_vae=args.use_vae,
                vae=vae_model,
                env_fn=env_factory,
                seed=int(seed),
            )
            learning_timesteps = int(timesteps)
        stop_candidate_schema = build_stop_candidate_schema(cfg)
        checkpoint_stop_schema = getattr(model, "stop_candidate_schema", None)
        if args.resume_checkpoint and checkpoint_stop_schema != stop_candidate_schema:
            raise ValueError(
                "resume checkpoint stop-candidate schema is missing or differs from the current "
                "run config; do not resume under a silently changed stop menu"
            )
        setattr(model, "stop_candidate_schema", stop_candidate_schema)
        _log_effective_ppo(model, cfg)

        aux_cb = None
        aux_cfg = getattr(cfg, "auxiliary", None)
        if aux_cfg is not None and float(aux_cfg.get("aux_weight", 0.0)) > 0.0:
            aux_cb = AuxiliaryTrainerCallback(
                prediction_horizon=int(aux_cfg.get("prediction_horizon", 5)),
                aux_weight=float(aux_cfg.get("aux_weight", 0.1)),
                lr=float(aux_cfg.get("lr", 1e-4)),
                grad_steps=int(aux_cfg.get("grad_steps", 4)),
                batch_windows=int(aux_cfg.get("batch_windows", 256)),
            )
            log.info(
                "Auxiliary loss enabled: aux_weight=%.3f horizon=%d",
                aux_cb.aux_weight,
                aux_cb.prediction_horizon,
            )

        callbacks: list[
            PeriodicCheckpointCallback
            | AuxiliaryTrainerCallback
            | BestCheckpointEvalCallback
            | ProgressLoggerCallback
            | LearningDiagnosticsCallback
            | LagrangianRiskCallback
            | ClipLogStdCallback
            | EpisodeEquityCallback
            | EarlyAbortCallback
            | TrainingDashboardCallback
        ] = [c for c in (lagrangian_cb, checkpoint_callback, aux_cb) if c is not None]

        callbacks.append(EpisodeEquityCallback(log_path=seed_model_dir / "episode_equity.csv"))
        if bool(cfg.get("quality_diagnostics", {}).get("enabled", False)):
            callbacks.append(
                LearningDiagnosticsCallback(log_path=seed_model_dir / "learning_diagnostics.csv")
            )
        abort_cb = _early_abort_callback(cfg)
        if abort_cb is not None:
            callbacks.append(abort_cb)
        self.phase_timings_seconds = phase_timings_seconds
        self.seed_dir = seed_dir
        self.seed_model_dir = seed_model_dir
        self.train_env = train_env
        self.env_factory = env_factory
        self.timesteps = int(timesteps)
        self.model = model
        self.learning_timesteps = int(learning_timesteps)
        self.stop_candidate_schema = stop_candidate_schema
        self.callbacks = callbacks
        self.abort_cb = abort_cb

    def train_model(self) -> None:
        ctx = self.context
        args = ctx.args
        cfg = ctx.cfg
        train_env = self.train_env
        model = self.model
        callbacks = self.callbacks
        seed_model_dir = self.seed_model_dir
        seed_dir = self.seed_dir
        timesteps = self.timesteps
        learning_timesteps = self.learning_timesteps
        abort_cb = self.abort_cb
        phase_timings_seconds = self.phase_timings_seconds
        train_bar_count = ctx.train_bar_count
        best_bars = ctx.best_bars
        best_features = ctx.best_features
        train_bars = ctx.train_bars
        train_feat = ctx.train_feat
        bundle_path = ctx.bundle_path
        train_env_spec = ctx.train_env_spec
        cache_dir = ctx.cache_dir
        feat_cache = ctx.feat_cache
        primary_ticks = ctx.primary_ticks
        env_vae = ctx.env_vae
        pre_ny_by_date = ctx.pre_ny_by_date
        dashboard_bars = ctx.dashboard_bars
        dashboard_features = ctx.dashboard_features
        # Best checkpoint: one short prefix of the train calendar, scored by
        # end equity. Full replay risk checks run after training; PnL and trade
        # counts are reported as metrics and do not stop this callback.
        short_n = min(20_000, train_bar_count)
        short_bars = best_bars if best_bars is not None else train_bars.iloc[:short_n]
        short_feat = best_features if best_features is not None else train_feat.iloc[:short_n]
        if bundle_path is not None and train_env_spec is not None:
            from simulation.environment.bundles import build_env_bundle
            from simulation.environment.spec import EnvSpec
            from simulation.environment.worker import make_bundle_env

            short_bundle = build_env_bundle(
                cache_dir,
                short_bars,
                short_feat,
                cfg,
                raw_columns=list(_strategy_from_cfg(cfg)[0].raw_columns),
                feature_config_hash=feat_cache.stem,
                tickbook=ticks_covering(primary_ticks, short_bars),
                pre_ny_by_date=None,
            )
            best_eval_spec = EnvSpec.from_cfg(
                cfg,
                algo=args.algo,
                reward=args.reward,
                arch=args.arch,
                bundle_dir=str(short_bundle),
                episodic=False,
                evaluation=True,
            )
            best_eval_factory = partial(make_bundle_env, best_eval_spec)
        else:
            best_eval_factory = partial(
                make_env,
                short_bars,
                short_feat,
                cfg,
                algo=args.algo,
                reward=args.reward,
                episodic=False,
                use_vae=args.use_vae,
                vae=env_vae,
                pre_ny_by_date=pre_ny_by_date,
                tickbook=ticks_covering(primary_ticks, short_bars),
                arch=args.arch,
            )

        best_eval_freq = max(1, timesteps)  # once at the end
        best_cb = BestCheckpointEvalCallback(
            eval_env_factory=best_eval_factory,
            eval_freq=best_eval_freq,
            best_model_path=seed_model_dir / "ppo_best",
        )
        callbacks.append(best_cb)

        progress_log = seed_model_dir / "training_log.csv"
        callbacks.append(ProgressLoggerCallback(log_path=progress_log))
        if isinstance(train_env.action_space, spaces.Box):
            callbacks.append(
                ClipLogStdCallback(
                    log_std_min=float(cfg.ppo.get("log_std_min", -0.7)),
                    log_std_max=float(cfg.ppo.get("log_std_max", 0.0)),
                )
            )

        eval_common, _strategy, _strategy_reward, _strategy_weight, strategy_actions = (
            _build_eval_common(cfg, args, env_vae, pre_ny_by_date, arch=args.arch)
        )
        dash = _dashboard_block(cfg)
        if bool(dash.get("enabled", True)):
            eval_bars = int(dash.get("eval_bars", 20_000))
            slice_n = min(eval_bars, train_bar_count)
            dash_bars = dashboard_bars if dashboard_bars is not None else train_bars.iloc[:slice_n]
            dash_feat = (
                dashboard_features if dashboard_features is not None else train_feat.iloc[:slice_n]
            )

            def _dashboard_eval() -> dict[str, Any]:
                result = evaluate_model(
                    model,
                    bars=dash_bars,
                    features=dash_feat,
                    max_episode_steps=None,
                    deterministic=False,
                    tickbook=ticks_covering(primary_ticks, dash_bars),
                    **eval_common,
                )
                metrics = calculate_metrics(
                    result["equity"],
                    trades=result["trades"],
                    n_sessions=result.get("n_sessions", 1),
                    n_breach_sessions=result.get("n_breach_sessions", 0),
                    periods_per_year=evaluation_periods_per_year(cfg),
                )
                n_steps = max(1, int(result.get("n_steps", 1)))
                from evaluation.metrics.decision_diversity import (
                    format_diversity_text,
                    thresholds_from_mapping,
                )

                diversity = format_diversity_text(
                    result["trades"],
                    thresholds_from_mapping(dash.get("diversity")),
                )
                return {
                    "evaluated_bars": int(len(dash_bars)),
                    "evaluation_split": "training_prefix_monitor",
                    "return_pct": float(metrics.total_return_pct),
                    "total_pnl": float(metrics.total_pnl),
                    "account_currency": str(cfg.account.currency),
                    "sharpe": float(metrics.sharpe),
                    "sortino": float(metrics.sortino),
                    "max_drawdown": float(metrics.max_drawdown),
                    "profit_factor": float(metrics.profit_factor),
                    "win_rate": float(metrics.win_rate),
                    "avg_trade": float(metrics.avg_trade),
                    "n_trades": float(metrics.n_trades),
                    "turnover": float(metrics.turnover),
                    "reward_mean": float(result.get("reward_sum", 0.0)) / n_steps,
                    "trading_policy_mode": result.get("trading_policy_mode", "legacy"),
                    "cumulative_payouts": float(result.get("cumulative_payouts", 0.0)),
                    "cycle_count": int(result.get("cycle_count", 0)),
                    "cycle_successes": int(result.get("cycle_successes", 0)),
                    "cycle_failures": int(result.get("cycle_failures", 0)),
                    "total_economic_value": float(result.get("total_economic_value", 0.0)),
                    "diversity_text": diversity,
                }

            callbacks.append(
                TrainingDashboardCallback(
                    total_timesteps=timesteps,
                    eval_every=int(dash.get("eval_every", 100_000)),
                    eval_fn=_dashboard_eval,
                    log_path=seed_model_dir / "dashboard_log.csv",
                    log_every=int(dash.get("log_every", 100_000)),
                )
            )

        log_every = int(dash.get("log_every", 100_000))
        log_interval = ppo_log_interval(
            n_steps=int(model.n_steps),
            n_envs=int(model.n_envs),
            log_every=log_every,
        )
        log.info("SB3 log every %d rollouts (~%d env steps)", log_interval, log_every)
        if isinstance(train_env.action_space, spaces.Box):
            install_post_update_std_clip(
                model,
                float(cfg.ppo.get("log_std_min", -0.7)),
                float(cfg.ppo.get("log_std_max", 0.0)),
            )
        debug_cfg = cfg.training.get("debug", {})
        if bool(debug_cfg.get("enabled", False)):
            install_flight_recorder(model, seed_dir, thresholds=debug_cfg)
        _initialize_value_head(model, resumed=bool(args.resume_checkpoint))
        phase_started = time.perf_counter()
        if learning_timesteps > 0:
            model.learn(
                total_timesteps=learning_timesteps,
                callback=cast(Any, callbacks),
                log_interval=log_interval,
                reset_num_timesteps=not bool(args.resume_checkpoint),
            )
        elif args.resume_checkpoint:
            best_cb.model = model
            best_cb._run_eval()
        phase_timings_seconds["ppo_learn"] = time.perf_counter() - phase_started
        log.info("PHASE_TIMING ppo_learn_seconds=%.3f", phase_timings_seconds["ppo_learn"])
        abort_reason = abort_cb.reason if abort_cb is not None else None
        if abort_reason:
            (seed_dir / "early_abort.json").write_text(
                json.dumps(
                    {"reason": abort_reason, "timesteps": int(model.num_timesteps)},
                    indent=2,
                )
            )
            log.error("EARLY_ABORT %s at timesteps=%s", abort_reason, model.num_timesteps)

        self.eval_common = eval_common
        self.strategy_actions = strategy_actions
        self.progress_log = progress_log
        self.abort_reason = abort_reason

    def evaluate_test_split(self) -> None:
        ctx = self.context
        cfg = ctx.cfg
        model = self.model
        seed_model_dir = self.seed_model_dir
        progress_log = self.progress_log
        phase_timings_seconds = self.phase_timings_seconds
        eval_common = self.eval_common
        test_eval_spec = ctx.test_eval_spec
        test_bars = ctx.test_bars
        test_feat = ctx.test_feat
        primary_ticks = ctx.primary_ticks
        # Save final model
        model_path = seed_model_dir / "ppo_final"
        save_ppo_checkpoint(model, model_path)
        log.info("Model saved: %s", model_path)

        # Learning-curve / loss charts from the SB3 progress CSV.
        try:
            from evaluation.plotting.training_plots import save_dashboard_plots, save_training_plots

            save_training_plots(
                progress_log,
                out_dir=seed_model_dir,
                dpi=getattr(cfg.output, "dpi", 150),
                save_html=getattr(cfg.output, "save_html", False),
            )
            save_dashboard_plots(
                seed_model_dir / "dashboard_log.csv",
                out_dir=seed_model_dir,
                dpi=getattr(cfg.output, "dpi", 150),
                save_html=getattr(cfg.output, "save_html", False),
            )
        except Exception as exc:
            log.warning("Training progress plots skipped: %s", exc)

        # Final report-only held-out evaluation uses deterministic policy actions.
        phase_started = time.perf_counter()
        log.info("Evaluating trained model on test set...")
        if test_eval_spec is not None:
            from simulation.environment.worker import make_bundle_env

            test_eval_env = make_bundle_env(test_eval_spec)
            empty_bars = pd.DataFrame()
            empty_features = pd.DataFrame()
            test_ticks = None
        else:
            test_eval_env = None
            empty_bars = test_bars
            empty_features = test_feat
            test_ticks = ticks_covering(primary_ticks, test_bars)
        test_result = _evaluate_final_split(
            model,
            bars=empty_bars,
            features=empty_features,
            max_episode_steps=None,
            tickbook=test_ticks,
            env=test_eval_env,
            **eval_common,
        )
        test_result["initial_balance"] = cfg.account.initial_balance
        test_m = calculate_metrics(
            test_result["equity"],
            trades=test_result["trades"],
            n_sessions=test_result.get("n_sessions", 1),
            n_breach_sessions=test_result.get("n_breach_sessions", 0),
            periods_per_year=evaluation_periods_per_year(cfg),
        )
        test_dir_sum = _direction_summary(test_result["trades"])
        log.info(
            "[test/report-only] NetPnL=%.2f %s  Return=%.2f%%  "
            "MaxDD=%.2f%%  Sharpe=%.3f  ProfitFactor=%.3f  WinRate=%.1f%%  Trades=%d  "
            "survived=%s  days=%d/%d  fail_time=%s  max_trailing_dd=%.2f%%  direction=%s  entry_diag=%s",
            test_m.total_pnl,
            str(cfg.account.currency),
            test_m.total_return_pct,
            test_m.max_drawdown * 100,
            test_m.sharpe,
            test_m.profit_factor,
            test_m.win_rate * 100,
            test_m.n_trades,
            test_result.get("survived_full_year"),
            test_result.get("days_traded", 0),
            test_result.get("n_sessions", 0),
            test_result.get("fail_time"),
            test_m.max_drawdown * 100,
            test_dir_sum,
            test_result.get("entry_diag", {}),
        )
        phase_timings_seconds["test_evaluation"] = time.perf_counter() - phase_started
        log.info(
            "PHASE_TIMING test_evaluation_seconds=%.3f",
            phase_timings_seconds["test_evaluation"],
        )

        self.model_path = model_path
        self.test_result = test_result
        self.test_m = test_m

    def evaluate_train_split(self) -> None:
        ctx = self.context
        cfg = ctx.cfg
        seed = self.seed
        model = self.model
        seed_dir = self.seed_dir
        phase_timings_seconds = self.phase_timings_seconds
        eval_common = self.eval_common
        test_result = self.test_result
        test_m = self.test_m
        train_eval_spec = ctx.train_eval_spec
        train_bars = ctx.train_bars
        train_feat = ctx.train_feat
        primary_ticks = ctx.primary_ticks
        # In-sample evaluation on the training split so the run dir carries the same
        # training/ + testing/ artifact layout as the baseline runners.
        phase_started = time.perf_counter()
        log.info("Evaluating trained model on training set (in-sample)...")
        if train_eval_spec is not None:
            from simulation.environment.worker import make_bundle_env

            train_eval_env = make_bundle_env(train_eval_spec)
            empty_bars = pd.DataFrame()
            empty_features = pd.DataFrame()
            train_ticks = None
        else:
            train_eval_env = None
            empty_bars = train_bars
            empty_features = train_feat
            train_ticks = ticks_covering(primary_ticks, train_bars)
        train_result = _evaluate_final_split(
            model,
            bars=empty_bars,
            features=empty_features,
            max_episode_steps=None,
            tickbook=train_ticks,
            env=train_eval_env,
            **eval_common,
        )
        train_result["initial_balance"] = cfg.account.initial_balance
        train_m = calculate_metrics(
            train_result["equity"],
            trades=train_result["trades"],
            n_sessions=train_result.get("n_sessions", 1),
            n_breach_sessions=train_result.get("n_breach_sessions", 0),
            periods_per_year=evaluation_periods_per_year(cfg),
        )
        train_dir_sum = _direction_summary(train_result["trades"])
        train_gate = assess_train_equity(
            train_result["equity"],
            initial=float(cfg.account.initial_balance),
            breached=not bool(train_result.get("survived_full_year", False)),
        )
        (seed_dir / "train_equity_gate.json").write_text(json.dumps(train_gate, indent=2))
        if not train_gate["ok"]:
            log.error("TRAIN_SURVIVAL_FAILED %s", train_gate)
        log.info(
            "[train/in-sample] NetPnL=%.2f %s  Return=%.2f%%  "
            "MaxDD=%.2f%%  Sharpe=%.3f  ProfitFactor=%.3f  WinRate=%.1f%%  Trades=%d  "
            "survived=%s  days=%d/%d  fail_time=%s  max_trailing_dd=%.2f%%  direction=%s  entry_diag=%s",
            train_m.total_pnl,
            str(cfg.account.currency),
            train_m.total_return_pct,
            train_m.max_drawdown * 100,
            train_m.sharpe,
            train_m.profit_factor,
            train_m.win_rate * 100,
            train_m.n_trades,
            train_result.get("survived_full_year"),
            train_result.get("days_traded", 0),
            train_result.get("n_sessions", 0),
            train_result.get("fail_time"),
            train_m.max_drawdown * 100,
            train_dir_sum,
            train_result.get("entry_diag", {}),
        )
        phase_timings_seconds["train_evaluation"] = time.perf_counter() - phase_started
        log.info(
            "PHASE_TIMING train_evaluation_seconds=%.3f",
            phase_timings_seconds["train_evaluation"],
        )
        _write_evaluation_artifacts(
            run_dir=seed_dir,
            seed=int(seed),
            completed_timesteps=int(model.num_timesteps),
            train_result=train_result,
            test_result=test_result,
            train_metrics=train_m,
            test_metrics=test_m,
            train_gate=train_gate,
            account_currency=str(cfg.account.currency),
        )
        log.info(
            "Final evaluation report written: %s",
            seed_dir / "evaluations" / "summary.md",
        )

        self.train_result = train_result
        self.train_m = train_m
        self.train_gate = train_gate

    def write_artifacts(self) -> None:
        ctx = self.context
        args = ctx.args
        cfg = ctx.cfg
        seed = self.seed
        model = self.model
        seed_dir = self.seed_dir
        train_env = self.train_env
        timesteps = self.timesteps
        stop_candidate_schema = self.stop_candidate_schema
        model_path = self.model_path
        test_result = self.test_result
        test_m = self.test_m
        train_result = self.train_result
        train_m = self.train_m
        phase_timings_seconds = self.phase_timings_seconds
        bundle_path = ctx.bundle_path
        test_bundle_path = ctx.test_bundle_path
        train_bars = ctx.train_bars
        test_bars = ctx.test_bars
        train_sec = ctx.train_sec
        test_sec = ctx.test_sec
        feat_cache = ctx.feat_cache
        dataset_records = ctx.dataset_records
        train_start = ctx.train_start
        train_end = ctx.train_end
        test_start = ctx.test_start
        data_source = ctx.data_source
        primary_ticks = ctx.primary_ticks
        primary_sym = ctx.primary_sym
        schedule_path = ctx.schedule_path
        is_multi_seed = ctx.is_multi_seed
        # Export both splits for this seed
        phase_started = time.perf_counter()
        if bundle_path is not None:
            from simulation.environment.bundles import bars_for_export

            train_export_bars = bars_for_export(bundle_path)
            test_export_bars = bars_for_export(test_bundle_path) if test_bundle_path else None
        else:
            train_export_bars = train_bars
            test_export_bars = test_bars
        save_run(
            run_dir=seed_dir,
            train_result=train_result,
            train_metrics=train_m,
            train_bars=train_export_bars,
            train_secondary=train_sec,
            test_result=test_result,
            test_metrics=test_m,
            test_bars=test_export_bars,
            test_secondary=test_sec,
            cfg=cfg,
            save_plots=getattr(cfg.output, "save_plots", True),
            save_html=getattr(cfg.output, "save_html", False),
            save_csv=getattr(cfg.output, "save_csv", True),
            dpi=getattr(cfg.output, "dpi", 150),
            max_order_charts=int(getattr(cfg.output, "max_order_charts", 0)),
            report_profile=str(getattr(cfg.output, "report_profile", "full")),
        )
        if bundle_path is not None:
            del train_export_bars, test_export_bars

        # Save config snapshot for this seed
        (seed_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        per_run_cache_hash = Path(str(feat_cache) + ".hash")
        cache_hash = per_run_cache_hash.read_text(encoding="utf-8").strip()
        checkpoint_path = Path(f"{model_path}.zip")
        if not checkpoint_path.is_file():
            checkpoint_path = model_path
        sequence_feature_schema = _sequence_feature_schema_manifest(train_env)
        manifest = build_training_run_manifest(
            repository=repository_root(__file__),
            config_path=seed_dir / "config.yaml",
            datasets={},
            dataset_records=dataset_records,
            split={
                "train_start": str(train_start or ""),
                "train_end": str(train_end),
                "test_start": str(test_start),
            },
            seed=int(seed),
            completed_steps=int(model.num_timesteps),
            action_schema=_action_schema_manifest(train_env),
            observation_schema=_observation_schema_manifest(train_env),
            sequence_feature_schema=sequence_feature_schema,
            stop_candidate_schema=stop_candidate_schema,
            feature_cache_path=feat_cache,
            feature_cache_hash=cache_hash,
            checkpoint_path=checkpoint_path,
            execution_protocol={
                "data_source": data_source,
                "trading_policy": resolve_trading_policy(cfg).to_dict(),
                "costs": OmegaConf.to_container(cfg.costs, resolve=True) if "costs" in cfg else {},
                "execution": OmegaConf.to_container(cfg.execution, resolve=True)
                if "execution" in cfg
                else {},
                "fill_delay_ms": _fill_delay_ms(cfg),
                "block_overnight": bool(cfg.env.get("block_overnight", True)),
                "tick_data_configured": bool(primary_ticks is not None),
                "primary_symbol": str(primary_sym),
                "policy_market_context": {
                    "version": POLICY_MARKET_CONTEXT_VERSION,
                    "schedule_file": str(schedule_path.resolve()),
                    "schedule_sha256": file_sha256(schedule_path),
                    "features": [
                        "ny_scheduled_short",
                        "ny_scheduled_close_fraction",
                        "ny_minutes_to_scheduled_close",
                        "tick_bid_offset_atr",
                        "tick_ask_offset_atr",
                        "tick_spread_atr",
                        "tick_quote_age_fraction",
                        "tick_quote_activity",
                        "tick_quote_available",
                    ],
                    "quote_interval": "completed M1 bar [open, next open)",
                    "observation_visibility": "feature row is available after its bar closes",
                },
            },
            training_architecture=_training_architecture_provenance(model, cfg, args.arch),
            training_protocol=_training_protocol_manifest(
                model=model,
                cfg=cfg,
                args=args,
                seed=int(seed),
                requested_steps=int(timesteps),
            ),
            metrics_protocol={
                "annualization": {
                    "periods_per_year": evaluation_periods_per_year(cfg),
                    "convention": str(cfg.evaluation.annualization_convention),
                },
                "initial_balance": float(cfg.account.initial_balance),
                "execution_seed": int(seed),
                "historical_test_role": "previously_inspected_holdout",
                "frequency": "bar_close_equity",
                "exposure_measure": BAR_CLOSE_EXPOSURE_MEASURE,
            },
            artifact_paths={
                "checkpoint": str(checkpoint_path.relative_to(seed_dir)),
                "training_log": str((seed_dir / "training_log.csv").relative_to(seed_dir)),
                "trade_log": str((seed_dir / "trade_log.json").relative_to(seed_dir)),
            },
        )
        if not manifest["source"]["available"] or not manifest["source"]["head"]:
            raise RuntimeError("training provenance requires a readable Git source identity")
        manifest["feature_cache"]["file_sha256"] = (
            file_sha256(feat_cache) if feat_cache.is_file() else None
        )
        manifest_path = seed_dir / "run_manifest.json"
        write_run_manifest(manifest_path, manifest)
        phase_timings_seconds["report_export"] = time.perf_counter() - phase_started
        log.info("PHASE_TIMING report_export_seconds=%.3f", phase_timings_seconds["report_export"])

        # Per-seed artifacts (multi-seed mode only)
        if is_multi_seed:
            _save_seed_artifacts(seed_dir, seed, cfg, train_result, test_result, train_m, test_m)

        self.sequence_feature_schema = sequence_feature_schema
        self.manifest_path = manifest_path

    def build_result(self) -> dict[str, Any]:
        ctx = self.context
        args = ctx.args
        cfg = ctx.cfg
        seed = self.seed
        train_env = self.train_env
        model = self.model
        test_result = self.test_result
        test_m = self.test_m
        train_result = self.train_result
        train_m = self.train_m
        train_gate = self.train_gate
        abort_reason = self.abort_reason
        sequence_feature_schema = self.sequence_feature_schema
        phase_timings_seconds = self.phase_timings_seconds
        manifest_path = self.manifest_path
        run_dir = ctx.run_dir
        # Collect per-seed results for aggregation
        entry_diag = test_result.get("entry_diag", {})
        entry_state_diag = test_result.get("entry_state_diagnostics", {})
        action_space_width = _action_space_width(train_env)
        action_schema = _action_schema_manifest(train_env)
        observation_account_dim = _observation_account_dim(train_env)
        observation_space_shapes = _observation_space_shapes(train_env)
        observation_schema = _observation_schema_manifest(train_env)
        env_flags = {
            key: bool(cfg.env.get(key, False))
            for key in (
                "strategy_actions",
                "allow_agent_sl_mode",
                "allow_agent_tp_mode",
                "allow_multi_tp",
                "allow_simplex",
                "entry_state_machine",
                "entry_state_observation",
                "arm_requires_retest",
            )
        }
        oos_role = (
            "selection_acknowledged"
            if getattr(args, "allow_locked_oos_for_selection", False)
            else "final_report_only"
        )
        return {
            "seed": seed,
            "train_m": train_m,
            "test_m": test_m,
            "train_result": train_result,
            "test_result": test_result,
            "train_gate": train_gate,
            "abort_reason": abort_reason,
            "model_num_timesteps": int(model.num_timesteps),
            "entry_diag": entry_diag,
            "entry_state_diagnostics": entry_state_diag,
            "action_space_width": action_space_width,
            "action_schema": action_schema,
            "observation_account_dim": observation_account_dim,
            "observation_space_shapes": observation_space_shapes,
            "observation_schema": observation_schema,
            "sequence_feature_schema": sequence_feature_schema,
            "env_flags": env_flags,
            "oos_role": oos_role,
            "phase_timings_seconds": phase_timings_seconds,
            "run_manifest": str(manifest_path.relative_to(run_dir)),
        }

    def run(self) -> dict[str, Any]:
        self.prepare_training()
        self.train_model()
        self.evaluate_test_split()
        self.evaluate_train_split()
        self.write_artifacts()
        return self.build_result()


def run_seed_training(context: SeedRunContext) -> tuple[list[dict[str, Any]], int]:
    """Run every configured seed and return aggregation payloads plus requested timesteps."""
    results: list[dict[str, Any]] = []
    timesteps = 0
    for seed in context.seeds:
        runner = SeedRunner(context, seed)
        results.append(runner.run())
        timesteps = runner.timesteps
    return results, timesteps
