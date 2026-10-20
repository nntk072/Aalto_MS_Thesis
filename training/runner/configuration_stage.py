"""Focused training configuration preparation stage."""

from __future__ import annotations

import logging

from compute.environment.device import get_device, scale_training_cfg
from compute.environment.precision import configure_accelerator_for_training
from training.configuration.cli import load_merged_config, parse_train_args
from training.runner.seed import _setup_rngs
from training.setup.callbacks import (
    _build_lagrangian_callback,
)
from training.setup.models import (
    _validate_q5_resume_state,
)

from .stages import TrainingConfigurationStage

log = logging.getLogger(__name__)


def _prepare_training_configuration() -> TrainingConfigurationStage:
    args = parse_train_args()

    if args.use_vae and not args.vae_path:
        raise SystemExit("--use-vae requires --vae-path pointing at a trained VAE .pth")
    if args.walk_forward and args.use_vae:
        raise SystemExit(
            "Walk-forward with an externally fitted VAE has unverified fold provenance; "
            "provide a fold-scoped VAE fitting pipeline before enabling this combination"
        )

    cfg = load_merged_config(args)
    configure_accelerator_for_training(cfg.get("device"))
    _setup_rngs(args.seed)
    try:
        _validate_q5_resume_state(cfg, resume_checkpoint=args.resume_checkpoint)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.arch is None:
        args.arch = str(cfg.agent.get("arch", "tcn"))
    elif getattr(args, "quality_variant", None) and args.arch != str(cfg.agent.get("arch", "tcn")):
        raise SystemExit(
            f"--arch={args.arch} conflicts with the selected quality variant encoder "
            f"{cfg.agent.get('arch', 'tcn')}"
        )

    device = get_device(cfg.get("device"))
    requested_n_envs = int(cfg.env.n_envs)
    quality_variant = str(cfg.get("quality_variant_id", ""))
    exact_n_envs = False
    preserve_ppo_geometry = False
    if quality_variant:
        from training.configuration.quality import quality_variant_resource_contract

        resource_contract = quality_variant_resource_contract(quality_variant)
        exact_n_envs = bool(resource_contract.get("exact_n_envs", False))
        preserve_ppo_geometry = exact_n_envs and "aggregate_rollout_steps" in resource_contract
    cfg.quality_execution.requested_n_envs = requested_n_envs
    cfg.quality_execution.exact_n_envs = exact_n_envs
    scale_training_cfg(
        cfg,
        device,
        args.arch,
        preserve_ppo_geometry=preserve_ppo_geometry,
    )
    cfg.quality_execution.effective_n_envs = int(cfg.env.n_envs)
    if quality_variant and exact_n_envs:
        from training.configuration.quality import validate_quality_resource_shape

        try:
            validate_quality_resource_shape(
                quality_variant,
                requested_n_envs=requested_n_envs,
                effective_n_envs=int(cfg.env.n_envs),
                aggregate_rollout_steps=int(cfg.env.n_envs * cfg.ppo.n_steps),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    log.info(
        "Using device: %s  n_envs=%s  ppo.batch_size=%s  ppo.n_steps=%s  n_epochs=%s",
        device,
        cfg.env.n_envs,
        cfg.ppo.batch_size,
        cfg.ppo.n_steps,
        cfg.ppo.n_epochs,
    )

    # Override for MVP mode
    if args.mvp:
        log.info("MVP mode: using first 30 days of training data")
        cfg.training.use_m1_only = True
        cfg.training.max_days = 30
    lagrangian_cb = _build_lagrangian_callback(cfg, algo=args.algo, mvp=args.mvp)
    if lagrangian_cb is not None:
        log.info(
            "Q5 primal-dual PPO enabled: episode_cost_budget=%.4f dual_lr=%.4f lambda_max=%.3f",
            lagrangian_cb.episode_cost_budget,
            lagrangian_cb.dual_learning_rate,
            lagrangian_cb.lambda_max,
        )
    return TrainingConfigurationStage(
        args=args, cfg=cfg, device=device, lagrangian_cb=lagrangian_cb
    )
