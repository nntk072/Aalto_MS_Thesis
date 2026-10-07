# mypy: ignore-errors
"""Rescore saved PPO checkpoints after SL/TP fills use the level price.

Writes a new directory per run. Does not modify the source output tree.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.utils._triton as _torch_triton

# This CPU env has no usable Triton compiler. Torch 2.14 still treats the
# package as present and then crashes while building the Adam optimizer.
_torch_triton.has_triton_package = lambda: False  # type: ignore[method-assign]

from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_rl.data.pipeline import run_pipeline  # noqa: E402
from quant_rl.data.split import get_split_config, make_train_mask, split_train_test  # noqa: E402
from quant_rl.eval.rollout import evaluate_model  # noqa: E402
from quant_rl.evaluation import (  # noqa: E402
    build_comparison_table,
    calculate_metrics,
    save_metrics_json,
)
from quant_rl.features.build import build_features, feature_cache_path  # noqa: E402
from quant_rl.models.encoder import GRUEncoder, TCNEncoder, TransformerEncoder  # noqa: E402
from quant_rl.models.ppo_policy import ClampedStdMultiInputPolicy  # noqa: E402
from quant_rl.train.train_rl import (  # noqa: E402
    _eval_guardrail_kwargs,
    _max_loss_per_trade,
    _strategy_from_cfg,
    _strategy_risk_ranges,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rescore_sl_fill")

RUNS: dict[str, str] = {
    "base_gru": "20260922_024235_116145_rl_train_seed50_gru",
    "base_tcn": "20260922_024235_194714_rl_train_seed50_tcn",
    "base_tf": "20260922_024235_109978_rl_train_seed50_transformer",
    "po3_tcn": "20260922_024144_516380_rl_train_seed50_tcn",
    "po3_gru": "20260922_024144_521964_rl_train_seed50_gru",
    "po3_tf": "20260922_024144_531686_rl_train_seed50_transformer",
}


def _closes(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty or "type" not in df.columns:
        return []
    last = None
    out: list[dict] = []
    for rec in df.to_dict(orient="records"):
        kind = str(rec.get("type", ""))
        if kind == "open":
            last = rec
            continue
        if "close" not in kind or last is None:
            continue
        try:
            pnl = float(rec["pnl"])
            direction = float(last["direction"])
            lots = float(last["lots"])
        except (TypeError, ValueError):
            last = None
            continue
        out.append(
            {
                "open_time": str(last.get("time")),
                "direction": direction,
                "lots": lots,
                "entry": float(last["price"]) if last.get("price") not in (None, "") else None,
                "sl": float(last["sl_price"]) if last.get("sl_price") not in (None, "") else None,
                "exit": float(rec["price"]) if rec.get("price") not in (None, "") else None,
                "pnl": pnl,
                "reason": str(rec.get("reason", "")),
            }
        )
        last = None
    return out


def _sides(closes: list[dict]) -> tuple[int, int]:
    n_long = sum(1 for c in closes if c["direction"] > 0)
    n_short = sum(1 for c in closes if c["direction"] < 0)
    return n_long, n_short


def _fill_delta(old: list[dict], new: list[dict]) -> dict:
    from collections import Counter

    old_counts = Counter((c["open_time"], c["direction"]) for c in old)
    new_counts = Counter((c["open_time"], c["direction"]) for c in new)
    old_pnl: dict[tuple, list[float]] = {}
    for c in old:
        old_pnl.setdefault((c["open_time"], c["direction"]), []).append(c["pnl"])
    moved_n = 0
    moved_delta = 0.0
    matched = 0
    for c in new:
        key = (c["open_time"], c["direction"])
        bucket = old_pnl.get(key)
        if not bucket:
            continue
        prev = bucket.pop(0)
        matched += 1
        delta = c["pnl"] - prev
        if abs(delta) > 1.0:
            moved_n += 1
            moved_delta += delta
    shared = sum((old_counts & new_counts).values())
    union = sum((old_counts | new_counts).values())
    return {
        "matched_closes": matched,
        "n_fill_moved_gt_1usd": moved_n,
        "pnl_delta_of_moved": moved_delta,
        "jaccard_open_time_dir": (shared / union) if union else 0.0,
    }


def _apr20(closes: list[dict]) -> dict | None:
    for c in closes:
        if c["open_time"].startswith("2026-04-20 16:31") and c["direction"] > 0:
            return {
                "open_time": c["open_time"],
                "lots": c["lots"],
                "entry": c["entry"],
                "sl": c["sl"],
                "exit": c["exit"],
                "pnl": c["pnl"],
                "reason": c["reason"],
            }
    return None


def _lot_block(closes: list[dict]) -> dict:
    lots = sorted(c["lots"] for c in closes)
    if not lots:
        return {"n": 0, "median": None, "n_gt_30": 0, "pnl_gt_30": 0.0}
    mid = lots[len(lots) // 2]
    big = [c for c in closes if c["lots"] > 30]
    return {
        "n": len(lots),
        "median": mid,
        "n_gt_30": len(big),
        "pnl_gt_30": sum(c["pnl"] for c in big),
    }


def _eval_common(cfg) -> dict:
    strategy, strategy_reward, strategy_weight = _strategy_from_cfg(cfg)
    risk_frac_range, rr_ratio_range = _strategy_risk_ranges(cfg)
    return dict(
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        guardrail_kwargs=_eval_guardrail_kwargs(cfg),
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=_max_loss_per_trade(cfg),
        dsr_eta=cfg.env.reward_dsr_eta,
        continuous_actions=False,
        block_overnight=bool(cfg.env.get("block_overnight", True)),
        eod_risk=dict(cfg.env.get("eod_risk", {})),
        strategy=strategy,
        strategy_actions=bool(cfg.env.get("strategy_actions", False)),
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        sl_buffer_pts=float(cfg.env.get("sl_buffer_pts", 0.0)),
        min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
        max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
        entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
        reward_mode=str(cfg.env.get("reward_mode", "dsr")),
        entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
        max_episode_steps=None,
    )


def _split_block(result: dict, metrics, closes: list[dict]) -> dict:
    n_long, n_short = _sides(closes)
    n = n_long + n_short
    return {
        "pnl": float(metrics.total_pnl),
        "return_pct": float(metrics.total_return_pct),
        "sharpe": float(metrics.sharpe),
        "max_drawdown": float(metrics.max_drawdown),
        "n_trades": int(metrics.n_trades),
        "n_long": n_long,
        "n_short": n_short,
        "long_share": (n_long / n) if n else None,
        "breaches": int(result.get("n_breach_sessions", 0)),
        "fail_time": None if result.get("fail_time") is None else str(result.get("fail_time")),
        "survived_full_year": bool(result.get("survived_full_year", False)),
        "days_traded": int(result.get("days_traded", 0)),
        "lots": _lot_block(closes),
    }


def rescore_one(name: str, source: Path, out_root: Path) -> dict:
    cfg = OmegaConf.load(source / "config.yaml")
    ckpt = source / "model" / "ppo_final.zip"
    log.info("Loading %s from %s", name, ckpt)
    _ = (ClampedStdMultiInputPolicy, TCNEncoder, GRUEncoder, TransformerEncoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(ckpt, device=device)
    log.info("%s device=%s", name, device)

    data = run_pipeline(cfg, force=False)
    primary = data[cfg.data.primary]["M1"]
    secondary = data.get(cfg.data.secondary, {}).get("M1")
    train_end, test_start = get_split_config(cfg)
    train_mask = make_train_mask(primary.index, train_end)
    cache = feature_cache_path(
        Path(cfg.data.cache_dir), cfg.data.primary, cfg, primary, train_mask=train_mask
    )
    features = build_features(
        primary,
        secondary=secondary,
        cfg=cfg,
        train_mask=train_mask,
        cache_path=cache,
    )
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary, features, train_end, test_start
    )
    common = _eval_common(cfg)

    log.info("%s test bars=%d", name, len(test_bars))
    test_result = evaluate_model(model, bars=test_bars, features=test_feat, **common)
    test_m = calculate_metrics(
        test_result["equity"],
        trades=test_result["trades"],
        n_sessions=test_result.get("n_sessions", 1),
        n_breach_sessions=test_result.get("n_breach_sessions", 0),
    )
    log.info("%s train bars=%d", name, len(train_bars))
    train_result = evaluate_model(model, bars=train_bars, features=train_feat, **common)
    train_m = calculate_metrics(
        train_result["equity"],
        trades=train_result["trades"],
        n_sessions=train_result.get("n_sessions", 1),
        n_breach_sessions=train_result.get("n_breach_sessions", 0),
    )

    out = out_root / name
    (out / "training").mkdir(parents=True, exist_ok=True)
    (out / "testing").mkdir(parents=True, exist_ok=True)
    train_result["trades"].to_csv(out / "training" / "trades.csv", index=False)
    test_result["trades"].to_csv(out / "testing" / "trades.csv", index=False)
    save_metrics_json(train_m, out / "training" / "metrics.json")
    save_metrics_json(test_m, out / "testing" / "metrics.json")
    (out / "summary.txt").write_text(build_comparison_table(train_m, test_m) + "\n")

    train_closes = _closes(train_result["trades"])
    test_closes = _closes(test_result["trades"])
    old_test = _closes(pd.read_csv(source / "testing" / "trades.csv"))
    old_train = _closes(pd.read_csv(source / "training" / "trades.csv"))
    checklist = {
        "name": name,
        "source": str(source),
        "reward_mode": str(cfg.env.get("reward_mode", "dsr")),
        "strategy_actions": bool(cfg.env.get("strategy_actions", False)),
        "ppo_batch_size": int(cfg.ppo.batch_size),
        "train": _split_block(train_result, train_m, train_closes),
        "test": _split_block(test_result, test_m, test_closes),
        "fill_delta_test": _fill_delta(old_test, test_closes),
        "fill_delta_train": _fill_delta(old_train, train_closes),
        "old_test_pnl": sum(c["pnl"] for c in old_test),
        "old_train_pnl": sum(c["pnl"] for c in old_train),
        "apr20_long": _apr20(test_closes),
    }
    (out / "checklist.json").write_text(json.dumps(checklist, indent=2) + "\n")
    log.info(
        "%s test pnl=%.1f long=%d short=%d fail=%s",
        name,
        checklist["test"]["pnl"],
        checklist["test"]["n_long"],
        checklist["test"]["n_short"],
        checklist["test"]["fail_time"],
    )
    return checklist


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    out_root = ROOT / "outputs" / "rescore_sl_fill"
    out_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    names = [only] if only else list(RUNS)
    for name in names:
        if name not in RUNS:
            raise SystemExit(f"unknown run {name}; choose from {list(RUNS)}")
        source = ROOT / "outputs" / RUNS[name]
        rescore_one(name, source, out_root)


if __name__ == "__main__":
    main()
