"""Headless tests for the ``live_trading_rl.py`` entrypoint helpers (Stage 8).

Covers the pieces of the live entrypoint that can be verified without MT5 or a
model checkpoint: SMT/secondary-symbol wiring decisions, live-risk extraction
(parity with the repo config), run-config resolution, and CLI/env defaults.
MetaTrader5 (Windows-only) is stubbed the same additive way as
``test_rl_robot.py`` so the module-level imports resolve on Linux.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

# --- additive headless MetaTrader5 stub (mirror of test_rl_robot.py) --------
if "MetaTrader5" not in sys.modules:
    try:  # pragma: no cover - depends on platform
        import MetaTrader5  # noqa: F401
    except ImportError:
        sys.modules["MetaTrader5"] = ModuleType("MetaTrader5")

_mt5_stub = sys.modules["MetaTrader5"]
for _attr in (
    "TIMEFRAME_M1",
    "TIMEFRAME_M5",
    "TIMEFRAME_M15",
    "TIMEFRAME_M30",
    "TIMEFRAME_H1",
    "TIMEFRAME_H4",
    "TIMEFRAME_D1",
):
    if not hasattr(_mt5_stub, _attr):
        setattr(_mt5_stub, _attr, 1)  # all timeframes equal is fine headless
if not hasattr(_mt5_stub, "ORDER_TYPE_BUY"):
    setattr(_mt5_stub, "ORDER_TYPE_BUY", 0)
if not hasattr(_mt5_stub, "ORDER_TYPE_SELL"):
    setattr(_mt5_stub, "ORDER_TYPE_SELL", 1)

import live_trading_rl  # noqa: E402


# --- _use_secondary_symbol: SMT wiring decision ----------------------------
def test_secondary_symbol_wired_when_mtf_training() -> None:
    """Mtf-trained runs (use_m1_only=False, primary_only=False) need SMT data."""
    cfg = OmegaConf.create({"training": {"use_m1_only": False, "primary_only": False}})
    assert live_trading_rl._use_secondary_symbol(cfg) is True


def test_secondary_symbol_skipped_when_m1_only() -> None:
    cfg = OmegaConf.create({"training": {"use_m1_only": True, "primary_only": False}})
    assert live_trading_rl._use_secondary_symbol(cfg) is False


def test_secondary_symbol_skipped_when_primary_only() -> None:
    cfg = OmegaConf.create({"training": {"use_m1_only": False, "primary_only": True}})
    assert live_trading_rl._use_secondary_symbol(cfg) is False


def test_secondary_symbol_defaults_true_without_training_block() -> None:
    """A config without a training block must not silently drop SMT features."""
    assert live_trading_rl._use_secondary_symbol(OmegaConf.create({})) is True


# --- load_risk_overrides: live risk parity ---------------------------------
def test_load_risk_overrides_reads_repo_default_yaml() -> None:
    """The repo default.yaml block must round-trip into the RiskManager dict."""
    cfg = OmegaConf.load("quant_rl/config/default.yaml")
    risk = live_trading_rl.load_risk_overrides(cfg)
    block = cfg.live_risk_overrides
    assert risk["risk_per_symbol"] == float(block.risk_per_symbol)
    assert risk["max_total_risk"] == float(block.max_total_risk)
    assert risk["default_lot_size"] == float(block.default_lot_size)
    assert risk["stop_loss_pips"] == float(block.stop_loss_pips)
    assert risk["interval_minutes"] == float(block.interval_minutes)


def test_load_risk_overrides_missing_block_falls_back_with_defaults() -> None:
    class NoBlock:
        pass

    risk = live_trading_rl.load_risk_overrides(NoBlock())
    assert risk == live_trading_rl._LIVE_RISK_DEFAULTS


def test_load_risk_overrides_partial_block_warns_and_fills(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dropped key must fall back explicitly, not silently shrink live risk."""
    warnings: list[str] = []
    monkeypatch.setattr(
        getattr(live_trading_rl, "logger"),
        "warning",
        lambda msg, *a: warnings.append(str(msg)),
    )
    cfg = OmegaConf.create({"live_risk_overrides": {"risk_per_symbol": 0.005}})
    risk = live_trading_rl.load_risk_overrides(cfg)
    assert risk["risk_per_symbol"] == 0.005
    assert risk["max_total_risk"] == live_trading_rl._LIVE_RISK_DEFAULTS["max_total_risk"]
    assert len(warnings) == len(live_trading_rl._LIVE_RISK_DEFAULTS) - 1


# --- resolve_run_config: train/live parity ---------------------------------
def test_resolve_run_config_prefers_saved_run_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "config.yaml").write_text("features:\n  window: 30\n", encoding="utf-8")
    model_path = run_dir / "model" / "ppo_final.zip"
    model_path.write_bytes(b"x")
    assert live_trading_rl.resolve_run_config(model_path) == run_dir / "config.yaml"


def test_resolve_run_config_falls_back_to_repo_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "model").mkdir(parents=True)
    model_path = run_dir / "model" / "ppo_final.zip"
    model_path.write_bytes(b"x")
    assert live_trading_rl.resolve_run_config(model_path) == live_trading_rl.DEFAULT_CONFIG


# --- parse_args: CLI/env contract ------------------------------------------
def test_parse_args_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RL_MODEL_PATH", "outputs/some_run/model/ppo_final")
    monkeypatch.setattr(sys, "argv", ["live_trading_rl.py"])
    args = live_trading_rl.parse_args()
    assert args.model == "outputs/some_run/model/ppo_final"
    assert args.symbol == "US100"
    assert args.secondary_symbol == "US500"
    assert args.once is False
    assert args.interval_minutes is None


def test_parse_args_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RL_MODEL_PATH", "from_env")
    monkeypatch.setattr(
        sys,
        "argv",
        ["live_trading_rl.py", "--model", "from_cli", "--once", "--symbol", "US500"],
    )
    args = live_trading_rl.parse_args()
    assert args.model == "from_cli"
    assert args.symbol == "US500"
    assert args.once is True


def test_parse_args_model_defaults_to_none_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no env/CLI model the entrypoint's guard must produce a clean exit."""
    monkeypatch.delenv("RL_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["live_trading_rl.py"])
    args = live_trading_rl.parse_args()
    assert SimpleNamespace(model=args.model).model is None
