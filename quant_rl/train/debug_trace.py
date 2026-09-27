"""Flight recorder for a PPO training run.

One JSON line per update, plus an incident when training raises, a number is
non-finite, or a check returns a reason. Extra collectors and checks are
ordinary functions, so a later bug does not need a new log format.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from quant_rl.envs.reward import PnLReward

Check = Callable[[Sequence[Mapping[str, Any]]], str]
Collector = Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]


class DebugStepError(Exception):
    """A step failure carrying the action, reward parts, and position."""

    def __init__(
        self,
        reason: str,
        snapshot: Mapping[str, Any],
        original: BaseException | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.snapshot = dict(snapshot)
        self.original = original


def new_debug_window() -> dict[str, Any]:
    """Empty per-env counters for one rollout."""
    return {
        "n": 0,
        "parts": {},
        "opens": 0,
        "shape_on_open": 0,
        "shape_on_hold": 0,
        "breach_events": 0,
        "entries": 0,
        "risk_sum": 0.0,
        "soft_daily_nz": 0,
        "soft_daily_repeat": 0,
        "soft_year_nz": 0,
        "soft_year_repeat": 0,
        "prev_soft_daily": False,
        "prev_soft_year": False,
        "rewards": [],
        "closes": 0,
        "structure_sl": 0,
        "structure_tp": 0,
    }


def note_debug_step(
    window: dict[str, Any],
    *,
    reward: float,
    reward_parts: Mapping[str, float],
    opened: bool,
    entered: bool,
    risk_frac: float,
    close_reason: str,
    action: Any,
    position: bool,
    position_direction: int,
) -> None:
    """Update counters. A non-finite reward raises :class:`DebugStepError`."""
    if not math.isfinite(float(reward)):
        raise DebugStepError(
            "non-finite reward",
            {
                "action": _jsonable(action),
                "reward": None if not math.isfinite(float(reward)) else float(reward),
                "reward_parts": _jsonable(dict(reward_parts)),
                "position": bool(position),
                "position_direction": int(position_direction),
            },
        )
    window["n"] += 1
    for key, value in reward_parts.items():
        window["parts"][str(key)] = float(window["parts"].get(str(key), 0.0)) + float(value)
    shaped = float(reward_parts.get("strategy", 0.0))
    if opened:
        window["opens"] += 1
        if shaped != 0.0:
            window["shape_on_open"] += 1
    elif shaped != 0.0:
        window["shape_on_hold"] += 1
    if float(reward_parts.get("breach", 0.0)) != 0.0:
        window["breach_events"] += 1
    if entered:
        window["entries"] += 1
        window["risk_sum"] += float(risk_frac)
    daily = float(reward_parts.get("soft_daily", 0.0)) != 0.0
    year = float(reward_parts.get("soft_year", 0.0)) != 0.0
    if daily:
        window["soft_daily_nz"] += 1
        if window["prev_soft_daily"]:
            window["soft_daily_repeat"] += 1
    if year:
        window["soft_year_nz"] += 1
        if window["prev_soft_year"]:
            window["soft_year_repeat"] += 1
    window["prev_soft_daily"] = daily
    window["prev_soft_year"] = year
    window["rewards"].append(float(reward))
    if close_reason:
        window["closes"] += 1
        if close_reason == "structure_sl":
            window["structure_sl"] += 1
        elif close_reason == "structure_tp":
            window["structure_tp"] += 1


def summarize_window(window: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one env window into a JSON-ready summary."""
    n = int(window["n"])
    rewards = [float(value) for value in window["rewards"]]
    nonzero = float(np.mean(np.abs(rewards) > 1e-8)) if rewards else 0.0
    return {
        "n": n,
        "parts": {key: float(value) for key, value in window["parts"].items()},
        "opens": int(window["opens"]),
        "shape_on_open": int(window["shape_on_open"]),
        "shape_on_hold": int(window["shape_on_hold"]),
        "breach_events": int(window["breach_events"]),
        "entries": int(window["entries"]),
        "risk_sum": float(window["risk_sum"]),
        "soft_daily_nz": int(window["soft_daily_nz"]),
        "soft_daily_repeat": int(window["soft_daily_repeat"]),
        "soft_year_nz": int(window["soft_year_nz"]),
        "soft_year_repeat": int(window["soft_year_repeat"]),
        "spike_share": _spike_share(rewards),
        "nonzero": nonzero,
        "closes": int(window["closes"]),
        "structure_sl": int(window["structure_sl"]),
        "structure_tp": int(window["structure_tp"]),
    }


def merge_env_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum per-env windows and scale reward parts to per 1k steps."""
    usable = [row for row in rows if row]
    n = sum(int(row.get("n", 0)) for row in usable)
    parts: dict[str, float] = {}
    for row in usable:
        for key, value in dict(row.get("parts", {})).items():
            parts[key] = parts.get(key, 0.0) + float(value)
    scale = 1000.0 / max(1, n)
    entries = sum(int(row.get("entries", 0)) for row in usable)
    risk = sum(float(row.get("risk_sum", 0.0)) for row in usable)
    closes = sum(int(row.get("closes", 0)) for row in usable)
    spike_num = sum(float(row.get("spike_share", 0.0)) * int(row.get("n", 0)) for row in usable)
    nonzero_num = sum(float(row.get("nonzero", 0.0)) * int(row.get("n", 0)) for row in usable)
    return {
        "reward_steps": n,
        "reward_per_1k": {key: float(value) * scale for key, value in parts.items()},
        "breach_events": sum(int(row.get("breach_events", 0)) for row in usable),
        "opens": sum(int(row.get("opens", 0)) for row in usable),
        "shape_on_open": sum(int(row.get("shape_on_open", 0)) for row in usable),
        "shape_on_hold": sum(int(row.get("shape_on_hold", 0)) for row in usable),
        "entries": entries,
        "mean_entry_risk": (risk / entries) if entries else 0.0,
        "soft_daily_nz": sum(int(row.get("soft_daily_nz", 0)) for row in usable),
        "soft_daily_repeat": sum(int(row.get("soft_daily_repeat", 0)) for row in usable),
        "soft_year_nz": sum(int(row.get("soft_year_nz", 0)) for row in usable),
        "soft_year_repeat": sum(int(row.get("soft_year_repeat", 0)) for row in usable),
        "spike_share": (spike_num / n) if n else 0.0,
        "nonzero_frac": (nonzero_num / n) if n else 0.0,
        "structure_sl_share": (
            sum(int(row.get("structure_sl", 0)) for row in usable) / closes if closes else 0.0
        ),
        "structure_tp_share": (
            sum(int(row.get("structure_tp", 0)) for row in usable) / closes if closes else 0.0
        ),
    }


def wiring_from_env(env: Any) -> dict[str, Any]:
    """Reward and risk settings the train env was built with."""
    reward = getattr(env, "reward_fn", None)
    base = getattr(reward, "_dsr_fn", None)
    if isinstance(reward, PnLReward):
        base = reward
    return {
        "reward_mode": str(getattr(env, "reward_mode", "")),
        "episodic": bool(getattr(env, "episodic", False)),
        "breach_penalty": float(
            getattr(base, "breach_penalty", getattr(reward, "breach_penalty", 0.0)) or 0.0
        ),
        "eval_breach_penalty": -10.0,
        "composite_dsr_weight": float(getattr(reward, "dsr_weight", 0.0) or 0.0),
        "pnl_dsr_weight": float(getattr(base, "dsr_weight", 0.0) or 0.0),
        "strategy_weight": float(getattr(reward, "strategy_weight", 0.0) or 0.0),
        "risk_floor": float(getattr(env, "risk_floor", 0.0) or 0.0),
        "risk_frac_range": [float(value) for value in getattr(env, "risk_frac_range", ())],
    }


def collect_rollout(model: Any) -> dict[str, Any]:
    """Return, value, and account correlations from the current rollout buffer."""
    empty = {
        "timesteps": int(getattr(model, "num_timesteps", 0) or 0),
        "ret_mean": 0.0,
        "ret_std": 0.0,
        "val_mean": 0.0,
        "val_std": 0.0,
        "explained_variance": 0.0,
        "mc_mean": 0.0,
        "mc_std": 0.0,
        "corr_val_open_pnl": 0.0,
        "corr_boot_open_pnl": 0.0,
        "corr_boot_unrealised_r": 0.0,
    }
    buffer = getattr(model, "rollout_buffer", None)
    returns = getattr(buffer, "returns", None) if buffer is not None else None
    values = getattr(buffer, "values", None) if buffer is not None else None
    if returns is None or values is None:
        return empty
    ret_all = np.asarray(returns, dtype=float).reshape(-1)
    val_all = np.asarray(values, dtype=float).reshape(-1)
    count = min(int(ret_all.shape[0]), int(val_all.shape[0]))
    ret = np.array(ret_all[:count], dtype=float, copy=True)
    val = np.array(val_all[:count], dtype=float, copy=True)
    if ret.size == 0:
        return empty
    resid = ret - val
    var_ret = float(np.var(ret))
    out = dict(empty)
    out.update(
        {
            "ret_mean": float(np.mean(ret)),
            "ret_std": float(np.std(ret)),
            "val_mean": float(np.mean(val)),
            "val_std": float(np.std(val)),
            "explained_variance": 0.0 if var_ret <= 1e-12 else 1.0 - float(np.var(resid)) / var_ret,
        }
    )
    rewards = getattr(buffer, "rewards", None)
    if rewards is not None:
        rew = np.asarray(rewards, dtype=float)
        if rew.ndim == 1:
            rew = rew.reshape(-1, 1)
        mc = _monte_carlo(
            rew, getattr(buffer, "episode_starts", None), float(getattr(model, "gamma", 0.0))
        )
        out["mc_mean"] = float(np.mean(mc))
        out["mc_std"] = float(np.std(mc))
    obs = getattr(buffer, "observations", None)
    account = obs.get("account") if isinstance(obs, dict) else None
    if account is not None:
        acc = np.asarray(account, dtype=float)
        if acc.shape[-1] > 3:
            out["corr_val_open_pnl"] = _corr(val, acc[..., 2])
            out["corr_boot_open_pnl"] = _corr(ret, acc[..., 2])
            out["corr_boot_unrealised_r"] = _corr(ret, acc[..., 3])
    return out


def policy_std(model: Any) -> float | None:
    """Mean Gaussian std after the post-update clamp, or None when absent."""
    log_std = getattr(getattr(model, "policy", None), "log_std", None)
    if log_std is None:
        return None
    return float(np.exp(log_std.detach().cpu().numpy()).mean())


def account_weights(model: Any) -> list[float]:
    """The six linear account-value weights, or an empty list."""
    weight = getattr(getattr(model, "policy", None), "account_value_weight", None)
    if weight is None:
        return []
    return [float(value) for value in weight.detach().cpu().reshape(-1).tolist()]


def builtin_checks(thresholds: Mapping[str, Any] | None = None) -> list[Check]:
    """Checks for the failures already seen, plus a wiring check."""
    cfg = dict(thresholds or {})
    ev_floor = float(cfg.get("ev_floor", 0.2))
    ev_windows = int(cfg.get("ev_windows", 5))
    std_min = float(cfg.get("std_min", 0.497))
    std_max = float(cfg.get("std_max", 1.0))
    part_vs_pnl = float(cfg.get("part_vs_pnl", 10.0))
    corr_floor = float(cfg.get("corr_floor", 0.3))
    corr_windows = int(cfg.get("corr_windows", 3))

    def explained_variance(records: Sequence[Mapping[str, Any]]) -> str:
        body = list(records)[1:]
        if len(body) < ev_windows:
            return ""
        window = body[-ev_windows:]
        if all(float(row.get("explained_variance", 0.0)) <= ev_floor for row in window):
            return "explained variance"
        return ""

    def std_band(records: Sequence[Mapping[str, Any]]) -> str:
        std = records[-1].get("std") if records else None
        if std is None:
            return ""
        value = float(std)
        if not math.isfinite(value) or value < std_min or value > std_max:
            return "policy std"
        return ""

    def reward_dominates(records: Sequence[Mapping[str, Any]]) -> str:
        parts = dict(records[-1].get("reward_per_1k", {})) if records else {}
        pnl = abs(float(parts.get("pnl", 0.0)))
        for key, value in parts.items():
            if key == "pnl":
                continue
            magnitude = abs(float(value))
            if magnitude > 1e-6 and magnitude > part_vs_pnl * max(pnl, 1e-12):
                return f"reward part {key}"
        return ""

    def shaping(records: Sequence[Mapping[str, Any]]) -> str:
        row = records[-1] if records else {}
        if int(row.get("shape_on_hold", 0)) > 0:
            return "shaping on hold"
        if int(row.get("soft_daily_repeat", 0)) > 0 or int(row.get("soft_year_repeat", 0)) > 0:
            return "soft band repeat"
        return ""

    def open_pnl(records: Sequence[Mapping[str, Any]]) -> str:
        body = list(records)[1:]
        if len(body) < corr_windows:
            return ""
        window = body[-corr_windows:]
        weak_value = all(float(row.get("corr_val_open_pnl", 0.0)) < corr_floor for row in window)
        strong_return = all(
            float(row.get("corr_boot_open_pnl", 0.0)) > corr_floor for row in window
        )
        if weak_value and strong_return:
            return "value ignores open pnl"
        return ""

    def wiring(records: Sequence[Mapping[str, Any]]) -> str:
        row = records[-1] if records else {}
        if "breach_penalty" not in row and "pnl_dsr_weight" not in row:
            return ""
        if float(row.get("breach_penalty", 0.0)) != 0.0:
            return "training breach penalty"
        if float(row.get("pnl_dsr_weight", 0.0)) != 0.0:
            return "pnl dsr weight"
        return ""

    return [explained_variance, std_band, reward_dominates, shaping, open_pnl, wiring]


class FlightRecorder:
    """Append update lines and incidents under a run directory."""

    def __init__(
        self,
        directory: Path | str,
        *,
        ring_size: int = 5,
        checks: Sequence[Check] | None = None,
        thresholds: Mapping[str, Any] | None = None,
        extra_checks: Sequence[Check] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ring: deque[dict[str, Any]] = deque(maxlen=max(1, int(ring_size)))
        self.checks: list[Check] = (
            list(checks) if checks is not None else builtin_checks(thresholds)
        )
        if extra_checks:
            self.checks.extend(extra_checks)
        self._last_tb = ""
        self.trace_path = self.directory / "debug.jsonl"
        self.incident_path = self.directory / "debug_incidents.jsonl"
        self.wiring_path = self.directory / "debug_wiring.json"

    def write_wiring(self, payload: Mapping[str, Any]) -> None:
        snap = _jsonable(dict(payload))
        self.wiring_path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
        reason = ""
        for check in self.checks:
            reason = check([snap])
            if reason:
                break
        if reason:
            self._write_incident(reason, snap, "")

    def record(self, snapshot: Mapping[str, Any]) -> None:
        """Append one update. Write an incident when the snapshot is unhealthy."""
        reason = nonfinite_reason(snapshot)
        snap = _jsonable(dict(snapshot))
        _append_jsonl(self.trace_path, snap)
        if not reason:
            window = list(self.ring) + [snap]
            for check in self.checks:
                reason = check(window)
                if reason:
                    break
        if reason:
            self._write_incident(reason, snap, "")
        self.ring.append(snap)

    def capture(self, reason: str, snapshot: Mapping[str, Any] | None = None) -> None:
        """Record an exception. A second capture of the same traceback is skipped."""
        current = sys.exc_info()[1]
        if current is not None and getattr(current, "_debug_captured", False):
            return
        if current is not None:
            try:
                current._debug_captured = True  # type: ignore[attr-defined]
            except Exception:
                pass
        text = traceback.format_exc()
        if text and text != "NoneType: None\n" and text == self._last_tb:
            return
        if text and text != "NoneType: None\n":
            self._last_tb = text
        snap = _jsonable(dict(snapshot or {}))
        self._write_incident(reason, snap, "" if text == "NoneType: None\n" else text)

    def _write_incident(self, reason: str, snapshot: Mapping[str, Any], tb: str) -> None:
        _append_jsonl(
            self.incident_path,
            {
                "reason": reason,
                "traceback": tb,
                "snapshot": _jsonable(dict(snapshot)),
                "history": [dict(row) for row in self.ring],
            },
        )


def nonfinite_reason(snapshot: Mapping[str, Any]) -> str:
    """Name the first non-finite number, or return an empty string."""
    for key, value in _walk(snapshot):
        if isinstance(value, float) and not math.isfinite(value):
            return f"non-finite {key}"
    return ""


def install_flight_recorder(
    model: Any,
    directory: Path | str,
    *,
    thresholds: Mapping[str, Any] | None = None,
    extra_checks: Sequence[Check] | None = None,
    extra_collectors: Sequence[Collector] | None = None,
) -> FlightRecorder:
    """Wrap ``learn`` and ``train``, and enable env counters when the env has them."""
    cfg = dict(thresholds or {})
    recorder = FlightRecorder(
        directory,
        ring_size=int(cfg.get("ring_size", 5)),
        thresholds=cfg,
        extra_checks=extra_checks,
    )
    collectors = list(extra_collectors or [])
    vec = model.get_env()
    if hasattr(vec, "env_method"):
        try:
            vec.env_method("enable_debug_trace")
            wiring_rows = vec.env_method("debug_wiring")
        except (AttributeError, TypeError):
            wiring_rows = []
        if wiring_rows:
            recorder.write_wiring(wiring_rows[0])

    train = model.train

    def _train() -> None:
        rollout = collect_rollout(model)
        try:
            train()
        except DebugStepError as exc:
            recorder.capture(exc.reason, exc.snapshot)
            if exc.original is not None:
                raise exc.original
            raise
        except Exception:
            recorder.capture("train exception", rollout)
            raise
        snapshot = dict(rollout)
        snapshot["std"] = policy_std(model)
        snapshot["account_value_weight"] = account_weights(model)
        snapshot.update(_flush_env_stats(model))
        for collector in collectors:
            snapshot.update(dict(collector(model, snapshot)))
        recorder.record(snapshot)

    model.train = _train
    learn = model.learn

    def _learn(*args: Any, **kwargs: Any) -> Any:
        try:
            return learn(*args, **kwargs)
        except DebugStepError as exc:
            recorder.capture(exc.reason, exc.snapshot)
            if exc.original is not None:
                raise exc.original
            raise
        except Exception:
            recorder.capture("learn exception", {})
            raise

    model.learn = _learn
    return recorder


def _flush_env_stats(model: Any) -> dict[str, Any]:
    vec = model.get_env()
    if not hasattr(vec, "env_method"):
        return merge_env_stats(())
    try:
        rows = vec.env_method("debug_flush")
    except (AttributeError, TypeError):
        return merge_env_stats(())
    return merge_env_stats(rows)


def _monte_carlo(
    rewards: np.ndarray[Any, np.dtype[np.floating[Any]]],
    starts: Any,
    gamma: float,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    start_arr = np.asarray(starts, dtype=float) if starts is not None else np.zeros_like(rewards)
    if start_arr.ndim == 1:
        start_arr = start_arr.reshape(-1, 1)
    running = np.zeros(rewards.shape[1], dtype=float)
    out = np.zeros_like(rewards)
    for t in range(rewards.shape[0] - 1, -1, -1):
        cont = 1.0 - start_arr[t + 1] if t + 1 < rewards.shape[0] else 0.0
        running = rewards[t] + gamma * running * cont
        out[t] = running
    return out


def _corr(
    left: np.ndarray[Any, np.dtype[np.floating[Any]]],
    right: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> float:
    x_all = np.asarray(left, dtype=float).reshape(-1)
    y_all = np.asarray(right, dtype=float).reshape(-1)
    n = min(int(x_all.shape[0]), int(y_all.shape[0]))
    if n < 2:
        return 0.0
    x = np.array(x_all[:n], dtype=float, copy=True)
    y = np.array(y_all[:n], dtype=float, copy=True)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spike_share(rewards: Sequence[float]) -> float:
    if not rewards:
        return 0.0
    values = np.asarray(rewards, dtype=float)
    centered = values - float(np.mean(values))
    total = float(np.dot(centered, centered))
    if total <= 0.0:
        return 0.0
    k = max(1, int(0.01 * values.size))
    top = np.argpartition(np.abs(values), -k)[-k:]
    return float(np.dot(centered[top], centered[top]) / total)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(payload))) + "\n")
        handle.flush()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _walk(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_walk(item, name))
        return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_walk(item, f"{prefix}[{index}]"))
        return found
    found.append((prefix or "value", value))
    return found
