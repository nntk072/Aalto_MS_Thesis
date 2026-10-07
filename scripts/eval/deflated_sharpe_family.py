# mypy: ignore-errors
"""Deflated Sharpe for the six finished seed-50 rescores.

Uses trade P&L as the return sample so Sharpe, skew, and kurtosis refer to
the same observations. Does not start another training run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_rl.evaluation.deflated_sharpe import (  # noqa: E402
    deflated_sharpe_probability,
    expected_maximum_sharpe,
    return_moments,
)

FAMILY = {
    "base_gru": "20260922_024235_116145_rl_train_seed50_gru",
    "base_tcn": "20260922_024235_194714_rl_train_seed50_tcn",
    "base_tf": "20260922_024235_109978_rl_train_seed50_transformer",
    "po3_tcn": "20260922_024144_516380_rl_train_seed50_tcn",
    "po3_gru": "20260922_024144_521964_rl_train_seed50_gru",
    "po3_tf": "20260922_024144_531686_rl_train_seed50_transformer",
}


def _trade_pnls(path: Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    closes = df[df["type"].astype(str).str.contains("close")]
    return pd.to_numeric(closes["pnl"], errors="coerce").dropna().to_numpy()


def main() -> None:
    root = ROOT / "outputs" / "rescore_sl_fill"
    rows = []
    for name in FAMILY:
        pnls = _trade_pnls(root / name / "testing" / "trades.csv")
        sr, skew, kurt, n = return_moments(pnls)
        checklist = json.loads((root / name / "checklist.json").read_text())
        rows.append(
            {
                "name": name,
                "source": FAMILY[name],
                "trade_sharpe": sr,
                "skew": skew,
                "kurtosis": kurt,
                "n_trades": n,
                "bar_sharpe": checklist["test"]["sharpe"],
                "test_pnl": checklist["test"]["pnl"],
            }
        )
    sharpes = [row["trade_sharpe"] for row in rows]
    sr_std = float(np.std(sharpes, ddof=1))
    sr0 = expected_maximum_sharpe(sr_std, n_trials=len(rows))
    best = max(rows, key=lambda row: row["trade_sharpe"])
    best_prob = deflated_sharpe_probability(
        best["trade_sharpe"],
        sr0=sr0,
        n_obs=best["n_trades"],
        skew=best["skew"],
        kurtosis=best["kurtosis"],
    )
    for row in rows:
        row["sr0"] = sr0
        row["deflated_sharpe_probability"] = deflated_sharpe_probability(
            row["trade_sharpe"],
            sr0=sr0,
            n_obs=row["n_trades"],
            skew=row["skew"],
            kurtosis=row["kurtosis"],
        )
    payload = {
        "family": "rescore_sl_fill test trades, six finished 20M policies",
        "n_trials": len(rows),
        "trade_sharpe_std": sr_std,
        "sr0": sr0,
        "selected": best["name"],
        "selected_deflated_sharpe_probability": best_prob,
        "runs": rows,
        "note": (
            "Sharpe here is mean(trade PnL) / sample std, not the annualised bar Sharpe. "
            "The in-flight 20260926 retrain is not included."
        ),
    }
    dest = root / "deflated_sharpe.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {k: payload[k] for k in ("selected", "selected_deflated_sharpe_probability", "sr0")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
