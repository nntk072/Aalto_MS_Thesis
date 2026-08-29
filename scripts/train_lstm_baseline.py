"""Train the LSTM sweep-direction classifier (supervised baseline).

Training samples come from the in-sample slice (≤ --train-end) and both the
validation accuracy and the trading evaluation use the held-out slice
(≥ --test-start).  Because the sweep dataset is built independently per
slice, the boundary carries an implicit purge gap of at least
``window + horizon`` bars — no window/label straddle.

After training, the classifier is wrapped in ``LSTMStrategy`` and scored
through the shared evaluation pipeline (``run_episode`` + ``TradingEnv``)
on the held-out slice, so its Sharpe/drawdown/PnL are directly comparable
to the RL agent and rule-based baselines.

Example:
    python scripts/train_lstm_baseline.py \
        --bars-csv data/us100_2025.csv --features-csv data/us100_feat.csv \
        --epochs 20 --out models/lstm_sweep.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_rl.baselines import (  # noqa: E402
    LSTMStrategy,
    LSTMSweepClassifier,
    build_sweep_dataset,
)
from quant_rl.data.split import split_train_test  # noqa: E402
from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.evaluation import build_run_report, run_episode  # noqa: E402

OBS_WINDOW = 60  # must match TradingEnv's default obs_window


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-csv", required=True, help="OHLCV CSV with DatetimeIndex")
    parser.add_argument("--features-csv", help="Feature CSV; built from bars if omitted")
    parser.add_argument("--index-col", default=argparse.SUPPRESS, help="Index column name")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/lstm_sweep.pt")
    parser.add_argument("--train-end", default="2025-12-31", help="Last in-sample date")
    parser.add_argument("--test-start", default="2026-01-01", help="First held-out date")
    parser.add_argument("--obs-window", type=int, default=OBS_WINDOW, help="Env observation window")
    return parser.parse_args()


def _load_csv(path: str, index_col: str | None) -> pd.DataFrame:
    """Load a CSV with a datetime index."""
    df = pd.read_csv(path, index_col=index_col or 0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def main() -> None:
    """Train on the in-sample slice, then report accuracy + trading metrics."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bars = _load_csv(args.bars_csv, getattr(args, "index_col", None))
    if args.features_csv:
        features = _load_csv(args.features_csv, getattr(args, "index_col", None))
    else:
        features = bars.select_dtypes(include=[np.number])

    train_bars, test_bars, train_features, test_features = split_train_test(
        bars, features, args.train_end, args.test_start
    )
    if train_bars.empty or test_bars.empty:
        raise SystemExit(
            f"empty split with --train-end {args.train_end} / --test-start {args.test_start}: "
            f"train={len(train_bars)} bars, test={len(test_bars)} bars"
        )

    x_train, y_train = build_sweep_dataset(
        train_bars, train_features, window=args.window, horizon=args.horizon
    )
    x_val, y_val = build_sweep_dataset(
        test_bars, test_features, window=args.window, horizon=args.horizon
    )
    print(f"sweep samples: train={len(x_train)} held-out={len(x_val)}")

    model = LSTMSweepClassifier(x_train.shape[-1], hidden_size=args.hidden_size)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.0, 1.0])  # classes: -1, 0, +1
    )

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        total = 0.0
        for start in range(0, len(x_train), args.batch_size):
            batch = perm[start : start + args.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch] + 1)  # map to {0,1,2}
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)

        with torch.no_grad():
            val_preds = model(x_val).argmax(dim=-1) - 1
            val_acc = float((val_preds == y_val).float().mean())
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={total / len(x_train):.4f} val_acc={val_acc:.3f}"
        )

    # Trading evaluation of the trained classifier on the held-out slice,
    # through the same pipeline as the RL agent and rule-based baselines.
    strategy = LSTMStrategy(model, test_features, window=args.window)
    strategy.fast_forward(args.obs_window)
    env = TradingEnv(
        bars=test_bars,
        features=test_features,
        continuous_actions=True,
        # Eval mode: a breach blocks trading for the rest of the session
        # instead of truncating the episode (mirrors run_backtest).
        episodic=False,
    )
    metrics = run_episode(
        env,
        # LSTMStrategy emits floats on the continuous [-1, 1] contract;
        # TradingEnv's continuous decoder expects an ndarray action.
        action_fn=lambda obs: np.array([strategy.act(obs)], dtype=np.float32),
    )
    report = {
        "run_name": "lstm_sweep_baseline",
        "val_acc": val_acc,
        "split": {
            "train_end": args.train_end,
            "test_start": args.test_start,
            "train_bars": int(len(train_bars)),
            "test_bars": int(len(test_bars)),
        },
        "out_of_sample": build_run_report(metrics, env.trade_log),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": x_train.shape[-1],
            "hidden_size": args.hidden_size,
            "window": args.window,
        },
        out_path,
    )
    metrics_path = out_path.parent / f"{out_path.stem}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    print(f"saved model to {out_path}")
    print(f"saved trading metrics to {metrics_path}")


if __name__ == "__main__":
    main()
