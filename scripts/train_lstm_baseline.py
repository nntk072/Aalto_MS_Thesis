"""Train the LSTM sweep-direction classifier (supervised baseline).

Example:
    python scripts/train_lstm_baseline.py \
        --bars-csv data/us100_2025.csv --features-csv data/us100_feat.csv \
        --epochs 20 --out models/lstm_sweep.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_rl.baselines import LSTMSweepClassifier, build_sweep_dataset  # noqa: E402


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
    return parser.parse_args()


def _load_csv(path: str, index_col: str | None) -> pd.DataFrame:
    """Load a CSV with a datetime index."""
    df = pd.read_csv(path, index_col=index_col or 0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def main() -> None:
    """Train, report accuracy, and save the model."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bars = _load_csv(args.bars_csv, getattr(args, "index_col", None))
    if args.features_csv:
        features = _load_csv(args.features_csv, getattr(args, "index_col", None))
    else:
        features = bars.select_dtypes(include=[np.number])

    x, y = build_sweep_dataset(bars, features, window=args.window, horizon=args.horizon)
    n_train = int(0.8 * len(x))
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]
    print(f"samples: train={len(x_train)} val={len(x_val)}")

    model = LSTMSweepClassifier(x.shape[-1], hidden_size=args.hidden_size)
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": x.shape[-1],
            "hidden_size": args.hidden_size,
            "window": args.window,
        },
        out_path,
    )
    print(f"saved model to {out_path}")


if __name__ == "__main__":
    main()
