"""Training-progress plots for RL agents.

Reads a ``training_log.csv`` produced by :class:`ProgressLoggerCallback` and
generates five chart types, each as a white-theme static PNG + interactive HTML.

Charts
------
1. ``learning_curve``  — episode reward mean over timesteps
2. ``losses``          — policy-gradient loss + value loss (dual y-axis)
3. ``entropy_explvar`` — entropy loss + explained variance (dual y-axis)
4. ``kl_clip``         — approx KL + clip fraction (dual y-axis)
5. ``lr``              — learning-rate schedule

Usage::

    from quant_rl.eval.training_plots import save_training_plots
    save_training_plots("model/training_log.csv", out_dir="model/")
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

import pandas as pd

# White-theme colour palette
_C0 = "#1565c0"   # blue
_C1 = "#c62828"   # red
_C2 = "#2e7d32"   # green
_C3 = "#e65100"   # orange
_C4 = "#6a1b9a"   # purple

_WHITE_RC = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#222222",
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "text.color":       "#222222",
    "grid.color":       "#e6e6e6",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "lines.linewidth":  1.4,
    "font.size":        10,
}


def _apply_white() -> None:
    plt.rcParams.update(_WHITE_RC)


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _col(df: pd.DataFrame, *keywords: str) -> str | None:
    """Return first column whose name contains any of *keywords*."""
    for kw in keywords:
        for c in df.columns:
            if kw in c:
                return c
    return None


def _valid(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[["timestep", col]].dropna()


# ---------------------------------------------------------------------------
# Static PNG charts
# ---------------------------------------------------------------------------

def plot_learning_curve(df: pd.DataFrame, out_path: Path | str | None = None,
                        dpi: int = 150) -> plt.Figure:
    _apply_white()
    col = _col(df, "ep_rew_mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    if col and len(_valid(df, col)) > 1:
        v = _valid(df, col)
        ax.plot(v["timestep"], v[col], color=_C0, label="Ep. reward mean")
        ax.axhline(0, color="#aaa", linewidth=0.7, linestyle=":")
        ax.set_ylabel("Mean episode reward")
    else:
        ax.text(0.5, 0.5, "No episode reward data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Learning Curve — Episode Reward", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, Path(out_path), dpi)
    return fig


def plot_losses(df: pd.DataFrame, out_path: Path | str | None = None,
                dpi: int = 150) -> plt.Figure:
    _apply_white()
    pol = _col(df, "policy_gradient_loss", "policy_loss")
    val = _col(df, "value_loss")
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    if pol and len(_valid(df, pol)) > 1:
        v = _valid(df, pol)
        ax.plot(v["timestep"], v[pol], color=_C0, label="Policy gradient loss")
        ax.set_ylabel("Policy gradient loss", color=_C0)
        ax.tick_params(axis="y", labelcolor=_C0)
        plotted = True
    if val and len(_valid(df, val)) > 1:
        ax2 = ax.twinx()
        v = _valid(df, val)
        ax2.plot(v["timestep"], v[val], color=_C1, linestyle="--", label="Value loss")
        ax2.set_ylabel("Value loss", color=_C1)
        ax2.tick_params(axis="y", labelcolor=_C1)
        ax2.legend(fontsize=8, loc="upper right")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No loss data yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Policy & Value Losses", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, Path(out_path), dpi)
    return fig


def plot_entropy_explvar(df: pd.DataFrame, out_path: Path | str | None = None,
                         dpi: int = 150) -> plt.Figure:
    _apply_white()
    ent = _col(df, "entropy_loss", "entropy")
    ev  = _col(df, "explained_variance")
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    if ent and len(_valid(df, ent)) > 1:
        v = _valid(df, ent)
        ax.plot(v["timestep"], v[ent], color=_C2, label="Entropy loss")
        ax.set_ylabel("Entropy loss", color=_C2)
        ax.tick_params(axis="y", labelcolor=_C2)
        plotted = True
    if ev and len(_valid(df, ev)) > 1:
        ax2 = ax.twinx()
        v = _valid(df, ev)
        ax2.plot(v["timestep"], v[ev], color=_C3, linestyle="--", label="Explained variance")
        ax2.set_ylabel("Explained variance", color=_C3)
        ax2.tick_params(axis="y", labelcolor=_C3)
        ax2.legend(fontsize=8, loc="upper right")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No entropy/explained-variance data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Entropy & Explained Variance", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, Path(out_path), dpi)
    return fig


def plot_kl_clip(df: pd.DataFrame, out_path: Path | str | None = None,
                 dpi: int = 150) -> plt.Figure:
    _apply_white()
    kl   = _col(df, "approx_kl")
    clip = _col(df, "clip_fraction")
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    if kl and len(_valid(df, kl)) > 1:
        v = _valid(df, kl)
        ax.plot(v["timestep"], v[kl], color=_C4, label="Approx KL")
        ax.set_ylabel("Approx KL", color=_C4)
        ax.tick_params(axis="y", labelcolor=_C4)
        plotted = True
    if clip and len(_valid(df, clip)) > 1:
        ax2 = ax.twinx()
        v = _valid(df, clip)
        ax2.plot(v["timestep"], v[clip], color=_C1, linestyle="--", label="Clip fraction")
        ax2.set_ylabel("Clip fraction", color=_C1)
        ax2.tick_params(axis="y", labelcolor=_C1)
        ax2.legend(fontsize=8, loc="upper right")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No KL/clip data yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Approx KL & Clip Fraction", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, Path(out_path), dpi)
    return fig


def plot_learning_rate(df: pd.DataFrame, out_path: Path | str | None = None,
                       dpi: int = 150) -> plt.Figure:
    _apply_white()
    lr = _col(df, "learning_rate")
    fig, ax = plt.subplots(figsize=(10, 3))
    if lr and len(_valid(df, lr)) > 1:
        v = _valid(df, lr)
        ax.plot(v["timestep"], v[lr], color=_C3, label="Learning rate")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, 4))
    else:
        ax.text(0.5, 0.5, "No learning rate data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Learning rate")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, Path(out_path), dpi)
    return fig


# ---------------------------------------------------------------------------
# Interactive HTML charts
# ---------------------------------------------------------------------------

def _save_training_html(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    T = "plotly_white"

    def _write(fig: "go.Figure", name: str) -> None:
        fig.write_html(str(out_dir / name), include_plotlyjs="cdn")

    def _dual(col_a: str | None, name_a: str, col_b: str | None, name_b: str,
              title: str, fname: str) -> None:
        if not (col_a or col_b):
            return
        fig = make_subplots(specs=[[{"secondary_y": bool(col_b)}]])
        if col_a:
            v = _valid(df, col_a)
            fig.add_trace(go.Scatter(x=v["timestep"], y=v[col_a], name=name_a), secondary_y=False)
        if col_b:
            v = _valid(df, col_b)
            fig.add_trace(
                go.Scatter(x=v["timestep"], y=v[col_b], name=name_b, line=dict(dash="dash")),
                secondary_y=True,
            )
        fig.update_layout(template=T, title=title, xaxis_title="Timestep", height=400)
        _write(fig, fname)

    # Learning curve
    col = _col(df, "ep_rew_mean")
    if col and len(_valid(df, col)) > 1:
        v = _valid(df, col)
        fig = go.Figure(go.Scatter(x=v["timestep"], y=v[col], mode="lines",
                                   name="Ep. reward mean"))
        fig.update_layout(template=T, title="Learning Curve — Episode Reward",
                          xaxis_title="Timestep", yaxis_title="Mean episode reward", height=400)
        _write(fig, "learning_curve.html")

    _dual(_col(df, "policy_gradient_loss", "policy_loss"), "Policy gradient loss",
          _col(df, "value_loss"), "Value loss",
          "Policy & Value Losses", "losses.html")

    _dual(_col(df, "entropy_loss", "entropy"), "Entropy loss",
          _col(df, "explained_variance"), "Explained variance",
          "Entropy & Explained Variance", "entropy_explvar.html")

    _dual(_col(df, "approx_kl"), "Approx KL",
          _col(df, "clip_fraction"), "Clip fraction",
          "Approx KL & Clip Fraction", "kl_clip.html")

    lr = _col(df, "learning_rate")
    if lr and len(_valid(df, lr)) > 1:
        v = _valid(df, lr)
        fig = go.Figure(go.Scatter(x=v["timestep"], y=v[lr], name="Learning rate"))
        fig.update_layout(template=T, title="Learning Rate Schedule",
                          xaxis_title="Timestep", yaxis_title="Learning rate", height=350)
        _write(fig, "lr.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def save_training_plots(
    log_path: Path | str,
    out_dir: Path | str,
    dpi: int = 150,
    save_html: bool = True,
) -> None:
    """Generate all training-progress charts from *log_path* CSV into *out_dir*."""
    log_path = Path(log_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists() or log_path.stat().st_size == 0:
        return

    df = pd.read_csv(log_path)

    plot_learning_curve(df, out_path=out_dir / "learning_curve.png", dpi=dpi)
    plot_losses(df, out_path=out_dir / "losses.png", dpi=dpi)
    plot_entropy_explvar(df, out_path=out_dir / "entropy_explvar.png", dpi=dpi)
    plot_kl_clip(df, out_path=out_dir / "kl_clip.png", dpi=dpi)
    plot_learning_rate(df, out_path=out_dir / "lr.png", dpi=dpi)

    if save_html:
        _save_training_html(df, out_dir)
