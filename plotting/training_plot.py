"""
Plot Lightning CSVLogger training, validation, and test curves.

This script is tailored to the metrics emitted by fitness_module.py:
train/loss, val/loss, test/loss plus mse, mae, pearson, spearman, and r2.

Example:
    python plot_training_curves.py --csv C:/Users/paulb/Downloads/metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV = Path(r"C:\Users\paulb\PycharmProjects\METL_GLOBAL\data\metrics_lora_400k.csv")
DEFAULT_OUTPUT = Path(r"C:\Users\paulb\PycharmProjects\METL_GLOBAL\plotting\training_curves_400k_lora.png")


PANEL_SPECS = [
    (
        "Loss",
        [
            ("train/loss_step", "Train loss, step", "#4C78A8", 0.25, True),
            ("train/loss_epoch", "Train loss, epoch", "#1F4E79", 0.95, False),
            ("val/loss", "Validation loss", "#F58518", 0.95, False),
            ("test/loss", "Test loss", "#B279A2", 1.0, False),
        ],
    ),
    (
        "MAE",
        [
            ("train/mae", "Train MAE", "#4C78A8", 0.9, False),
            ("val/mae", "Validation MAE", "#F58518", 0.95, False),
            ("test/mae", "Test MAE", "#B279A2", 1.0, False),
        ],
    ),
    (
        "MSE",
        [
            ("train/mse", "Train MSE", "#4C78A8", 0.9, False),
            ("val/mse", "Validation MSE", "#F58518", 0.95, False),
            ("test/mse", "Test MSE", "#B279A2", 1.0, False),
        ],
    ),
    (
        "Correlation",
        [
            ("train/pearson", "Train Pearson", "#4C78A8", 0.9, False),
            ("val/pearson", "Validation Pearson", "#F58518", 0.95, False),
            ("test/pearson", "Test Pearson", "#B279A2", 1.0, False),
            ("train/spearman", "Train Spearman", "#72B7B2", 0.9, False),
            ("val/spearman", "Validation Spearman", "#E45756", 0.95, False),
            ("test/spearman", "Test Spearman", "#54A24B", 1.0, False),
        ],
    ),
    (
        "R2",
        [
            ("train/r2", "Train R2", "#4C78A8", 0.9, False),
            ("val/r2", "Validation R2", "#F58518", 0.95, False),
            ("test/r2", "Test R2", "#B279A2", 1.0, False),
        ],
    ),
    (
        "Learning rate",
        [
            ("lr-AdamW", "AdamW learning rate", "#2F4B7C", 1.0, False),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training/validation curves from a Lightning metrics.csv file."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to metrics.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path. Use .png, .pdf, or another matplotlib-supported format.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("step", "epoch"),
        default="step",
        help="Use global step or epoch on the x-axis. Step is clearer when validation runs mid-epoch.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="Rolling window for noisy step-level training loss. Set 1 to disable.",
    )
    parser.add_argument("--title", default="", help="Optional figure title")
    return parser.parse_args()


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "step" not in df.columns and "epoch" not in df.columns:
        raise ValueError("The CSV needs at least a 'step' or 'epoch' column.")

    return df


def metric_points(df: pd.DataFrame, metric: str, x_col: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame(columns=[x_col, metric])

    points = df[[x_col, metric]].dropna()
    if points.empty:
        return points

    # Lightning can emit multiple rows for the same x value. Keep them readable.
    points = points.groupby(x_col, as_index=False)[metric].mean()
    return points.sort_values(x_col)


def plot_series(
    ax: plt.Axes,
    points: pd.DataFrame,
    x_col: str,
    metric: str,
    label: str,
    color: str,
    alpha: float,
    smooth_step_metric: bool,
    smooth_window: int,
) -> bool:
    if points.empty:
        return False

    marker = "o" if len(points) <= 40 else None
    linewidth = 1.6

    if smooth_step_metric and smooth_window > 1 and len(points) >= smooth_window:
        ax.plot(
            points[x_col],
            points[metric],
            color=color,
            alpha=0.18,
            linewidth=0.8,
            label=f"{label} raw",
        )
        smoothed = points[metric].rolling(smooth_window, min_periods=1).mean()
        ax.plot(
            points[x_col],
            smoothed,
            color=color,
            alpha=1.0,
            linewidth=2.0,
            label=f"{label} ({smooth_window}-pt mean)",
        )
        return True

    ax.plot(
        points[x_col],
        points[metric],
        color=color,
        alpha=alpha,
        marker=marker,
        markersize=4,
        linewidth=linewidth,
        label=label,
    )
    return True


def build_figure(df: pd.DataFrame, x_col: str, smooth_window: int, title: str) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (panel_title, series_specs) in zip(axes_flat, PANEL_SPECS):
        plotted = False
        for metric, label, color, alpha, smooth_step_metric in series_specs:
            points = metric_points(df, metric, x_col)
            plotted |= plot_series(
                ax,
                points,
                x_col,
                metric,
                label,
                color,
                alpha,
                smooth_step_metric,
                smooth_window,
            )

        ax.set_title(panel_title, fontsize=12, weight="bold")
        ax.set_xlabel("Global step" if x_col == "step" else "Epoch")
        ax.tick_params(axis="both", labelsize=9)
        if plotted:
            ax.legend(fontsize=8, frameon=True)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    figure_title = title or "ESM-2 LoRA fitness model training curves"
    fig.suptitle(figure_title, fontsize=15, weight="bold")
    return fig


def main() -> None:
    args = parse_args()
    df = load_metrics(args.csv)

    x_col = args.x_axis
    if x_col not in df.columns:
        fallback = "epoch" if x_col == "step" else "step"
        if fallback not in df.columns:
            raise ValueError(f"Neither '{x_col}' nor '{fallback}' is present in the CSV.")
        x_col = fallback

    fig = build_figure(df, x_col=x_col, smooth_window=args.smooth, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved training curves to {args.out.resolve()}")


if __name__ == "__main__":
    main()
