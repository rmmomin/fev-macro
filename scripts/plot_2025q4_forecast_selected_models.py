#!/usr/bin/env python3
"""Plot a column chart of 2025Q4 real GDP q/q SAAR forecasts for selected models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MODELS = ["rw_drift_log", "auto_ets", "drift", "theta", "chronos2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a column chart for selected models' 2025Q4 q/q SAAR forecasts."
    )
    parser.add_argument(
        "--input_csv",
        default="results/realtime_latest_v2026m1_2025Q4_forecasts.csv",
        help="Input forecast CSV from scripts/run_latest_vintage_forecast.py",
    )
    parser.add_argument(
        "--target_quarter",
        default="2025Q4",
        help="Target quarter to plot.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to include in the plot.",
    )
    parser.add_argument(
        "--output_png",
        default="results/forecast_2025Q4_qoq_saar_selected_models.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"model", "target_quarter", "g_hat_saar"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    model_order = list(args.models)
    plot_df = df.loc[
        (df["target_quarter"].astype(str) == str(args.target_quarter))
        & (df["model"].astype(str).isin(model_order))
    ].copy()
    if plot_df.empty:
        raise ValueError("No matching rows found for requested target quarter/models.")

    plot_df = plot_df.set_index("model").reindex(model_order).reset_index()
    if plot_df["g_hat_saar"].isna().any():
        missing_models = plot_df.loc[plot_df["g_hat_saar"].isna(), "model"].tolist()
        raise ValueError(f"Missing forecast values for models: {missing_models}")

    plot_df = plot_df.sort_values("g_hat_saar", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(plot_df["model"], plot_df["g_hat_saar"], color="#2f6f9f")

    for bar, val in zip(bars, plot_df["g_hat_saar"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.08,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("2025Q4 Real GDP Forecasts (q/q SAAR)")
    ax.set_ylabel("q/q SAAR (%)")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    print(f"Saved plot: {output_png}")
    print(plot_df[["model", "g_hat_saar"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
