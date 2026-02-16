#!/usr/bin/env python3
"""Plot a column chart of 2025Q4 real GDP q/q SAAR forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_EVAL_FULL_MODELS = [
    "naive_last",
    "mean",
    "drift",
    "seasonal_naive",
    "random_normal",
    "random_uniform",
    "random_permutation",
    "random_forest",
    "xgboost",
    "local_trend_ssm",
    "bvar_minnesota_8",
    "bvar_minnesota_20",
    "factor_pca_qd",
    "mixed_freq_dfm_md",
    "ensemble_avg_top3",
    "ensemble_weighted_top5",
    "auto_arima",
    "chronos2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a column chart for model q/q SAAR forecasts for a target quarter."
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
        default=RUN_EVAL_FULL_MODELS,
        help="Model names to include in the plot.",
    )
    parser.add_argument(
        "--output_png",
        default="results/forecast_2025Q4_qoq_saar_run_eval_models.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--allow_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow missing/NaN forecast values. Missing models are plotted at 0 with 'NA' labels. "
            "Use --no-allow-missing to fail on missing values."
        ),
    )
    parser.add_argument(
        "--footnote",
        default="",
        help="Optional footnote text rendered at the bottom of the chart.",
    )
    parser.add_argument(
        "--footnote_fontsize",
        type=float,
        default=9.0,
        help="Font size for the optional footnote.",
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
    if plot_df["g_hat_saar"].isna().any() and not args.allow_missing:
        missing_models = plot_df.loc[plot_df["g_hat_saar"].isna(), "model"].tolist()
        raise ValueError(f"Missing forecast values for models: {missing_models}")

    plot_df["is_missing"] = plot_df["g_hat_saar"].isna()
    plot_df["g_hat_saar_plot"] = plot_df["g_hat_saar"].fillna(0.0)

    plot_df = plot_df.sort_values("g_hat_saar_plot", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#b0b0b0" if miss else "#2f6f9f" for miss in plot_df["is_missing"]]
    bars = ax.bar(plot_df["model"], plot_df["g_hat_saar_plot"], color=bar_colors)

    for bar, val, is_missing in zip(bars, plot_df["g_hat_saar"], plot_df["is_missing"]):
        label = "NA" if is_missing else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("2025Q4 Real GDP Forecasts (q/q SAAR)")
    ax.set_ylabel("q/q SAAR (%)")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    if str(args.footnote).strip():
        fig.subplots_adjust(bottom=0.2)
        fig.text(
            0.5,
            0.035,
            str(args.footnote).strip(),
            ha="center",
            va="bottom",
            fontsize=float(args.footnote_fontsize),
            color="#444444",
            wrap=True,
        )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    print(f"Saved plot: {output_png}")
    print(plot_df[["model", "g_hat_saar"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
