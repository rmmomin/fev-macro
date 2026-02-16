#!/usr/bin/env python3
# DEV ONLY: non-core utility script retained for development workflows.
"""Plot clustered columns for q/q growth of first/second/third GDP releases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a clustered column chart of q/q growth rates for first, second, "
            "and third GDPC1 estimates."
        )
    )
    parser.add_argument(
        "--input_csv",
        default="data/panels/gdpc1_releases_first_second_third.csv",
        help="Path to release dataset CSV.",
    )
    parser.add_argument(
        "--start_quarter",
        default="2018Q1",
        help="First quarter to include (e.g., 2018Q1).",
    )
    parser.add_argument(
        "--output_png",
        default="results/gdpc1_release_qoq_growth_clustered_2018q1.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--metric",
        choices=["qoq", "qoq_saar"],
        default="qoq",
        help="Growth metric: qoq = 100*(x_t/x_{t-1}-1), qoq_saar = 100*((x_t/x_{t-1})^4-1).",
    )
    parser.add_argument(
        "--auto_build",
        action="store_true",
        help="Build the input dataset with scripts/build_gdp_releases.py if missing.",
    )
    return parser.parse_args()


def maybe_build_input(path: Path) -> None:
    if path.exists():
        return
    cmd = [
        sys.executable,
        "scripts/build_gdp_releases.py",
        "--series",
        "GDPC1",
        "--output_csv",
        str(path),
    ]
    subprocess.run(cmd, check=True)


def _quarter_label(ts: pd.Timestamp) -> str:
    q = ((ts.month - 1) // 3) + 1
    return f"{ts.year}Q{q}"


def _growth(series: pd.Series, metric: str) -> pd.Series:
    ratio = series / series.shift(1)
    if metric == "qoq":
        return (ratio - 1.0) * 100.0
    return (ratio.pow(4) - 1.0) * 100.0


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()

    if args.auto_build:
        input_csv.parent.mkdir(parents=True, exist_ok=True)
        maybe_build_input(path=input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}. "
            "Run scripts/build_gdp_releases.py first or pass --auto_build."
        )

    df = pd.read_csv(input_csv)
    if "observation_date" not in df.columns:
        raise ValueError("Input CSV missing required column: observation_date")

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df = df.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)

    if args.metric == "qoq":
        realtime_cols = {
            "first": "qoq_growth_realtime_first_pct",
            "second": "qoq_growth_realtime_second_pct",
            "third": "qoq_growth_realtime_third_pct",
        }
    else:
        realtime_cols = {
            "first": "qoq_saar_growth_realtime_first_pct",
            "second": "qoq_saar_growth_realtime_second_pct",
            "third": "qoq_saar_growth_realtime_third_pct",
        }

    has_realtime_cols = all(col in df.columns for col in realtime_cols.values())
    use_realtime_cols = False
    if has_realtime_cols:
        first_rt = pd.to_numeric(df[realtime_cols["first"]], errors="coerce")
        second_rt = pd.to_numeric(df[realtime_cols["second"]], errors="coerce")
        third_rt = pd.to_numeric(df[realtime_cols["third"]], errors="coerce")
        use_realtime_cols = not (first_rt.isna().all() and second_rt.isna().all() and third_rt.isna().all())

    if use_realtime_cols:
        df["first_growth"] = pd.to_numeric(df[realtime_cols["first"]], errors="coerce")
        df["second_growth"] = pd.to_numeric(df[realtime_cols["second"]], errors="coerce")
        df["third_growth"] = pd.to_numeric(df[realtime_cols["third"]], errors="coerce")
        growth_source = "panel_asof_stage_release"
    else:
        required_level_cols = {"first_release", "second_release", "third_release"}
        missing_levels = sorted(required_level_cols.difference(df.columns))
        if missing_levels:
            raise ValueError(
                "Input CSV missing release-level columns needed for fallback growth construction: "
                f"{missing_levels}"
            )
        df["first_growth"] = _growth(df["first_release"], metric=args.metric)
        df["second_growth"] = _growth(df["second_release"], metric=args.metric)
        df["third_growth"] = _growth(df["third_release"], metric=args.metric)
        growth_source = "stitched_release_levels_fallback"

    start_period = pd.Period(args.start_quarter, freq="Q")
    start_ts = pd.Timestamp(start_period.start_time)
    plot_df = df.loc[df["observation_date"] >= start_ts].copy()
    plot_df = plot_df.dropna(subset=["first_growth", "second_growth", "third_growth"], how="all")

    if plot_df.empty:
        raise ValueError(f"No rows available at/after {args.start_quarter}.")

    plot_df["quarter"] = plot_df["observation_date"].map(_quarter_label)

    x = np.arange(len(plot_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width, plot_df["first_growth"], width=width, label="First Estimate q/q", color="#4c78a8")
    ax.bar(x, plot_df["second_growth"], width=width, label="Second Estimate q/q", color="#f58518")
    ax.bar(x + width, plot_df["third_growth"], width=width, label="Third Estimate q/q", color="#54a24b")

    tick_step = 1 if len(plot_df) <= 20 else 2
    tick_idx = x[::tick_step]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(plot_df["quarter"].iloc[::tick_step], rotation=45, ha="right")

    if args.metric == "qoq":
        ax.set_ylabel("Quarter-over-quarter growth (%)")
        title_metric = "q/q %"
    else:
        ax.set_ylabel("Quarter-over-quarter SAAR growth (%)")
        title_metric = "q/q SAAR %"

    ax.set_title(f"Real GDP (GDPC1) Growth by Release Vintage: {title_metric} ({args.start_quarter} onward)")
    if use_realtime_cols and args.metric == "qoq_saar":
        ax.text(
            0.5,
            1.03,
            "growth computed using GDPC1 from fred_qd_vintage_panel as-of stage release date",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4d4d4d",
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    print(f"Saved plot: {output_png}")
    print(f"Rows plotted: {len(plot_df)}")
    print(
        f"Date range: {plot_df['observation_date'].min().date()} "
        f"to {plot_df['observation_date'].max().date()}"
    )
    print(f"Growth source: {growth_source}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
