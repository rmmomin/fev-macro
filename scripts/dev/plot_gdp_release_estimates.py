#!/usr/bin/env python3
# DEV ONLY: non-core utility script retained for development workflows.
"""Plot clustered columns for first/second/third GDP release estimates."""

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
            "Create a clustered column chart of first, second, and third GDPC1 "
            "estimates from a release dataset."
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
        default="results/gdpc1_release_estimates_clustered_2018q1.png",
        help="Output plot path.",
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
    required_cols = {"observation_date", "first_release", "second_release", "third_release"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df = df.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)

    start_period = pd.Period(args.start_quarter, freq="Q")
    start_ts = pd.Timestamp(start_period.start_time)
    plot_df = df.loc[df["observation_date"] >= start_ts].copy()
    plot_df = plot_df.dropna(subset=["first_release", "second_release", "third_release"])

    if plot_df.empty:
        raise ValueError(f"No rows available at/after {args.start_quarter}.")

    plot_df["quarter"] = plot_df["observation_date"].map(_quarter_label)

    x = np.arange(len(plot_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width, plot_df["first_release"], width=width, label="First Estimate", color="#4c78a8")
    ax.bar(x, plot_df["second_release"], width=width, label="Second Estimate", color="#f58518")
    ax.bar(x + width, plot_df["third_release"], width=width, label="Third Estimate", color="#54a24b")

    tick_step = 1 if len(plot_df) <= 20 else 2
    tick_idx = x[::tick_step]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(plot_df["quarter"].iloc[::tick_step], rotation=45, ha="right")
    ax.set_ylabel("Billions of chained 2017 dollars (SAAR)")
    ax.set_title(f"Real GDP (GDPC1) Release Estimates: First/Second/Third ({args.start_quarter} onward)")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
