#!/usr/bin/env python3
# DEV ONLY: non-core utility script retained for development workflows.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import (  # noqa: E402
    DEFAULT_TARGET_SERIES_NAME,
    SUPPORTED_TARGET_TRANSFORMS,
    apply_gdpc1_release_truth_target,
    build_release_target_scaffold,
    exclude_years,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect release-table benchmark target availability.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_SERIES_NAME)
    parser.add_argument(
        "--target_transform",
        type=str,
        default="saar_growth",
        choices=sorted(SUPPORTED_TARGET_TRANSFORMS),
    )
    parser.add_argument(
        "--release_csv",
        type=str,
        default="data/panels/gdpc1_releases_first_second_third.csv",
        help="Path to gdpc1 release table CSV.",
    )
    parser.add_argument(
        "--release_metric",
        type=str,
        choices=["realtime_qoq_saar", "level"],
        default="realtime_qoq_saar",
    )
    parser.add_argument(
        "--release_stage",
        type=str,
        choices=["first", "second", "third", "latest"],
        default="first",
    )
    parser.add_argument(
        "--exclude_years",
        type=int,
        nargs="*",
        default=[],
        help="Calendar years excluded from inspected target series.",
    )
    return parser.parse_args()


def infer_frequency(timestamps: pd.Series) -> str:
    ts = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if len(ts) < 3:
        return "unknown"

    freq = pd.infer_freq(ts)
    if freq:
        return str(freq)

    deltas = ts.diff().dropna().dt.days
    if deltas.empty:
        return "unknown"

    median_days = float(deltas.median())
    if 80 <= median_days <= 100:
        return "quarterly (inferred)"
    if 25 <= median_days <= 35:
        return "monthly (inferred)"
    if 360 <= median_days <= 370:
        return "yearly (inferred)"
    return f"irregular (median_delta_days={median_days:.1f})"


def main() -> None:
    args = parse_args()

    if args.release_metric == "realtime_qoq_saar" and args.target_transform != "saar_growth":
        raise ValueError(
            "--release_metric realtime_qoq_saar requires --target_transform saar_growth."
        )

    scaffold_df, scaffold_meta = build_release_target_scaffold(
        release_csv_path=args.release_csv,
        target_series_name=args.target,
    )
    gdp_df, meta = apply_gdpc1_release_truth_target(
        dataset_df=scaffold_df,
        release_csv_path=args.release_csv,
        release_stage=args.release_stage,
        release_metric=args.release_metric,
        target_transform=args.target_transform,
    )

    years_to_exclude = sorted({int(y) for y in (args.exclude_years or [])})
    before_n = len(gdp_df)
    if years_to_exclude:
        gdp_df = exclude_years(gdp_df, years=years_to_exclude)
    after_n = len(gdp_df)

    ts = pd.to_datetime(gdp_df["timestamp"], errors="coerce").dropna().sort_values()
    start = ts.min()
    end = ts.max()
    freq = infer_frequency(ts)

    print(f"Release table: {meta['release_csv_path']}")
    print(f"Scaffold rows: {scaffold_meta['rows']}")
    print(f"Target selected: {args.target}")
    print(f"Target transform: {args.target_transform}")
    print(
        "Truth mapping: "
        f"metric={meta.get('release_metric')} stage={meta.get('release_stage')} "
        f"column={meta.get('release_column')}"
    )
    print(
        "Coverage: "
        f"quarters={meta.get('release_quarters_available')}, rows_with_release_target={meta.get('rows_with_release_target')}"
    )
    print(f"Excluded years: {years_to_exclude} (rows {before_n} -> {after_n})")
    print(f"Date range: {start} -> {end}")
    print(f"Frequency: {freq}")
    print("First rows:")
    print(gdp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
