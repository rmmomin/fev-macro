#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import (  # noqa: E402
    DEFAULT_SOURCE_SERIES_CANDIDATES,
    build_gdp_saar_growth_series,
    find_gdp_column_candidates,
    load_fev_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect fev dataset and GDP target availability.")
    parser.add_argument("--dataset_path", type=str, default="autogluon/fev_datasets")
    parser.add_argument("--dataset_config", type=str, default="fred_qd_2025")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--target", type=str, default="GDP_SAAR")
    parser.add_argument(
        "--source_series_candidates",
        nargs="+",
        default=list(DEFAULT_SOURCE_SERIES_CANDIDATES),
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

    dataset = load_fev_dataset(
        config=args.dataset_config,
        dataset_path=args.dataset_path,
        dataset_revision=args.dataset_revision,
    )

    candidates = find_gdp_column_candidates(dataset)
    gdp_df, meta = build_gdp_saar_growth_series(
        dataset=dataset,
        target_series_name=args.target,
        source_series_candidates=args.source_series_candidates,
        include_covariates=True,
    )

    ts = pd.to_datetime(gdp_df["timestamp"], errors="coerce").dropna().sort_values()
    start = ts.min()
    end = ts.max()
    freq = infer_frequency(ts)

    print(f"Dataset: {args.dataset_path} (config={args.dataset_config})")
    print(f"Rows: {len(dataset)}, Columns: {dataset.column_names}")
    print(f"Candidate GDP IDs: {candidates['id_candidates']}")
    print(f"Candidate GDP columns: {candidates['column_candidates']}")
    print(f"Target selected: {meta['target_series']} (computed={meta['computed']}, source={meta['source_series']})")
    covs = meta.get("covariate_columns", [])
    print(f"Covariates discovered: {len(covs)}")
    if covs:
        print(f"Covariate sample: {covs[:20]}")
    print(f"Date range: {start} -> {end}")
    print(f"Frequency: {freq}")
    print("First rows:")
    print(gdp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
