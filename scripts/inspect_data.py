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
    DEFAULT_TARGET_SERIES_NAME,
    DEFAULT_TARGET_TRANSFORM,
    SUPPORTED_TARGET_TRANSFORMS,
    build_real_gdp_target_series,
    exclude_years,
    find_gdp_column_candidates,
    load_fred_qd_transform_codes,
    load_fev_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect fev dataset and GDP target availability.")
    parser.add_argument("--dataset_path", type=str, default="autogluon/fev_datasets")
    parser.add_argument("--dataset_config", type=str, default="fred_qd_2025")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_SERIES_NAME)
    parser.add_argument(
        "--target_transform",
        type=str,
        default=DEFAULT_TARGET_TRANSFORM,
        choices=sorted(SUPPORTED_TARGET_TRANSFORMS),
    )
    parser.add_argument(
        "--source_series_candidates",
        nargs="+",
        default=list(DEFAULT_SOURCE_SERIES_CANDIDATES),
    )
    parser.add_argument(
        "--exclude_years",
        type=int,
        nargs="*",
        default=[2020],
        help="Calendar years excluded from inspected target series.",
    )
    parser.add_argument(
        "--historical_qd_dir",
        type=str,
        default="data/historical/qd/vintages_2018_2026",
        help="Path to historical FRED-QD vintage CSVs used to load transform codes.",
    )
    parser.add_argument(
        "--disable_fred_transforms",
        action="store_true",
        help="Disable FRED-MD/QD tcode transforms for covariates during dataset construction.",
    )
    parser.add_argument(
        "--fred_transform_vintage",
        type=str,
        default=None,
        help="Optional transform-code vintage month (YYYY-MM); default uses latest available historical vintage.",
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

    apply_fred_transforms = not args.disable_fred_transforms
    transform_vintage = pd.Period(args.fred_transform_vintage, freq="M") if args.fred_transform_vintage else None
    fred_transform_codes: dict[str, int] = {}
    if apply_fred_transforms:
        try:
            fred_transform_codes = load_fred_qd_transform_codes(
                historical_qd_dir=args.historical_qd_dir,
                vintage_period=transform_vintage,
            )
        except Exception as err:
            print(f"Warning: unable to load FRED transform codes ({err}); continuing without transforms.")
            apply_fred_transforms = False

    candidates = find_gdp_column_candidates(dataset)
    gdp_df, meta = build_real_gdp_target_series(
        dataset=dataset,
        target_series_name=args.target,
        target_transform=args.target_transform,
        source_series_candidates=args.source_series_candidates,
        include_covariates=True,
        apply_fred_transforms=apply_fred_transforms,
        fred_transform_codes=fred_transform_codes,
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

    print(f"Dataset: {args.dataset_path} (config={args.dataset_config})")
    print(f"Rows: {len(dataset)}, Columns: {dataset.column_names}")
    print(f"Candidate GDP IDs: {candidates['id_candidates']}")
    print(f"Candidate GDP columns: {candidates['column_candidates']}")
    print(f"Target selected: {meta['target_series']} (computed={meta['computed']}, source={meta['source_series']})")
    print(f"Target transform: {meta.get('target_transform', args.target_transform)}")
    covs = meta.get("covariate_columns", [])
    print(f"Covariates discovered: {len(covs)}")
    if covs:
        print(f"Covariate sample: {covs[:20]}")
    print(
        "FRED transforms enabled: "
        f"{apply_fred_transforms}; transform codes loaded={len(fred_transform_codes)}; "
        f"covariates transformed={len(meta.get('transformed_covariates', []))}"
    )
    print(f"Excluded years: {years_to_exclude} (rows {before_n} -> {after_n})")
    print(f"Date range: {start} -> {end}")
    print(f"Frequency: {freq}")
    print("First rows:")
    print(gdp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
