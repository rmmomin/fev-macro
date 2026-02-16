#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.fred_transforms import apply_fred_transform_codes, extract_fred_transform_codes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process latest FRED-MD and FRED-QD snapshots using documented "
            "FRED transform codes and MD outlier filtering."
        )
    )
    parser.add_argument(
        "--md_input",
        type=str,
        default="data/latest/fred_md_latest.csv",
        help="Path to latest monthly (MD) snapshot CSV.",
    )
    parser.add_argument(
        "--qd_input",
        type=str,
        default="data/latest/fred_qd_latest.csv",
        help="Path to latest quarterly (QD) snapshot CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory where processed files will be written.",
    )
    parser.add_argument(
        "--md_output_name",
        type=str,
        default="fred_md_latest_processed.csv",
        help="Output filename for processed MD data.",
    )
    parser.add_argument(
        "--qd_output_name",
        type=str,
        default="fred_qd_latest_processed.csv",
        help="Output filename for processed QD data.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="fred_latest_processing_manifest.json",
        help="Output filename for processing metadata.",
    )
    parser.add_argument(
        "--md_drop_initial_rows",
        type=int,
        default=2,
        help=(
            "Rows to drop from the beginning of transformed MD data "
            "(MATLAB fredfactors.m behavior)."
        ),
    )
    parser.add_argument(
        "--qd_drop_initial_rows",
        type=int,
        default=0,
        help="Rows to drop from the beginning of transformed QD data.",
    )
    return parser.parse_args()


def _load_snapshot_rows(csv_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_df = pd.read_csv(csv_path, dtype=str)
    if raw_df.empty:
        raise ValueError(f"CSV has no rows: {csv_path}")

    first_col_name = str(raw_df.columns[0])
    transform_codes = extract_fred_transform_codes(raw_df=raw_df, first_col_name=first_col_name)

    parsed_dates = pd.to_datetime(raw_df[first_col_name], format="%m/%d/%Y", errors="coerce")
    data = raw_df.loc[parsed_dates.notna()].copy()
    if data.empty:
        raise ValueError(f"No data rows with mm/dd/YYYY dates found in: {csv_path}")

    data["date"] = pd.to_datetime(data[first_col_name], format="%m/%d/%Y", errors="coerce")
    data = data.drop(columns=[first_col_name])
    for col in data.columns:
        if col == "date":
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    ordered = ["date"] + [c for c in data.columns if c != "date"]
    data = data[ordered].sort_values("date").reset_index(drop=True)
    return data, transform_codes


def _apply_transform_codes(df: pd.DataFrame, transform_codes: dict[str, int]) -> pd.DataFrame:
    series_cols = [c for c in df.columns if c != "date"]
    return apply_fred_transform_codes(df, transform_codes=transform_codes, columns=series_cols)


def _remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    panel = out[columns]

    median_vals = panel.median(axis=0, skipna=True)
    q1 = panel.quantile(0.25, axis=0, numeric_only=True)
    q3 = panel.quantile(0.75, axis=0, numeric_only=True)
    iqr = q3 - q1

    distance = panel.sub(median_vals, axis=1).abs()
    outlier_mask = distance.gt(10.0 * iqr, axis=1)

    out.loc[:, columns] = panel.mask(outlier_mask)
    return out, int(outlier_mask.to_numpy(dtype=bool).sum())


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    md_input = Path(args.md_input).expanduser().resolve()
    qd_input = Path(args.qd_input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_raw, md_codes = _load_snapshot_rows(md_input)
    md_transformed = _apply_transform_codes(md_raw, md_codes)
    md_series_cols = [c for c in md_transformed.columns if c != "date"]
    md_no_outliers, md_outlier_count = _remove_outliers_iqr(md_transformed, md_series_cols)
    md_processed = md_no_outliers.iloc[max(0, args.md_drop_initial_rows) :].reset_index(drop=True)

    qd_raw, qd_codes = _load_snapshot_rows(qd_input)
    qd_transformed = _apply_transform_codes(qd_raw, qd_codes)
    qd_processed = qd_transformed.iloc[max(0, args.qd_drop_initial_rows) :].reset_index(drop=True)

    md_output_path = output_dir / args.md_output_name
    qd_output_path = output_dir / args.qd_output_name
    manifest_path = output_dir / args.manifest_name

    _write_csv(md_processed, md_output_path)
    _write_csv(qd_processed, qd_output_path)

    manifest = {
        "md": {
            "input": str(md_input),
            "output": str(md_output_path),
            "raw_rows": int(len(md_raw)),
            "processed_rows": int(len(md_processed)),
            "transform_code_count": int(len(md_codes)),
            "outliers_removed": int(md_outlier_count),
            "drop_initial_rows": int(max(0, args.md_drop_initial_rows)),
        },
        "qd": {
            "input": str(qd_input),
            "output": str(qd_output_path),
            "raw_rows": int(len(qd_raw)),
            "processed_rows": int(len(qd_processed)),
            "transform_code_count": int(len(qd_codes)),
            "drop_initial_rows": int(max(0, args.qd_drop_initial_rows)),
        },
        "notes": {
            "transform_logic": "FRED tcodes 1-7 from MATLAB prepare_missing.m / R fredmd.R and fredqd.R",
            "md_outlier_rule": "abs(x - median) > 10 * IQR from MATLAB remove_outliers.m / R rm_outliers.fredmd",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Processed MD rows: {len(md_processed)} -> {md_output_path}")
    print(f"Processed QD rows: {len(qd_processed)} -> {qd_output_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
