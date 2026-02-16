#!/usr/bin/env python3
# DEV ONLY: non-core utility script retained for development workflows.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.vintage_panels import discover_md_vintage_files, discover_qd_vintage_files  # noqa: E402

SMALL = 1e-6
SUPPORTED_TCODES = {1, 2, 3, 4, 5, 6, 7}
TRANSFORM_ROW_KEYS = {"transform", "transformation", "tcode", "tcodes"}
META_COLS = ("vintage", "vintage_timestamp", "timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process MD/QD vintage panels by vintage date using documented "
            "FRED tcode transforms and MD outlier filtering."
        )
    )
    parser.add_argument(
        "--md_panel_input",
        type=str,
        default="data/panels/fred_md_vintage_panel.parquet",
        help="Input MD vintage panel parquet.",
    )
    parser.add_argument(
        "--qd_panel_input",
        type=str,
        default="data/panels/fred_qd_vintage_panel.parquet",
        help="Input QD vintage panel parquet.",
    )
    parser.add_argument(
        "--md_vintage_dir",
        type=str,
        default="data/historical/md/vintages_1999_2026",
        help="Directory containing raw MD vintage CSVs (for transform codes).",
    )
    parser.add_argument(
        "--qd_vintage_dir",
        type=str,
        default="data/historical/qd/vintages_2018_2026",
        help="Directory containing raw QD vintage CSVs (for transform codes).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/panels",
        help="Output directory.",
    )
    parser.add_argument(
        "--md_output_name",
        type=str,
        default="fred_md_vintage_panel_processed.parquet",
        help="Output filename for processed MD panel.",
    )
    parser.add_argument(
        "--qd_output_name",
        type=str,
        default="fred_qd_vintage_panel_processed.parquet",
        help="Output filename for processed QD panel.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="fred_vintage_panel_processed_manifest.json",
        help="Output filename for processing summary metadata.",
    )
    parser.add_argument(
        "--md_drop_initial_rows",
        type=int,
        default=2,
        help="Rows to drop from start of each transformed MD vintage slice.",
    )
    parser.add_argument(
        "--qd_drop_initial_rows",
        type=int,
        default=0,
        help="Rows to drop from start of each transformed QD vintage slice.",
    )
    return parser.parse_args()


def _normalize_key(value: object) -> str:
    return str(value).strip().lower().rstrip(":")


def _extract_transform_codes(vintage_csv_path: Path) -> dict[str, int]:
    raw_df = pd.read_csv(vintage_csv_path, dtype=str)
    if raw_df.empty:
        return {}

    first_col_name = str(raw_df.columns[0])
    key = raw_df[first_col_name].map(_normalize_key)
    mask = key.isin(TRANSFORM_ROW_KEYS)
    if not mask.any():
        return {}

    row = raw_df.loc[mask].iloc[0]
    codes: dict[str, int] = {}
    for col in raw_df.columns:
        if col == first_col_name:
            continue
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.isna(val):
            continue
        code = int(val)
        if code in SUPPORTED_TCODES:
            codes[str(col)] = code
    return codes


def _transxf(x: np.ndarray, tcode: int) -> np.ndarray:
    n = len(x)
    y = np.full(n, np.nan, dtype=float)

    if tcode == 1:
        return x.copy()

    if tcode == 2:
        y[1:] = x[1:] - x[:-1]
        return y

    if tcode == 3:
        y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
        return y

    finite = x[np.isfinite(x)]
    can_log = finite.size == 0 or np.min(finite) > SMALL

    if tcode == 4:
        if can_log:
            y = np.log(x)
        return y

    if tcode == 5:
        if can_log:
            lx = np.log(x)
            y[1:] = lx[1:] - lx[:-1]
        return y

    if tcode == 6:
        if can_log:
            lx = np.log(x)
            y[2:] = lx[2:] - 2 * lx[1:-1] + lx[:-2]
        return y

    if tcode == 7:
        y1 = np.full(n, np.nan, dtype=float)
        y1[1:] = (x[1:] - x[:-1]) / x[:-1]
        y[2:] = y1[2:] - y1[1:-1]
        return y

    raise ValueError(f"Unsupported transform code: {tcode}")


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


def _process_panel_by_vintage(
    panel_df: pd.DataFrame,
    vintage_file_map: dict[pd.Period, Path],
    mode: str,
    drop_initial_rows: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = set(META_COLS)
    if not required.issubset(panel_df.columns):
        missing = sorted(required - set(panel_df.columns))
        raise ValueError(f"Panel is missing required columns: {missing}")

    out_frames: list[pd.DataFrame] = []
    codes_cache: dict[Path, dict[str, int]] = {}
    missing_vintage_files: list[str] = []
    data_cols = [c for c in panel_df.columns if c not in required]
    total_outliers = 0

    grouped = panel_df.groupby("vintage", sort=True, dropna=False)
    for vintage_label, vintage_slice in grouped:
        if pd.isna(vintage_label):
            continue

        period = pd.Period(str(vintage_label), freq="M")
        csv_path = vintage_file_map.get(period)
        if csv_path is None:
            missing_vintage_files.append(str(vintage_label))
            continue

        if csv_path not in codes_cache:
            codes_cache[csv_path] = _extract_transform_codes(csv_path)
        transform_codes = codes_cache[csv_path]

        work = vintage_slice.sort_values("timestamp").reset_index(drop=True).copy()
        for col in data_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            tcode = transform_codes.get(col)
            if tcode in SUPPORTED_TCODES:
                work[col] = _transxf(work[col].to_numpy(dtype=float), int(tcode))

        if mode == "md":
            work, outlier_count = _remove_outliers_iqr(work, data_cols)
            total_outliers += int(outlier_count)

        if drop_initial_rows > 0:
            work = work.iloc[drop_initial_rows:].reset_index(drop=True)

        out_frames.append(work)

    if missing_vintage_files:
        first = ", ".join(missing_vintage_files[:10])
        extra = "" if len(missing_vintage_files) <= 10 else f" (+{len(missing_vintage_files)-10} more)"
        raise FileNotFoundError(
            f"Could not resolve source CSV for {len(missing_vintage_files)} vintages: {first}{extra}"
        )

    if not out_frames:
        raise ValueError(f"No processed slices were produced for mode={mode}.")

    out = pd.concat(out_frames, axis=0, ignore_index=True, sort=False)
    out = out.sort_values(["vintage", "timestamp"]).reset_index(drop=True)
    out = out[list(panel_df.columns)]
    summary = {
        "input_rows": int(len(panel_df)),
        "output_rows": int(len(out)),
        "vintage_count": int(panel_df["vintage"].nunique()),
        "variable_count": int(len(data_cols)),
        "drop_initial_rows": int(max(0, drop_initial_rows)),
        "outliers_removed": int(total_outliers),
    }
    return out, summary


def main() -> None:
    args = parse_args()

    md_panel_input = Path(args.md_panel_input).expanduser().resolve()
    qd_panel_input = Path(args.qd_panel_input).expanduser().resolve()
    md_vintage_dir = Path(args.md_vintage_dir).expanduser().resolve()
    qd_vintage_dir = Path(args.qd_vintage_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_panel = pd.read_parquet(md_panel_input)
    qd_panel = pd.read_parquet(qd_panel_input)

    md_vintage_files = discover_md_vintage_files(md_vintage_dir)
    qd_vintage_files = discover_qd_vintage_files(qd_vintage_dir)

    md_processed, md_summary = _process_panel_by_vintage(
        panel_df=md_panel,
        vintage_file_map=md_vintage_files,
        mode="md",
        drop_initial_rows=max(0, args.md_drop_initial_rows),
    )
    qd_processed, qd_summary = _process_panel_by_vintage(
        panel_df=qd_panel,
        vintage_file_map=qd_vintage_files,
        mode="qd",
        drop_initial_rows=max(0, args.qd_drop_initial_rows),
    )

    md_output = output_dir / args.md_output_name
    qd_output = output_dir / args.qd_output_name
    manifest_output = output_dir / args.manifest_name

    md_processed.to_parquet(md_output, index=False)
    qd_processed.to_parquet(qd_output, index=False)

    manifest = {
        "md": {
            "input_panel": str(md_panel_input),
            "output_panel": str(md_output),
            "vintage_dir": str(md_vintage_dir),
            **md_summary,
        },
        "qd": {
            "input_panel": str(qd_panel_input),
            "output_panel": str(qd_output),
            "vintage_dir": str(qd_vintage_dir),
            **qd_summary,
        },
        "notes": {
            "transform_logic": "FRED tcodes 1-7 from MATLAB prepare_missing.m and R fredmd.R/fredqd.R",
            "md_outlier_rule": "abs(x - median) > 10*IQR from MATLAB remove_outliers.m and R rm_outliers.fredmd",
            "applied_by": "vintage slice",
        },
    }
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved MD processed panel: {md_output}")
    print(f"Saved QD processed panel: {qd_output}")
    print(f"Saved processing manifest: {manifest_output}")
    print(
        f"MD rows {md_summary['input_rows']} -> {md_summary['output_rows']} "
        f"(outliers_removed={md_summary['outliers_removed']})"
    )
    print(
        f"QD rows {qd_summary['input_rows']} -> {qd_summary['output_rows']} "
        f"(outliers_removed={qd_summary['outliers_removed']})"
    )


if __name__ == "__main__":
    main()
