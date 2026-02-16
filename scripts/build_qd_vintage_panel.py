#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.vintage_panels import build_qd_vintage_panel, write_panel  # noqa: E402


DEFAULT_OUTPUT_BY_MODE = {
    "unprocessed": "data/panels/fred_qd_vintage_panel.parquet",
    "processed": "data/panels/fred_qd_vintage_panel_processed.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined FRED-QD vintage panel with one row per (vintage, timestamp)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/historical/qd/vintages_2018_2026",
        help="Directory containing quarterly FRED-QD vintage CSVs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output panel path (.parquet or .csv). Defaults by mode.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["unprocessed", "processed"],
        default="unprocessed",
        help="Panel mode. 'processed' applies FRED transform codes to covariates and keeps GDPC1 untransformed.",
    )
    return parser.parse_args()


def _assert_qd_processed_vs_unprocessed(
    *,
    input_dir: str,
    mode: str,
    built_panel: pd.DataFrame,
) -> tuple[bool, int]:
    if mode == "processed":
        processed = built_panel
        unprocessed = build_qd_vintage_panel(qd_dir=input_dir, mode="unprocessed")
    else:
        unprocessed = built_panel
        processed = build_qd_vintage_panel(qd_dir=input_dir, mode="processed")

    key_cols = ["vintage", "timestamp"]
    left = unprocessed[key_cols + ["GDPC1"]].copy()
    right = processed[key_cols + ["GDPC1"]].copy()
    merged = left.merge(right, on=key_cols, how="inner", suffixes=("_unprocessed", "_processed"))
    if merged.empty:
        raise ValueError("QD sanity check failed: no overlapping rows across processed/unprocessed panels.")

    gdpc1_same = np.allclose(
        pd.to_numeric(merged["GDPC1_unprocessed"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(merged["GDPC1_processed"], errors="coerce").to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    if not gdpc1_same:
        raise AssertionError("QD sanity check failed: GDPC1 differs between processed and unprocessed panels.")

    shared_cols = sorted(
        set(unprocessed.columns).intersection(set(processed.columns)).difference({"vintage", "vintage_timestamp", "timestamp", "GDPC1"})
    )
    changed_count = 0
    for col in shared_cols:
        left_col = pd.to_numeric(unprocessed[col], errors="coerce").to_numpy(dtype=float)
        right_col = pd.to_numeric(processed[col], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left_col, right_col, rtol=0.0, atol=0.0, equal_nan=True):
            changed_count += 1

    if changed_count <= 0:
        raise AssertionError(
            "QD sanity check failed: no covariate columns changed between processed and unprocessed panels."
        )

    return gdpc1_same, changed_count


def main() -> None:
    args = parse_args()
    mode = str(args.mode).strip().lower()
    output_target = args.output or DEFAULT_OUTPUT_BY_MODE[mode]

    panel = build_qd_vintage_panel(qd_dir=args.input_dir, mode=mode)
    gdpc1_same, changed_count = _assert_qd_processed_vs_unprocessed(
        input_dir=args.input_dir,
        mode=mode,
        built_panel=panel,
    )
    output_path = write_panel(panel, output_target)

    ts = pd.to_datetime(panel["timestamp"], errors="coerce").dropna()
    vintage_count = panel["vintage"].nunique()
    variable_count = len([c for c in panel.columns if c not in {"vintage", "vintage_timestamp", "timestamp"}])

    print(f"Input dir: {Path(args.input_dir).expanduser().resolve()}")
    print(f"Mode: {mode}")
    print(f"Rows: {len(panel)}")
    print(f"Vintages: {vintage_count}")
    print(f"Variables: {variable_count}")
    if not ts.empty:
        print(f"Timestamp range: {ts.min().date()} -> {ts.max().date()}")
    print(f"Sanity: GDPC1 unchanged across modes={gdpc1_same}")
    print(f"Sanity: transformed covariates changed={changed_count}")
    print(f"Saved panel: {output_path}")


if __name__ == "__main__":
    main()
