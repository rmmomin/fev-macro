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

from fev_macro.vintage_panels import build_md_vintage_panel, write_panel  # noqa: E402


DEFAULT_OUTPUT_BY_MODE = {
    "unprocessed": "data/panels/fred_md_vintage_panel.parquet",
    "processed": "data/panels/fred_md_vintage_panel_processed.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined FRED-MD vintage panel with one row per (vintage, timestamp)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/historical/md/vintages_1999_2026",
        help="Directory containing monthly FRED-MD vintage CSVs.",
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
        help="Panel mode. 'processed' applies FRED transform codes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = str(args.mode).strip().lower()
    output_target = args.output or DEFAULT_OUTPUT_BY_MODE[mode]

    panel = build_md_vintage_panel(md_dir=args.input_dir, mode=mode)
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
    print(f"Saved panel: {output_path}")


if __name__ == "__main__":
    main()
