#!/usr/bin/env python3
"""Query an as-of snapshot from the DuckDB store and write a CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.asof_store import AsofStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=str, default="data/realtime/asof.duckdb")
    p.add_argument("--asof", type=str, required=True, help="As-of cutoff (YYYY-MM-DD or ISO timestamp).")
    p.add_argument("--series", type=str, required=True, help="Comma-separated FRED series IDs.")
    p.add_argument("--obs_start", type=str, default=None)
    p.add_argument("--obs_end", type=str, default=None)
    p.add_argument("--out", type=str, default="data/realtime/asof_snapshot.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    series_ids = [s.strip() for s in str(args.series).split(",") if s.strip()]
    if not series_ids:
        raise SystemExit("No series provided.")

    store = AsofStore(Path(args.db).expanduser().resolve())
    try:
        df = store.snapshot_wide(
            asof_ts=pd.Timestamp(args.asof),
            series_ids=series_ids,
            obs_start=args.obs_start,
            obs_end=args.obs_end,
            timestamp_name="timestamp",
        )
    finally:
        store.close()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
