#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run latest-vintage forecast using latest processed MD/QD datasets. "
            "By default, this restores GDPC1 levels from data/latest/fred_qd_latest.csv "
            "before calling scripts/run_latest_vintage_forecast.py."
        )
    )
    parser.add_argument(
        "--processed_qd_csv",
        type=str,
        default="data/processed/fred_qd_latest_processed.csv",
        help="Processed latest QD CSV input.",
    )
    parser.add_argument(
        "--processed_md_csv",
        type=str,
        default="data/processed/fred_md_latest_processed.csv",
        help="Processed latest MD CSV input.",
    )
    parser.add_argument(
        "--raw_qd_csv",
        type=str,
        default="data/latest/fred_qd_latest.csv",
        help="Raw latest QD CSV used to restore level target values.",
    )
    parser.add_argument(
        "--latest_manifest",
        type=str,
        default="data/latest/fred_latest_fetch_manifest.json",
        help="Latest-fetch manifest used to infer processed vintage when not provided.",
    )
    parser.add_argument(
        "--processed_vintage",
        type=str,
        default="",
        help="Optional YYYY-MM vintage label for processed latest datasets.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="GDPC1",
        help="Target level column to forecast.",
    )
    parser.add_argument(
        "--target_quarter",
        type=str,
        default="auto",
        help="Target quarter in YYYYQ# format. Use 'auto' for next quarter after last observed target.",
    )
    parser.add_argument(
        "--release_csv",
        type=str,
        default="",
        help="Optional release table path passed through to the base script.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model list passed through to the base script.",
    )
    parser.add_argument("--train_window", type=str, default="expanding", choices=["expanding", "rolling"])
    parser.add_argument("--rolling_size", type=int, default=None)
    parser.add_argument(
        "--exclude_years",
        type=int,
        nargs="*",
        default=[],
        help="Optional years excluded inside mixed-frequency MD factor models.",
    )
    parser.add_argument(
        "--per_model_timeout_sec",
        type=int,
        default=180,
        help="Timeout in seconds per model prediction attempt.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/realtime_latest_processed_data_forecast.csv",
        help="Output CSV for forecasts.",
    )
    parser.add_argument(
        "--tmp_qd_csv",
        type=str,
        default="",
        help="Optional explicit path for temporary adjusted processed QD CSV.",
    )
    return parser.parse_args()


def _infer_vintage_from_filename(path: Path) -> str | None:
    text = path.name.lower()
    m = re.search(r"(\d{4})m(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    m = re.search(r"(\d{4})-(\d{2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    return None


def _resolve_processed_vintage(args: argparse.Namespace, processed_qd_path: Path) -> str:
    if args.processed_vintage:
        return str(pd.Period(str(args.processed_vintage), freq="M"))

    manifest_path = Path(args.latest_manifest).expanduser().resolve()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            generated = manifest.get("generated_at_utc")
            if generated:
                return str(pd.Timestamp(generated).to_period("M"))
        except Exception:
            pass

    inferred = _infer_vintage_from_filename(processed_qd_path)
    if inferred:
        return str(pd.Period(inferred, freq="M"))

    return str(pd.Timestamp.today().to_period("M"))


def _load_raw_target_levels(raw_qd_path: Path, target_col: str) -> pd.Series:
    raw = pd.read_csv(raw_qd_path, dtype=str)
    if raw.empty:
        raise ValueError(f"Raw latest QD CSV is empty: {raw_qd_path}")
    if target_col not in raw.columns:
        raise ValueError(f"Raw latest QD CSV missing target column '{target_col}': {raw_qd_path}")

    first_col = str(raw.columns[0])
    parsed_dates = pd.to_datetime(raw[first_col], format="%m/%d/%Y", errors="coerce")
    data = raw.loc[parsed_dates.notna(), [target_col]].copy()
    if data.empty:
        raise ValueError(f"No data rows with mm/dd/YYYY dates found in: {raw_qd_path}")

    data["date"] = parsed_dates.loc[parsed_dates.notna()].dt.strftime("%Y-%m-%d").values
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    out = data.drop_duplicates(subset=["date"], keep="last").set_index("date")[target_col]
    return out.astype(float)


def _build_adjusted_processed_qd_csv(
    processed_qd_path: Path,
    raw_qd_path: Path,
    target_col: str,
    output_csv_path: Path,
) -> tuple[Path, int]:
    processed = pd.read_csv(processed_qd_path)
    if "date" not in processed.columns:
        raise ValueError(f"Processed latest QD CSV must include 'date' column: {processed_qd_path}")
    if target_col not in processed.columns:
        raise ValueError(f"Processed latest QD CSV missing target column '{target_col}': {processed_qd_path}")

    processed = processed.copy()
    processed["date"] = pd.to_datetime(processed["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw_target = _load_raw_target_levels(raw_qd_path=raw_qd_path, target_col=target_col)

    processed[target_col] = pd.to_numeric(processed["date"].map(raw_target), errors="coerce")
    replaced = int(processed[target_col].notna().sum())
    if replaced == 0:
        raise ValueError(
            "No target levels were restored from raw latest QD CSV. "
            f"Check date alignment between {processed_qd_path} and {raw_qd_path}."
        )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_csv_path, index=False)
    return output_csv_path, replaced


def _resolve_target_quarter(adjusted_qd_csv: Path, target_col: str, requested: str) -> str:
    if str(requested).lower() != "auto":
        return str(pd.Period(str(requested), freq="Q-DEC"))

    df = pd.read_csv(adjusted_qd_csv)
    if "date" not in df.columns:
        raise ValueError(f"Adjusted processed QD CSV missing 'date' column: {adjusted_qd_csv}")
    if target_col not in df.columns:
        raise ValueError(f"Adjusted processed QD CSV missing '{target_col}' column: {adjusted_qd_csv}")

    dates = pd.to_datetime(df["date"], errors="coerce")
    target_values = pd.to_numeric(df[target_col], errors="coerce")
    observed_quarters = pd.PeriodIndex(dates[target_values.notna()], freq="Q-DEC")
    if observed_quarters.empty:
        raise ValueError(f"No observed non-missing target values for '{target_col}'.")
    return str(observed_quarters.max() + 1)


def main() -> int:
    args = parse_args()

    processed_qd_path = Path(args.processed_qd_csv).expanduser().resolve()
    processed_md_path = Path(args.processed_md_csv).expanduser().resolve()
    raw_qd_path = Path(args.raw_qd_csv).expanduser().resolve()
    output_csv_path = Path(args.output_csv).expanduser().resolve()
    run_script_path = (ROOT / "scripts" / "run_latest_vintage_forecast.py").resolve()

    if not run_script_path.exists():
        raise FileNotFoundError(f"Base script not found: {run_script_path}")
    if not processed_qd_path.exists():
        raise FileNotFoundError(f"Processed latest QD CSV not found: {processed_qd_path}")
    if not processed_md_path.exists():
        raise FileNotFoundError(f"Processed latest MD CSV not found: {processed_md_path}")
    if not raw_qd_path.exists():
        raise FileNotFoundError(f"Raw latest QD CSV not found: {raw_qd_path}")

    vintage_id = _resolve_processed_vintage(args=args, processed_qd_path=processed_qd_path)
    if args.tmp_qd_csv:
        tmp_qd_path = Path(args.tmp_qd_csv).expanduser().resolve()
    else:
        tmp_qd_path = output_csv_path.parent / f"{processed_qd_path.stem}_with_{args.target_col}_levels.csv"

    adjusted_qd_path, restored_count = _build_adjusted_processed_qd_csv(
        processed_qd_path=processed_qd_path,
        raw_qd_path=raw_qd_path,
        target_col=args.target_col,
        output_csv_path=tmp_qd_path,
    )
    target_quarter = _resolve_target_quarter(
        adjusted_qd_csv=adjusted_qd_path,
        target_col=args.target_col,
        requested=args.target_quarter,
    )

    cmd = [
        sys.executable,
        str(run_script_path),
        "--mode",
        "processed",
        "--processed_qd_csv",
        str(adjusted_qd_path),
        "--processed_md_csv",
        str(processed_md_path),
        "--processed_vintage",
        str(vintage_id),
        "--target_col",
        str(args.target_col),
        "--target_quarter",
        str(target_quarter),
        "--train_window",
        str(args.train_window),
        "--output_csv",
        str(output_csv_path),
        "--per_model_timeout_sec",
        str(int(args.per_model_timeout_sec)),
    ]
    if args.release_csv:
        cmd.extend(["--release_csv", str(Path(args.release_csv).expanduser().resolve())])
    if args.rolling_size is not None:
        cmd.extend(["--rolling_size", str(int(args.rolling_size))])
    if args.exclude_years:
        cmd.extend(["--exclude_years", *[str(int(y)) for y in args.exclude_years]])
    if args.models:
        cmd.extend(["--models", *[str(m) for m in args.models]])

    print(f"Resolved processed vintage: {vintage_id}")
    print(f"Adjusted processed QD CSV: {adjusted_qd_path} (restored_target_rows={restored_count})")
    print(f"Forecast target quarter: {target_quarter}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Running: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
