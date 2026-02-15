#!/usr/bin/env python3
"""Build first/second/third GDPC1 release dataset from ALFRED.

The script downloads ALFRED vintage data for a series (default: GDPC1), extracts
first/second/third available releases per observation date, and adds q/q growth
columns computed from the latest-available estimate for the current and previous
quarter. It also computes realtime q/q SAAR growth for first/second/third
releases using the previous-quarter level available as-of each current release
vintage date.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ALFRED_HOST = "https://alfred.stlouisfed.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ALFRED vintage data and build first/second/third release "
            "dataset with latest-based q/q growth columns."
        )
    )
    parser.add_argument("--series", default="GDPC1", help="ALFRED series ID (default: GDPC1).")
    parser.add_argument(
        "--output_csv",
        default="data/panels/gdpc1_releases_first_second_third.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output_parquet",
        default="",
        help="Optional output parquet path (requires pyarrow/fastparquet).",
    )
    parser.add_argument(
        "--obs_start",
        default="",
        help="Optional observation start date YYYY-MM-DD (defaults to ALFRED form default).",
    )
    parser.add_argument(
        "--obs_end",
        default="",
        help="Optional observation end date YYYY-MM-DD (defaults to ALFRED form default).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for requests (default: 180).",
    )
    return parser.parse_args()


def _extract_input_value(html: str, input_id: str) -> str | None:
    pattern = rf'<input[^>]*id="{re.escape(input_id)}"[^>]*value="([^"]*)"'
    match = re.search(pattern, html, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _extract_select_values(html: str, select_id: str) -> list[str]:
    select_pattern = rf'<select[^>]*id="{re.escape(select_id)}"[^>]*>(.*?)</select>'
    select_match = re.search(select_pattern, html, flags=re.IGNORECASE | re.DOTALL)
    if not select_match:
        return []
    block = select_match.group(1)
    values = re.findall(r'<option\s+value="([^"]+)"', block, flags=re.IGNORECASE)
    return [v.strip() for v in values if v.strip()]


def fetch_download_form(session: requests.Session, series: str, timeout: int) -> str:
    url = f"{ALFRED_HOST}/series/downloaddata?seid={series}"
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_download_form(html: str) -> tuple[str, str, list[str]]:
    obs_start = _extract_input_value(html, "form_obs_start_date")
    obs_end = _extract_input_value(html, "form_obs_end_date")
    vintage_dates = _extract_select_values(html, "form_selected_vintage_dates")

    if not obs_start or not obs_end:
        raise ValueError("Could not parse observation date bounds from ALFRED downloaddata form.")
    if not vintage_dates:
        raise ValueError("Could not parse vintage date options from ALFRED downloaddata form.")

    return obs_start, obs_end, vintage_dates


def download_vintage_zip(
    session: requests.Session,
    series: str,
    obs_start: str,
    obs_end: str,
    vintage_dates: list[str],
    timeout: int,
) -> bytes:
    url = f"{ALFRED_HOST}/series/downloaddata?seid={series}"

    form_data: list[tuple[str, str]] = [
        ("form[units]", "lin"),
        ("form[obs_start_date]", obs_start),
        ("form[obs_end_date]", obs_end),
        ("form[file_type]", "2"),
        ("form[file_format]", "csv"),
        ("form[download_data]", "Download data"),
    ]
    form_data.extend(("form[selected_vintage_dates][]", d) for d in vintage_dates)

    response = session.post(url, data=form_data, timeout=timeout)
    response.raise_for_status()

    ctype = (response.headers.get("content-type") or "").lower()
    if "zip" not in ctype and not response.content.startswith(b"PK"):
        raise RuntimeError(
            "ALFRED did not return a zip payload. "
            f"content-type={response.headers.get('content-type')}"
        )

    return response.content


def load_wide_vintage_csv(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV found in ALFRED zip payload.")
        csv_name = sorted(csv_files)[0]
        with zf.open(csv_name) as handle:
            wide = pd.read_csv(handle)

    if "observation_date" not in wide.columns:
        raise ValueError("Expected 'observation_date' column in ALFRED CSV.")

    return wide


def _extract_vintage_date_from_col(col: str, series: str) -> pd.Timestamp | None:
    match = re.fullmatch(rf"{re.escape(series)}_(\d{{8}})", col)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def _compute_saar_growth(current_level: float, previous_level: float) -> float:
    if not np.isfinite(current_level) or not np.isfinite(previous_level):
        return float("nan")
    if current_level <= 0 or previous_level <= 0:
        return float("nan")
    return float(100.0 * ((current_level / previous_level) ** 4 - 1.0))


def _latest_available_asof(row: np.ndarray, upto_idx: int) -> float:
    if upto_idx < 0:
        return float("nan")
    upto = row[: upto_idx + 1]
    valid_idx = np.flatnonzero(~np.isnan(upto))
    if valid_idx.size == 0:
        return float("nan")
    return float(upto[int(valid_idx[-1])])


def build_release_dataset(wide: pd.DataFrame, series: str) -> pd.DataFrame:
    vintage_cols: list[str] = []
    vintage_ts: list[pd.Timestamp] = []

    for col in wide.columns:
        ts = _extract_vintage_date_from_col(col, series=series)
        if ts is not None and not pd.isna(ts):
            vintage_cols.append(col)
            vintage_ts.append(ts)

    if not vintage_cols:
        raise ValueError(f"No vintage columns matched pattern '{series}_YYYYMMDD'.")

    order = np.argsort(np.array(vintage_ts, dtype="datetime64[ns]"))
    vintage_cols = [vintage_cols[i] for i in order]
    vintage_ts = [vintage_ts[i] for i in order]

    obs = pd.to_datetime(wide["observation_date"], errors="coerce")
    values = wide[vintage_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    n = len(wide)
    first_val = np.full(n, np.nan)
    second_val = np.full(n, np.nan)
    third_val = np.full(n, np.nan)
    latest_val = np.full(n, np.nan)
    first_idx = np.full(n, -1, dtype=int)
    second_idx = np.full(n, -1, dtype=int)
    third_idx = np.full(n, -1, dtype=int)

    first_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    second_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    third_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    latest_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    for i in range(n):
        row = values[i, :]
        valid_idx = np.flatnonzero(~np.isnan(row))
        if valid_idx.size == 0:
            continue

        i1 = int(valid_idx[0])
        first_idx[i] = i1
        first_val[i] = row[i1]
        first_date[i] = np.datetime64(vintage_ts[i1])

        if valid_idx.size >= 2:
            i2 = int(valid_idx[1])
            second_idx[i] = i2
            second_val[i] = row[i2]
            second_date[i] = np.datetime64(vintage_ts[i2])

        if valid_idx.size >= 3:
            i3 = int(valid_idx[2])
            third_idx[i] = i3
            third_val[i] = row[i3]
            third_date[i] = np.datetime64(vintage_ts[i3])

        ilast = int(valid_idx[-1])
        latest_val[i] = row[ilast]
        latest_date[i] = np.datetime64(vintage_ts[ilast])

    out = pd.DataFrame(
        {
            "observation_date": obs,
            "first_release_date": pd.to_datetime(first_date),
            "first_release": first_val,
            "second_release_date": pd.to_datetime(second_date),
            "second_release": second_val,
            "third_release_date": pd.to_datetime(third_date),
            "third_release": third_val,
            "latest_release_date": pd.to_datetime(latest_date),
            "latest_release": latest_val,
        }
    )

    out = out.sort_values("observation_date").reset_index(drop=True)

    out["latest_prev_release"] = out["latest_release"].shift(1)
    ratio = out["latest_release"] / out["latest_prev_release"]
    out["qoq_growth_latest_pct"] = (ratio - 1.0) * 100.0
    out["qoq_saar_growth_latest_pct"] = (ratio.pow(4) - 1.0) * 100.0

    # Realtime growth: compare current-quarter release level with previous-quarter
    # level as-of the same release vintage date.
    qoq_saar_growth_realtime_first_pct = np.full(n, np.nan)
    qoq_saar_growth_realtime_second_pct = np.full(n, np.nan)
    qoq_saar_growth_realtime_third_pct = np.full(n, np.nan)

    for i in range(1, n):
        prev_row = values[i - 1, :]

        if first_idx[i] >= 0:
            prev_level_first = _latest_available_asof(prev_row, int(first_idx[i]))
            qoq_saar_growth_realtime_first_pct[i] = _compute_saar_growth(first_val[i], prev_level_first)

        if second_idx[i] >= 0:
            prev_level_second = _latest_available_asof(prev_row, int(second_idx[i]))
            qoq_saar_growth_realtime_second_pct[i] = _compute_saar_growth(second_val[i], prev_level_second)

        if third_idx[i] >= 0:
            prev_level_third = _latest_available_asof(prev_row, int(third_idx[i]))
            qoq_saar_growth_realtime_third_pct[i] = _compute_saar_growth(third_val[i], prev_level_third)

    out["qoq_saar_growth_realtime_first_pct"] = qoq_saar_growth_realtime_first_pct
    out["qoq_saar_growth_realtime_second_pct"] = qoq_saar_growth_realtime_second_pct
    out["qoq_saar_growth_realtime_third_pct"] = qoq_saar_growth_realtime_third_pct

    # Gaps make vintage-history limitations explicit (e.g., very old observations).
    out["first_release_lag_days"] = (out["first_release_date"] - out["observation_date"]).dt.days
    out["second_release_lag_days"] = (out["second_release_date"] - out["observation_date"]).dt.days
    out["third_release_lag_days"] = (out["third_release_date"] - out["observation_date"]).dt.days

    return out


def main() -> int:
    args = parse_args()
    series = args.series.upper()

    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        print(f"Fetching ALFRED download form for {series}...")
        html = fetch_download_form(session=session, series=series, timeout=args.timeout)

        default_obs_start, default_obs_end, vintage_dates = parse_download_form(html)

        obs_start = args.obs_start or default_obs_start
        obs_end = args.obs_end or default_obs_end

        print(
            "Parsed form defaults: "
            f"obs_start={default_obs_start}, obs_end={default_obs_end}, "
            f"vintages={len(vintage_dates)}"
        )
        print(
            "Requesting zip payload: "
            f"series={series}, obs_start={obs_start}, obs_end={obs_end}, "
            f"selected_vintages={len(vintage_dates)}"
        )

        zip_bytes = download_vintage_zip(
            session=session,
            series=series,
            obs_start=obs_start,
            obs_end=obs_end,
            vintage_dates=vintage_dates,
            timeout=args.timeout,
        )

    wide = load_wide_vintage_csv(zip_bytes)
    releases = build_release_dataset(wide=wide, series=series)

    releases.to_csv(out_csv, index=False)
    print(f"Wrote CSV: {out_csv} ({len(releases)} rows)")

    if args.output_parquet:
        out_parquet = Path(args.output_parquet).expanduser().resolve()
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        try:
            releases.to_parquet(out_parquet, index=False)
            print(f"Wrote parquet: {out_parquet} ({len(releases)} rows)")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to write parquet to {out_parquet}: {exc}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
