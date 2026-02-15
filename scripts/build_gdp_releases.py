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
DEFAULT_QD_VINTAGE_PANEL_PATH = "data/panels/fred_qd_vintage_panel.parquet"


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
    parser.add_argument(
        "--vintage_panel_path",
        default=DEFAULT_QD_VINTAGE_PANEL_PATH,
        help=(
            "Path to FRED-QD vintage panel parquet used to compute realtime q/q SAAR growth "
            "from as-of GDPC1 levels."
        ),
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


def load_qd_vintage_series_panel(panel_path: str | Path, series: str) -> pd.DataFrame:
    panel_path = Path(panel_path).expanduser().resolve()
    if not panel_path.exists():
        raise FileNotFoundError(
            "QD vintage panel not found: "
            f"{panel_path}. Build it with `make panel-qd` or provide --vintage_panel_path."
        )

    panel = pd.read_parquet(panel_path)
    required = {"vintage", "vintage_timestamp", "timestamp", series}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"QD vintage panel missing required columns {missing}: {panel_path}")

    out = panel[["vintage", "timestamp", series, "vintage_timestamp"]].copy()
    out["vintage"] = out["vintage"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out[series], errors="coerce")
    out["vintage_timestamp"] = pd.to_datetime(out["vintage_timestamp"], errors="coerce")
    out["quarter"] = pd.PeriodIndex(out["timestamp"], freq="Q-DEC")
    out["quarter_ord"] = out["quarter"].astype("int64")

    out = out.dropna(subset=["timestamp", "vintage", "value", "vintage_timestamp"]).copy()
    out = out.sort_values(["vintage_timestamp", "vintage", "timestamp"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["vintage_timestamp", "quarter_ord"], keep="last")
    return out[["vintage", "vintage_timestamp", "quarter", "quarter_ord", "value"]]


def _compute_realtime_saar_from_panel(
    out: pd.DataFrame,
    qd_panel: pd.DataFrame,
) -> pd.DataFrame:
    if qd_panel.empty:
        out["qoq_growth_realtime_first_pct"] = np.nan
        out["qoq_growth_realtime_second_pct"] = np.nan
        out["qoq_growth_realtime_third_pct"] = np.nan
        out["qoq_saar_growth_realtime_first_pct"] = np.nan
        out["qoq_saar_growth_realtime_second_pct"] = np.nan
        out["qoq_saar_growth_realtime_third_pct"] = np.nan
        return out

    release_long = out[
        [
            "observation_date",
            "first_release_date",
            "second_release_date",
            "third_release_date",
        ]
    ].copy()
    release_long = release_long.melt(
        id_vars=["observation_date"],
        value_vars=["first_release_date", "second_release_date", "third_release_date"],
        var_name="release_stage_col",
        value_name="asof_date",
    )
    stage_map = {
        "first_release_date": "first",
        "second_release_date": "second",
        "third_release_date": "third",
    }
    release_long["stage"] = release_long["release_stage_col"].map(stage_map)
    release_long["observation_date"] = pd.to_datetime(release_long["observation_date"], errors="coerce")
    release_long["asof_date"] = pd.to_datetime(release_long["asof_date"], errors="coerce")
    release_long["quarter"] = pd.PeriodIndex(release_long["observation_date"], freq="Q-DEC")
    release_long["quarter_ord"] = release_long["quarter"].astype("int64")
    release_long["prev_quarter_ord"] = release_long["quarter_ord"] - 1
    release_long = release_long.sort_values("asof_date").reset_index(drop=True)

    vintage_calendar = (
        qd_panel[["vintage_timestamp"]]
        .drop_duplicates()
        .sort_values("vintage_timestamp")
        .reset_index(drop=True)
    )
    release_long_valid = release_long.loc[release_long["asof_date"].notna()].copy()
    release_long_missing = release_long.loc[release_long["asof_date"].isna()].copy()
    release_long_missing["vintage_timestamp"] = pd.NaT

    if not release_long_valid.empty:
        release_long_valid = pd.merge_asof(
            release_long_valid,
            vintage_calendar,
            left_on="asof_date",
            right_on="vintage_timestamp",
            direction="backward",
        )

    release_long = pd.concat([release_long_valid, release_long_missing], axis=0, ignore_index=True)

    panel_cur = qd_panel[["vintage_timestamp", "quarter_ord", "value"]].rename(columns={"value": "y_q"})
    release_long = release_long.merge(
        panel_cur,
        on=["vintage_timestamp", "quarter_ord"],
        how="left",
    )

    panel_prev = qd_panel[["vintage_timestamp", "quarter_ord", "value"]].rename(
        columns={"quarter_ord": "prev_quarter_ord", "value": "y_qm1"}
    )
    release_long = release_long.merge(
        panel_prev,
        on=["vintage_timestamp", "prev_quarter_ord"],
        how="left",
    )

    valid = (
        release_long["asof_date"].notna()
        & release_long["vintage_timestamp"].notna()
        & release_long["y_q"].notna()
        & release_long["y_qm1"].notna()
        & np.isfinite(release_long["y_q"])
        & np.isfinite(release_long["y_qm1"])
        & (release_long["y_q"] > 0)
        & (release_long["y_qm1"] > 0)
    )
    ratio = release_long["y_q"] / release_long["y_qm1"]
    release_long["qoq_growth_pct"] = np.where(valid, (ratio - 1.0) * 100.0, np.nan)
    release_long["qoq_saar_growth_pct"] = np.where(valid, (ratio.pow(4) - 1.0) * 100.0, np.nan)

    qoq_pivot = (
        release_long.pivot_table(index="observation_date", columns="stage", values="qoq_growth_pct", aggfunc="first")
        .rename(
            columns={
                "first": "qoq_growth_realtime_first_pct",
                "second": "qoq_growth_realtime_second_pct",
                "third": "qoq_growth_realtime_third_pct",
            }
        )
        .reset_index()
    )
    saar_pivot = (
        release_long.pivot_table(index="observation_date", columns="stage", values="qoq_saar_growth_pct", aggfunc="first")
        .rename(
            columns={
                "first": "qoq_saar_growth_realtime_first_pct",
                "second": "qoq_saar_growth_realtime_second_pct",
                "third": "qoq_saar_growth_realtime_third_pct",
            }
        )
        .reset_index()
    )

    out = out.merge(qoq_pivot, on="observation_date", how="left")
    out = out.merge(saar_pivot, on="observation_date", how="left")

    for col in [
        "qoq_growth_realtime_first_pct",
        "qoq_growth_realtime_second_pct",
        "qoq_growth_realtime_third_pct",
        "qoq_saar_growth_realtime_first_pct",
        "qoq_saar_growth_realtime_second_pct",
        "qoq_saar_growth_realtime_third_pct",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_release_dataset(wide: pd.DataFrame, series: str, qd_panel: pd.DataFrame | None = None) -> pd.DataFrame:
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
        first_val[i] = row[i1]
        first_date[i] = np.datetime64(vintage_ts[i1])

        if valid_idx.size >= 2:
            i2 = int(valid_idx[1])
            second_val[i] = row[i2]
            second_date[i] = np.datetime64(vintage_ts[i2])

        if valid_idx.size >= 3:
            i3 = int(valid_idx[2])
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

    # Realtime growth uses GDPC1 levels from the selected as-of vintage panel
    # to avoid mixing vintages around level breaks.
    if qd_panel is None:
        out["qoq_growth_realtime_first_pct"] = np.nan
        out["qoq_growth_realtime_second_pct"] = np.nan
        out["qoq_growth_realtime_third_pct"] = np.nan
        out["qoq_saar_growth_realtime_first_pct"] = np.nan
        out["qoq_saar_growth_realtime_second_pct"] = np.nan
        out["qoq_saar_growth_realtime_third_pct"] = np.nan
    else:
        out = _compute_realtime_saar_from_panel(out=out, qd_panel=qd_panel)

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
    print(f"Loading QD vintage panel: {args.vintage_panel_path}")
    qd_panel = load_qd_vintage_series_panel(panel_path=args.vintage_panel_path, series=series)
    releases = build_release_dataset(wide=wide, series=series, qd_panel=qd_panel)

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
