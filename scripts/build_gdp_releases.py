#!/usr/bin/env python3
"""Build first/second/third GDPC1 release dataset from ALFRED.

The script downloads ALFRED vintage data for a series (default: GDPC1), extracts
first/second/third available releases per observation date, and computes ALFRED
stage q/q SAAR growth (`qoq_saar_growth_alfred_*`) using same-vintage q and q-1
levels. The written release file is intentionally KPI-focused: it keeps ALFRED
stage SAAR columns plus level inputs needed to reconstruct those growth values.
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
DEFAULT_VALIDATION_REPORT_PATH = "data/panels/gdpc1_release_validation_report.csv"
STAGES = ("first", "second", "third")
STAGE_RELEASE_DATE_COLS = {
    "first": "first_release_date",
    "second": "second_release_date",
    "third": "third_release_date",
}
STAGE_RELEASE_LEVEL_COLS = {
    "first": "first_release",
    "second": "second_release",
    "third": "third_release",
}
STAGE_QOQ_COLS = {
    "first": "qoq_growth_realtime_first_pct",
    "second": "qoq_growth_realtime_second_pct",
    "third": "qoq_growth_realtime_third_pct",
}
STAGE_QOQ_SAAR_COLS = {
    "first": "qoq_saar_growth_realtime_first_pct",
    "second": "qoq_saar_growth_realtime_second_pct",
    "third": "qoq_saar_growth_realtime_third_pct",
}
STAGE_QOQ_ALFRED_COLS = {
    "first": "qoq_growth_alfred_first_pct",
    "second": "qoq_growth_alfred_second_pct",
    "third": "qoq_growth_alfred_third_pct",
}
STAGE_QOQ_SAAR_ALFRED_COLS = {
    "first": "qoq_saar_growth_alfred_first_pct",
    "second": "qoq_saar_growth_alfred_second_pct",
    "third": "qoq_saar_growth_alfred_third_pct",
}
STAGE_ALFRED_PREV_LEVEL_COLS = {
    "first": "alfred_prev_level_first",
    "second": "alfred_prev_level_second",
    "third": "alfred_prev_level_third",
}
DEFAULT_SPIKE_SHOCK_WHITELIST = {"2020Q2", "2020Q3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ALFRED vintage data and build first/second/third release "
            "dataset with ALFRED q/q SAAR KPI columns and supporting ALFRED levels."
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
    parser.add_argument(
        "--vintage_select",
        choices=["next", "prev"],
        default="next",
        help=(
            "Map release date to panel vintage with either the earliest vintage at/after "
            "the date ('next') or the latest vintage at/before the date ('prev')."
        ),
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run release-construction validation checks (default: enabled).",
    )
    parser.add_argument(
        "--fail_on_validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit code 2 when validation finds spike flags.",
    )
    parser.add_argument(
        "--validation_report_csv",
        default=DEFAULT_VALIDATION_REPORT_PATH,
        help="Validation report CSV path written when --validate is enabled.",
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

    try:
        panel = pd.read_parquet(panel_path)
    except ImportError as exc:
        raise RuntimeError(
            "Unable to read parquet panel because no parquet engine is installed. "
            "Install one with `pip install pyarrow` (recommended) or `pip install fastparquet`."
        ) from exc

    required = {"timestamp", series}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"QD vintage panel missing required columns {missing}: {panel_path}")

    if "vintage_timestamp" not in panel.columns and "vintage" not in panel.columns:
        raise ValueError(
            "QD vintage panel must include either 'vintage_timestamp' or 'vintage' (YYYY-MM): "
            f"{panel_path}"
        )

    out = pd.DataFrame(index=panel.index)
    out["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(panel[series], errors="coerce")

    if "vintage_timestamp" in panel.columns:
        out["vintage_timestamp"] = pd.to_datetime(panel["vintage_timestamp"], errors="coerce")
    else:
        out["vintage_timestamp"] = pd.to_datetime(
            panel["vintage"].astype(str) + "-01",
            errors="coerce",
        )

    if "vintage" in panel.columns:
        out["vintage"] = panel["vintage"].astype(str)
    else:
        out["vintage"] = out["vintage_timestamp"].dt.to_period("M").astype(str)
    out["vintage"] = out["vintage"].replace({"nan": np.nan, "NaT": np.nan})

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["vintage_timestamp"] = pd.to_datetime(out["vintage_timestamp"], errors="coerce")
    out["quarter"] = pd.PeriodIndex(out["timestamp"], freq="Q-DEC")
    out["quarter_ord"] = out["quarter"].astype("int64")

    out = out.dropna(subset=["timestamp", "vintage", "value", "vintage_timestamp"]).copy()
    out = out.sort_values(["vintage_timestamp", "vintage", "timestamp"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["vintage_timestamp", "quarter_ord"], keep="last")
    return out[["vintage", "vintage_timestamp", "quarter", "quarter_ord", "value"]]


def select_panel_vintage(
    release_date: pd.Timestamp | str | np.datetime64 | None,
    panel_vintage_timestamps: pd.DatetimeIndex,
    vintage_select: str = "next",
) -> pd.Timestamp | None:
    if vintage_select not in {"next", "prev"}:
        raise ValueError(f"Unsupported vintage_select={vintage_select!r}; expected one of ['next', 'prev'].")
    if release_date is None or pd.isna(release_date):
        return pd.NaT
    if len(panel_vintage_timestamps) == 0:
        return pd.NaT

    release_ts = pd.Timestamp(release_date)
    if pd.isna(release_ts):
        return pd.NaT

    if vintage_select == "next":
        idx = int(panel_vintage_timestamps.searchsorted(release_ts, side="left"))
        if idx >= len(panel_vintage_timestamps):
            return pd.NaT
        return pd.Timestamp(panel_vintage_timestamps[idx])

    idx = int(panel_vintage_timestamps.searchsorted(release_ts, side="right")) - 1
    if idx < 0:
        return pd.NaT
    return pd.Timestamp(panel_vintage_timestamps[idx])


def _build_realtime_stage_table(
    releases_df: pd.DataFrame,
    qd_panel: pd.DataFrame,
    vintage_select: str,
) -> pd.DataFrame:
    required_cols = {"observation_date", *STAGE_RELEASE_DATE_COLS.values(), *STAGE_RELEASE_LEVEL_COLS.values()}
    missing_release_cols = sorted(required_cols.difference(releases_df.columns))
    if missing_release_cols:
        raise ValueError(f"Release table missing columns required for realtime growth: {missing_release_cols}")

    obs_dates = pd.to_datetime(releases_df["observation_date"], errors="coerce")
    quarter = pd.PeriodIndex(obs_dates, freq="Q-DEC")
    quarter_ord = quarter.astype("int64")
    base = pd.DataFrame(
        {
            "observation_date": obs_dates,
            "quarter": quarter.astype(str),
            "quarter_ord": quarter_ord,
            "prev_quarter_ord": quarter_ord - 1,
        }
    )

    stage_frames: list[pd.DataFrame] = []
    for stage in STAGES:
        frame = base.copy()
        frame["stage"] = stage
        frame["release_date"] = pd.to_datetime(releases_df[STAGE_RELEASE_DATE_COLS[stage]], errors="coerce")
        frame["stage_release_level"] = pd.to_numeric(releases_df[STAGE_RELEASE_LEVEL_COLS[stage]], errors="coerce")
        stage_frames.append(frame)

    stage_table = pd.concat(stage_frames, axis=0, ignore_index=True)
    stage_table = stage_table.sort_values(["observation_date", "stage"]).reset_index(drop=True)

    if qd_panel.empty:
        stage_table["selected_vintage"] = pd.NaT
        stage_table["selected_vintage_label"] = np.nan
        stage_table["panel_y_q"] = np.nan
        stage_table["panel_y_qm1"] = np.nan
        stage_table["qoq_growth_pct"] = np.nan
        stage_table["qoq_saar_growth_pct"] = np.nan
        stage_table["panel_to_release_ratio"] = np.nan
        return stage_table

    panel = qd_panel.copy()
    expected_panel_cols = {"vintage_timestamp", "quarter_ord", "value"}
    missing_panel_cols = sorted(expected_panel_cols.difference(panel.columns))
    if missing_panel_cols:
        raise ValueError(f"QD panel missing required columns {missing_panel_cols} for realtime growth.")

    panel["vintage_timestamp"] = pd.to_datetime(panel["vintage_timestamp"], errors="coerce")
    panel["quarter_ord"] = pd.to_numeric(panel["quarter_ord"], errors="coerce")
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    if "vintage" not in panel.columns:
        panel["vintage"] = panel["vintage_timestamp"].dt.to_period("M").astype(str)

    panel = panel.dropna(subset=["vintage_timestamp", "quarter_ord", "value"]).copy()
    panel["quarter_ord"] = panel["quarter_ord"].astype("int64")
    panel = panel.sort_values(["vintage_timestamp", "quarter_ord"]).reset_index(drop=True)
    panel = panel.drop_duplicates(subset=["vintage_timestamp", "quarter_ord"], keep="last")

    panel_vintage_timestamps = pd.DatetimeIndex(
        panel["vintage_timestamp"].drop_duplicates().sort_values().to_numpy(dtype="datetime64[ns]")
    )
    stage_table["selected_vintage"] = stage_table["release_date"].map(
        lambda d: select_panel_vintage(
            release_date=d,
            panel_vintage_timestamps=panel_vintage_timestamps,
            vintage_select=vintage_select,
        )
    )
    stage_table["selected_vintage"] = pd.to_datetime(stage_table["selected_vintage"], errors="coerce")

    vintage_labels = (
        panel[["vintage_timestamp", "vintage"]]
        .drop_duplicates(subset=["vintage_timestamp"], keep="last")
        .rename(columns={"vintage": "selected_vintage_label"})
    )
    stage_table = stage_table.merge(
        vintage_labels,
        left_on="selected_vintage",
        right_on="vintage_timestamp",
        how="left",
    ).drop(columns=["vintage_timestamp"], errors="ignore")

    panel_cur = panel[["vintage_timestamp", "quarter_ord", "value"]].rename(
        columns={
            "vintage_timestamp": "selected_vintage",
            "value": "panel_y_q",
        }
    )
    stage_table = stage_table.merge(panel_cur, on=["selected_vintage", "quarter_ord"], how="left")

    panel_prev = panel[["vintage_timestamp", "quarter_ord", "value"]].rename(
        columns={
            "vintage_timestamp": "selected_vintage",
            "quarter_ord": "prev_quarter_ord",
            "value": "panel_y_qm1",
        }
    )
    stage_table = stage_table.merge(panel_prev, on=["selected_vintage", "prev_quarter_ord"], how="left")

    valid = (
        stage_table["selected_vintage"].notna()
        & stage_table["panel_y_q"].notna()
        & stage_table["panel_y_qm1"].notna()
        & np.isfinite(stage_table["panel_y_q"])
        & np.isfinite(stage_table["panel_y_qm1"])
        & (stage_table["panel_y_q"] > 0)
        & (stage_table["panel_y_qm1"] > 0)
    )
    ratio = stage_table["panel_y_q"] / stage_table["panel_y_qm1"]
    stage_table["qoq_growth_pct"] = np.where(valid, (ratio - 1.0) * 100.0, np.nan)
    stage_table["qoq_saar_growth_pct"] = np.where(valid, (ratio.pow(4) - 1.0) * 100.0, np.nan)

    level_ratio_valid = (
        stage_table["stage_release_level"].notna()
        & np.isfinite(stage_table["stage_release_level"])
        & (stage_table["stage_release_level"] != 0.0)
        & stage_table["panel_y_q"].notna()
        & np.isfinite(stage_table["panel_y_q"])
    )
    stage_table["panel_to_release_ratio"] = np.where(
        level_ratio_valid,
        stage_table["panel_y_q"] / stage_table["stage_release_level"],
        np.nan,
    )

    return stage_table


def _compute_realtime_saar_from_panel(
    out: pd.DataFrame,
    qd_panel: pd.DataFrame,
    vintage_select: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_table = _build_realtime_stage_table(
        releases_df=out,
        qd_panel=qd_panel,
        vintage_select=vintage_select,
    )

    out = out.copy()
    out = out.drop(columns=[*STAGE_QOQ_COLS.values(), *STAGE_QOQ_SAAR_COLS.values()], errors="ignore")

    qoq_pivot = (
        stage_table.pivot_table(index="observation_date", columns="stage", values="qoq_growth_pct", aggfunc="first")
        .rename(columns=STAGE_QOQ_COLS)
        .reset_index()
    )
    saar_pivot = (
        stage_table.pivot_table(index="observation_date", columns="stage", values="qoq_saar_growth_pct", aggfunc="first")
        .rename(columns=STAGE_QOQ_SAAR_COLS)
        .reset_index()
    )

    out = out.merge(qoq_pivot, on="observation_date", how="left")
    out = out.merge(saar_pivot, on="observation_date", how="left")

    for col in [*STAGE_QOQ_COLS.values(), *STAGE_QOQ_SAAR_COLS.values()]:
        if col not in out.columns:
            out[col] = np.nan

    return out, stage_table


def _compute_alfred_stage_growth_from_wide(
    out: pd.DataFrame,
    wide: pd.DataFrame,
    vintage_cols: list[str],
    vintage_ts: list[pd.Timestamp],
) -> pd.DataFrame:
    """Compute stage growth from ALFRED release vintages using same-vintage q and q-1 levels."""
    out = out.copy()
    growth_cols = [*STAGE_QOQ_ALFRED_COLS.values(), *STAGE_QOQ_SAAR_ALFRED_COLS.values()]
    prev_level_cols = [*STAGE_ALFRED_PREV_LEVEL_COLS.values()]
    out = out.drop(columns=growth_cols, errors="ignore")
    out = out.drop(columns=prev_level_cols, errors="ignore")

    if not vintage_cols:
        for col in growth_cols:
            out[col] = np.nan
        for col in prev_level_cols:
            out[col] = np.nan
        return out

    wide_obs = pd.to_datetime(wide["observation_date"], errors="coerce")
    levels = wide[vintage_cols].apply(pd.to_numeric, errors="coerce")
    levels.columns = pd.to_datetime(pd.Index(vintage_ts), errors="coerce")
    levels.insert(0, "observation_date", wide_obs)
    levels = levels.dropna(subset=["observation_date"]).copy()

    if levels.empty:
        for col in growth_cols:
            out[col] = np.nan
        for col in prev_level_cols:
            out[col] = np.nan
        return out

    levels["quarter_ord"] = pd.PeriodIndex(levels["observation_date"], freq="Q-DEC").astype("int64")
    levels = (
        levels.sort_values(["quarter_ord", "observation_date"])
        .drop_duplicates(subset=["quarter_ord"], keep="last")
        .set_index("quarter_ord")
        .drop(columns=["observation_date"])
    )

    out_quarter_ord = pd.PeriodIndex(pd.to_datetime(out["observation_date"], errors="coerce"), freq="Q-DEC").astype("int64")
    cur_levels = levels.reindex(out_quarter_ord)
    prev_levels = levels.reindex(out_quarter_ord - 1)

    cur_arr = cur_levels.to_numpy(dtype=float)
    prev_arr = prev_levels.to_numpy(dtype=float)
    col_to_pos = {pd.Timestamp(col): idx for idx, col in enumerate(cur_levels.columns)}
    row_idx = np.arange(len(out), dtype=int)

    for stage in STAGES:
        release_dates = pd.to_datetime(out[STAGE_RELEASE_DATE_COLS[stage]], errors="coerce")
        col_idx = (
            release_dates.map(
                lambda d: col_to_pos.get(pd.Timestamp(d), -1) if pd.notna(d) else -1
            )
            .astype(int)
            .to_numpy()
        )
        has_col = col_idx >= 0

        y_q = np.full(len(out), np.nan, dtype=float)
        y_qm1 = np.full(len(out), np.nan, dtype=float)
        y_q[has_col] = cur_arr[row_idx[has_col], col_idx[has_col]]
        y_qm1[has_col] = prev_arr[row_idx[has_col], col_idx[has_col]]

        valid = np.isfinite(y_q) & np.isfinite(y_qm1) & (y_q > 0.0) & (y_qm1 > 0.0)
        ratio = np.where(valid, y_q / y_qm1, np.nan)
        out[STAGE_QOQ_ALFRED_COLS[stage]] = np.where(valid, (ratio - 1.0) * 100.0, np.nan)
        out[STAGE_QOQ_SAAR_ALFRED_COLS[stage]] = np.where(valid, (ratio**4 - 1.0) * 100.0, np.nan)
        out[STAGE_ALFRED_PREV_LEVEL_COLS[stage]] = y_qm1

    for col in growth_cols:
        if col not in out.columns:
            out[col] = np.nan
    for col in prev_level_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _format_release_output_for_alfred_qoq_saar(releases_df: pd.DataFrame) -> pd.DataFrame:
    """Keep ALFRED SAAR KPI columns and ALFRED levels used to construct them."""
    out = releases_df.copy()

    # Backfill missing previous-quarter ALFRED levels from stage level + SAAR growth when needed.
    for stage in STAGES:
        prev_col = STAGE_ALFRED_PREV_LEVEL_COLS[stage]
        if prev_col in out.columns:
            continue
        level_col = STAGE_RELEASE_LEVEL_COLS[stage]
        saar_col = STAGE_QOQ_SAAR_ALFRED_COLS[stage]
        if level_col not in out.columns or saar_col not in out.columns:
            out[prev_col] = np.nan
            continue
        y_q = pd.to_numeric(out[level_col], errors="coerce")
        g_saar = pd.to_numeric(out[saar_col], errors="coerce")
        base = 1.0 + (g_saar / 100.0)
        ratio = np.where(base > 0.0, np.power(base, 0.25), np.nan)
        valid = (
            np.isfinite(y_q.to_numpy(dtype=float))
            & np.isfinite(ratio)
            & (ratio != 0.0)
        )
        prev = np.full(len(out), np.nan, dtype=float)
        y_q_arr = y_q.to_numpy(dtype=float)
        prev[valid] = y_q_arr[valid] / ratio[valid]
        out[prev_col] = prev

    # Remove all qoq/qoq_saar derivatives except ALFRED stage SAAR KPI columns.
    keep_growth = set(STAGE_QOQ_SAAR_ALFRED_COLS.values())
    drop_growth = [col for col in out.columns if col.startswith("qoq_") and col not in keep_growth]
    out = out.drop(columns=drop_growth, errors="ignore")

    ordered_cols = [
        "observation_date",
        "first_release_date",
        "first_release",
        STAGE_ALFRED_PREV_LEVEL_COLS["first"],
        STAGE_QOQ_SAAR_ALFRED_COLS["first"],
        "second_release_date",
        "second_release",
        STAGE_ALFRED_PREV_LEVEL_COLS["second"],
        STAGE_QOQ_SAAR_ALFRED_COLS["second"],
        "third_release_date",
        "third_release",
        STAGE_ALFRED_PREV_LEVEL_COLS["third"],
        STAGE_QOQ_SAAR_ALFRED_COLS["third"],
        "latest_release_date",
        "latest_release",
        "latest_prev_release",
        "first_release_lag_days",
        "second_release_lag_days",
        "third_release_lag_days",
    ]
    present_cols = [col for col in ordered_cols if col in out.columns]
    return out[present_cols].copy()


def build_release_dataset(
    wide: pd.DataFrame,
    series: str,
    qd_panel: pd.DataFrame | None = None,
    vintage_select: str = "next",
) -> pd.DataFrame:
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

    out = _compute_alfred_stage_growth_from_wide(
        out=out,
        wide=wide,
        vintage_cols=vintage_cols,
        vintage_ts=vintage_ts,
    )

    # Realtime growth uses GDPC1 levels from the selected as-of vintage panel
    # to avoid mixing vintages around level breaks.
    if qd_panel is None:
        for col in [*STAGE_QOQ_COLS.values(), *STAGE_QOQ_SAAR_COLS.values()]:
            out[col] = np.nan
    else:
        out, _ = _compute_realtime_saar_from_panel(
            out=out,
            qd_panel=qd_panel,
            vintage_select=vintage_select,
        )

    # Gaps make vintage-history limitations explicit (e.g., very old observations).
    out["first_release_lag_days"] = (out["first_release_date"] - out["observation_date"]).dt.days
    out["second_release_lag_days"] = (out["second_release_date"] - out["observation_date"]).dt.days
    out["third_release_lag_days"] = (out["third_release_date"] - out["observation_date"]).dt.days

    return out


def validate_release_table(
    releases_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    vintage_select: str = "next",
    report_path: str | Path = DEFAULT_VALIDATION_REPORT_PATH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stage_table = _build_realtime_stage_table(
        releases_df=releases_df,
        qd_panel=panel_df,
        vintage_select=vintage_select,
    ).copy()

    stage_table["observation_date"] = pd.to_datetime(stage_table["observation_date"], errors="coerce")
    stage_table["release_date"] = pd.to_datetime(stage_table["release_date"], errors="coerce")
    stage_table["selected_vintage"] = pd.to_datetime(stage_table["selected_vintage"], errors="coerce")
    quarter_period = pd.PeriodIndex(stage_table["observation_date"], freq="Q-DEC")
    stage_table["quarter"] = quarter_period.astype(str)
    stage_table["prev_quarter"] = (quarter_period - 1).astype(str)

    stage_table = stage_table.sort_values(["stage", "quarter_ord"]).reset_index(drop=True)
    stage_table["prev_growth_pct"] = stage_table.groupby("stage", dropna=False)["qoq_saar_growth_pct"].shift(1)
    stage_table["growth_delta_pct"] = stage_table["qoq_saar_growth_pct"] - stage_table["prev_growth_pct"]

    from_2010 = stage_table["observation_date"] >= pd.Timestamp("2010-01-01")
    modern = stage_table["observation_date"] >= pd.Timestamp("2018-01-01")
    is_shock_quarter = stage_table["quarter"].isin(DEFAULT_SPIKE_SHOCK_WHITELIST)
    prev_is_shock_quarter = stage_table["prev_quarter"].isin(DEFAULT_SPIKE_SHOCK_WHITELIST)

    has_stage_release = stage_table["stage_release_level"].notna() & np.isfinite(stage_table["stage_release_level"])
    panel_ranges = (
        panel_df[["vintage_timestamp", "quarter_ord"]]
        .assign(
            vintage_timestamp=lambda df_: pd.to_datetime(df_["vintage_timestamp"], errors="coerce"),
            quarter_ord=lambda df_: pd.to_numeric(df_["quarter_ord"], errors="coerce"),
        )
        .dropna(subset=["vintage_timestamp", "quarter_ord"])
        .groupby("vintage_timestamp", as_index=False)["quarter_ord"]
        .agg(panel_min_quarter_ord="min", panel_max_quarter_ord="max")
    )
    stage_table = stage_table.merge(
        panel_ranges,
        left_on="selected_vintage",
        right_on="vintage_timestamp",
        how="left",
    ).drop(columns=["vintage_timestamp"], errors="ignore")
    within_q_span = (
        stage_table["selected_vintage"].notna()
        & stage_table["panel_min_quarter_ord"].notna()
        & (stage_table["quarter_ord"] >= stage_table["panel_min_quarter_ord"])
        & (stage_table["quarter_ord"] <= stage_table["panel_max_quarter_ord"])
    )
    within_prev_q_span = (
        stage_table["selected_vintage"].notna()
        & stage_table["panel_min_quarter_ord"].notna()
        & (stage_table["prev_quarter_ord"] >= stage_table["panel_min_quarter_ord"])
        & (stage_table["prev_quarter_ord"] <= stage_table["panel_max_quarter_ord"])
    )
    spike_abs = (
        from_2010
        & stage_table["qoq_saar_growth_pct"].notna()
        & (stage_table["qoq_saar_growth_pct"].abs() > 15.0)
        & ~is_shock_quarter
    )
    spike_delta = (
        from_2010
        & stage_table["growth_delta_pct"].notna()
        & (stage_table["growth_delta_pct"].abs() > 12.0)
        & ~(is_shock_quarter | prev_is_shock_quarter)
    )
    ratio_mismatch = (
        modern
        & has_stage_release
        & stage_table["panel_to_release_ratio"].notna()
        & (
            (stage_table["panel_to_release_ratio"] < 0.98)
            | (stage_table["panel_to_release_ratio"] > 1.02)
        )
    )
    missing_y_q = has_stage_release & within_q_span & stage_table["panel_y_q"].isna()
    missing_y_qm1 = has_stage_release & within_prev_q_span & stage_table["panel_y_qm1"].isna()

    report_columns = [
        "flag_type",
        "quarter",
        "stage",
        "observation_date",
        "release_date",
        "selected_vintage",
        "selected_vintage_label",
        "qoq_saar_growth_pct",
        "growth_delta_pct",
        "panel_y_q",
        "panel_y_qm1",
        "stage_release_level",
        "panel_to_release_ratio",
    ]

    flagged_frames: list[pd.DataFrame] = []
    for mask, flag_type in [
        (spike_abs, "spike_abs_gt_15"),
        (spike_delta, "spike_delta_gt_12"),
        (ratio_mismatch, "ratio_outside_0p98_1p02"),
        (missing_y_q, "missing_panel_y_q"),
        (missing_y_qm1, "missing_panel_y_qm1"),
    ]:
        if not bool(mask.any()):
            continue
        flagged = stage_table.loc[mask, report_columns[1:]].copy()
        flagged.insert(0, "flag_type", flag_type)
        flagged_frames.append(flagged)

    if flagged_frames:
        report_df = pd.concat(flagged_frames, axis=0, ignore_index=True)
        report_df = report_df.sort_values(
            ["flag_type", "quarter", "stage", "observation_date"],
            na_position="last",
        ).reset_index(drop=True)
    else:
        report_df = pd.DataFrame(columns=report_columns)

    report_path = Path(report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    spike_rows = stage_table.loc[
        spike_abs | spike_delta,
        [
            "quarter",
            "stage",
            "release_date",
            "selected_vintage",
            "qoq_saar_growth_pct",
        ],
    ].copy()
    spike_flags = int(len(spike_rows))
    ratio_flags = int((report_df["flag_type"] == "ratio_outside_0p98_1p02").sum())
    coverage_flags = int(report_df["flag_type"].isin({"missing_panel_y_q", "missing_panel_y_qm1"}).sum())

    if spike_rows.empty:
        print("0 flagged spikes; worst offenders: none")
    else:
        offenders = (
            spike_rows.assign(abs_growth=spike_rows["qoq_saar_growth_pct"].abs())
            .sort_values("abs_growth", ascending=False)
            .drop_duplicates(subset=["quarter", "stage"], keep="first")
            .head(5)
        )
        offender_parts = []
        for row in offenders.itertuples(index=False):
            g = "nan" if pd.isna(row.qoq_saar_growth_pct) else f"{row.qoq_saar_growth_pct:.2f}%"
            rel = "NaT" if pd.isna(row.release_date) else pd.Timestamp(row.release_date).date().isoformat()
            vint = "NaT" if pd.isna(row.selected_vintage) else pd.Timestamp(row.selected_vintage).date().isoformat()
            offender_parts.append(f"{row.quarter}/{row.stage} (release={rel}, vintage={vint}, g={g})")
        print(f"{spike_flags} flagged spikes; worst offenders: {'; '.join(offender_parts)}")

    print(
        "Validation flags: "
        f"spikes={spike_flags}, ratio_mismatch={ratio_flags}, coverage={coverage_flags}"
    )
    print(f"Wrote validation report: {report_path}")

    return report_df, {
        "spike_flags": spike_flags,
        "ratio_flags": ratio_flags,
        "coverage_flags": coverage_flags,
        "total_flags": int(len(report_df)),
    }


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
    try:
        qd_panel = load_qd_vintage_series_panel(panel_path=args.vintage_panel_path, series=series)
    except Exception as exc:
        print(f"Failed to load QD vintage panel: {exc}", file=sys.stderr)
        return 2

    releases = build_release_dataset(
        wide=wide,
        series=series,
        qd_panel=qd_panel,
        vintage_select=args.vintage_select,
    )

    validation_summary = {"spike_flags": 0}
    if args.validate:
        _, validation_summary = validate_release_table(
            releases_df=releases,
            panel_df=qd_panel,
            vintage_select=args.vintage_select,
            report_path=args.validation_report_csv,
        )
        if args.fail_on_validate and validation_summary["spike_flags"] > 0:
            print(
                "Spike flags detected during validation "
                f"({validation_summary['spike_flags']}); failing due to --fail_on_validate.",
                file=sys.stderr,
            )
            return 2

    export_df = _format_release_output_for_alfred_qoq_saar(releases_df=releases)
    export_df.to_csv(out_csv, index=False)
    print(f"Wrote CSV: {out_csv} ({len(export_df)} rows)")

    if args.output_parquet:
        out_parquet = Path(args.output_parquet).expanduser().resolve()
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        try:
            export_df.to_parquet(out_parquet, index=False)
            print(f"Wrote parquet: {out_parquet} ({len(export_df)} rows)")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to write parquet to {out_parquet}: {exc}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
