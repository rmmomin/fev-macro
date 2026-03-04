#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.asof_store import AsofStore  # noqa: E402
from fev_macro.fred_aliases import candidate_series_ids, dedupe_preserve_order  # noqa: E402
from fev_macro.fred_transforms import apply_fred_transform_codes, extract_fred_transform_codes  # noqa: E402
from fev_macro.realtime_feeds import select_covariate_columns, train_df_to_datasets  # noqa: E402
from fev_macro.realtime_oos import (  # noqa: E402
    apply_model_runtime_options,
    build_origin_datasets,
    build_vintage_calendar,
    load_release_table,
    load_vintage_panel,
    resolve_models,
    select_training_vintage,
    to_saar_growth,
)
from fev_macro.realtime_runner import resolve_qd_panel_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate realtime as-of nowcast trajectory for 2025Q4 GDP q/q SAAR growth "
            "using mixed-frequency multivariate models."
        )
    )
    p.add_argument("--asof_db", type=str, default="data/realtime/asof.duckdb")
    p.add_argument("--release_csv", type=str, default="data/panels/gdpc1_releases_first_second_third.csv")
    p.add_argument("--target_quarter", type=str, default="2025Q4")
    p.add_argument("--start_asof", type=str, default="2026-01-02")
    p.add_argument("--target_col", type=str, default="GDPC1")
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "mixed_freq_dfm_md",
            "mixed_freq_dfm_md_growth",
            "chronos2",
            "chronos2_growth",
            "lstm_multivariate_growth",
        ],
        help="Model names resolved by fev_macro.realtime_oos.resolve_models.",
    )
    p.add_argument(
        "--md_template",
        type=str,
        default="data/historical/md/vintages_1999_2026/2026-01.csv",
        help="MD template CSV used to define variable list and transform codes.",
    )
    p.add_argument("--md_drop_initial_rows", type=int, default=2)
    p.add_argument("--output_dir", type=str, default="results/realtime_asof_2025q4_mixedfreq")
    return p.parse_args()


def _load_md_template_columns_and_codes(path: Path) -> tuple[list[str], dict[str, int]]:
    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        raise ValueError(f"MD template is empty: {path}")

    first_col = str(raw.columns[0])
    variables = [str(c) for c in raw.columns[1:] if str(c).strip() and not str(c).startswith("Unnamed")]
    codes = extract_fred_transform_codes(raw_df=raw, first_col_name=first_col)
    return variables, codes


def _remove_outliers_iqr(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return frame
    out = frame.copy()
    panel = out[columns]
    med = panel.median(axis=0, skipna=True)
    q1 = panel.quantile(0.25, axis=0, numeric_only=True)
    q3 = panel.quantile(0.75, axis=0, numeric_only=True)
    iqr = q3 - q1
    distance = panel.sub(med, axis=1).abs()
    mask = distance.gt(10.0 * iqr, axis=1)
    out.loc[:, columns] = panel.mask(mask)
    return out


def _resolve_md_series_map(store: AsofStore, md_variables: list[str]) -> dict[str, str]:
    alias = store.alias_map(universe="md")
    available = store.available_series_ids()
    mapping: dict[str, str] = {}
    for var in md_variables:
        sid = alias.get(var)
        if sid and sid in available:
            mapping[var] = sid
            continue
        for cand in candidate_series_ids(var):
            if cand in available:
                mapping[var] = cand
                break
    return mapping


def _query_release_dates_for_md_series(
    store: AsofStore,
    *,
    series_ids: list[str],
    start_date: pd.Timestamp,
    end_date_inclusive: pd.Timestamp,
) -> list[pd.Timestamp]:
    if not series_ids:
        return []
    filt = pd.DataFrame({"series_id": series_ids})
    con = store._con  # noqa: SLF001 - controlled internal usage in script
    con.register("series_filter", filt)
    try:
        out = con.execute(
            """
            SELECT DISTINCT CAST(o.asof_ts AS DATE) AS asof_date
            FROM asof_observations o
            JOIN series_filter f ON o.series_id = f.series_id
            WHERE o.asof_ts >= ? AND o.asof_ts < ?
            ORDER BY asof_date;
            """,
            [
                pd.Timestamp(start_date).normalize(),
                (pd.Timestamp(end_date_inclusive).normalize() + pd.Timedelta(days=1)),
            ],
        ).df()
    finally:
        con.unregister("series_filter")

    if out.empty:
        return []
    return [pd.Timestamp(x).normalize() for x in out["asof_date"]]


def _build_processed_md_snapshot(
    *,
    store: AsofStore,
    asof_date: pd.Timestamp,
    md_variables: list[str],
    md_series_map: dict[str, str],
    md_transform_codes: dict[str, int],
    md_drop_initial_rows: int,
    obs_start: pd.Timestamp,
) -> pd.DataFrame:
    requested_ids = dedupe_preserve_order(md_series_map.values())
    snap = store.snapshot_wide(
        asof_ts=asof_date,
        series_ids=requested_ids,
        obs_start=obs_start,
        obs_end=asof_date,
        timestamp_name="timestamp",
    )
    if snap.empty:
        return pd.DataFrame(columns=["timestamp", *md_variables])

    wide = pd.DataFrame({"timestamp": pd.to_datetime(snap["timestamp"], errors="coerce")})
    for var in md_variables:
        sid = md_series_map.get(var)
        if sid and sid in snap.columns:
            wide[var] = pd.to_numeric(snap[sid], errors="coerce")
        else:
            wide[var] = np.nan

    wide = wide.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if wide.empty:
        return pd.DataFrame(columns=["timestamp", *md_variables])

    values = wide[md_variables].apply(pd.to_numeric, errors="coerce")
    transformed = apply_fred_transform_codes(
        data_df=values,
        transform_codes=md_transform_codes,
        columns=md_variables,
    )
    processed = pd.concat([wide[["timestamp"]], transformed], axis=1)
    processed = _remove_outliers_iqr(processed, columns=md_variables)
    processed = processed.iloc[max(0, int(md_drop_initial_rows)) :].reset_index(drop=True)
    return processed


def _make_forecast_plot(
    *,
    forecast_df: pd.DataFrame,
    release_date: pd.Timestamp,
    actual_outturn_saar: float | None,
    target_quarter: str,
    output_png: Path,
) -> None:
    plt.figure(figsize=(11, 6))
    for model_name, grp in forecast_df.groupby("model", sort=True):
        sub = grp.sort_values("asof_date")
        plt.plot(sub["asof_date"], sub["g_hat_saar"], marker="o", linewidth=1.8, label=str(model_name))

    plt.axvline(pd.Timestamp(release_date), color="black", linestyle="--", linewidth=1.25, label="First Release Date")
    if actual_outturn_saar is not None and np.isfinite(actual_outturn_saar):
        plt.scatter(
            [pd.Timestamp(release_date)],
            [float(actual_outturn_saar)],
            marker="x",
            s=110,
            linewidths=2.4,
            color="black",
            label="Actual First Release (X)",
            zorder=6,
        )
    plt.title(f"Realtime As-Of Forecasts for {target_quarter} GDP q/q SAAR")
    plt.xlabel("As-Of Date")
    plt.ylabel("Forecast (q/q SAAR, %)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_quarter = pd.Period(str(args.target_quarter), freq="Q-DEC")
    observed_quarter = target_quarter - 1
    start_asof = pd.Timestamp(args.start_asof).normalize()

    release_table = load_release_table(path=Path(args.release_csv).expanduser().resolve())
    q_by_first_release = (
        release_table.loc[release_table["first_release_date"].notna(), ["quarter", "first_release_date"]]
        .copy()
        .drop_duplicates(subset=["quarter"], keep="last")
    )
    q_by_first_level = (
        release_table.loc[release_table["first_release"].notna(), ["quarter", "first_release"]]
        .copy()
        .drop_duplicates(subset=["quarter"], keep="last")
    )
    q_by_first_saar = (
        release_table.loc[release_table["g_true_saar_first"].notna(), ["quarter", "g_true_saar_first"]]
        .copy()
        .drop_duplicates(subset=["quarter"], keep="last")
    )
    first_release_map = {
        pd.Period(q, freq="Q-DEC"): pd.Timestamp(d).normalize()
        for q, d in zip(q_by_first_release["quarter"], q_by_first_release["first_release_date"], strict=False)
    }
    first_level_map = {
        pd.Period(q, freq="Q-DEC"): float(v)
        for q, v in zip(q_by_first_level["quarter"], q_by_first_level["first_release"], strict=False)
    }
    first_saar_map = {
        pd.Period(q, freq="Q-DEC"): float(v)
        for q, v in zip(q_by_first_saar["quarter"], q_by_first_saar["g_true_saar_first"], strict=False)
    }
    if target_quarter not in first_release_map:
        raise ValueError(f"Target quarter {target_quarter} not found in release table first-release mapping.")
    first_release_date = first_release_map[target_quarter]
    actual_first_release_level = float(first_level_map.get(target_quarter, np.nan))
    actual_outturn_saar = float(first_saar_map.get(target_quarter, np.nan))
    if not np.isfinite(actual_outturn_saar):
        actual_outturn_saar = to_saar_growth(
            current_level=float(first_level_map.get(target_quarter, np.nan)),
            previous_level=float(first_level_map.get(target_quarter - 1, np.nan)),
        )
    end_asof = first_release_date - pd.Timedelta(days=1)
    if end_asof < start_asof:
        raise ValueError(
            f"start_asof {start_asof.date()} is after pre-release cutoff {end_asof.date()} for {target_quarter}."
        )

    qd_panel_path = resolve_qd_panel_path(mode="processed", explicit_path=None)
    vintage_panel = load_vintage_panel(path=qd_panel_path, target_col=args.target_col)
    vintage_calendar = build_vintage_calendar(vintage_panel=vintage_panel)

    md_template = Path(args.md_template).expanduser().resolve()
    md_variables, md_transform_codes = _load_md_template_columns_and_codes(md_template)

    store = AsofStore(db_path=Path(args.asof_db).expanduser().resolve())
    try:
        md_series_map = _resolve_md_series_map(store=store, md_variables=md_variables)
        md_series_ids = dedupe_preserve_order(md_series_map.values())
        candidate_asof_dates = _query_release_dates_for_md_series(
            store=store,
            series_ids=md_series_ids,
            start_date=start_asof,
            end_date_inclusive=end_asof,
        )
        asof_dates = dedupe_preserve_order(
            [start_asof.strftime("%Y-%m-%d"), *[d.strftime("%Y-%m-%d") for d in candidate_asof_dates], end_asof.strftime("%Y-%m-%d")]
        )
        asof_dates_ts = [pd.Timestamp(d).normalize() for d in asof_dates]

        if not asof_dates_ts:
            raise ValueError("No as-of dates found in requested pre-release window.")

        panel_rows: list[pd.DataFrame] = []
        asof_to_vintage: dict[pd.Timestamp, str] = {}
        obs_start = pd.Timestamp("1959-01-01")
        for asof_date in asof_dates_ts:
            v_id = f"asof_{asof_date.strftime('%Y%m%d')}"
            asof_to_vintage[asof_date] = v_id
            snap = _build_processed_md_snapshot(
                store=store,
                asof_date=asof_date,
                md_variables=md_variables,
                md_series_map=md_series_map,
                md_transform_codes=md_transform_codes,
                md_drop_initial_rows=int(args.md_drop_initial_rows),
                obs_start=obs_start,
            )
            if snap.empty:
                continue
            snap = snap.copy()
            snap.insert(0, "vintage", v_id)
            snap.insert(1, "vintage_timestamp", pd.Timestamp(asof_date))
            panel_rows.append(snap)

        if not panel_rows:
            raise ValueError("Processed MD as-of panel is empty for all requested as-of dates.")

        md_asof_panel = pd.concat(panel_rows, axis=0, ignore_index=True)
        md_panel_path = output_dir / "processed_md_asof_panel.parquet"
        md_asof_panel.to_parquet(md_panel_path, index=False)

        models = resolve_models(args.models)
        apply_model_runtime_options(
            model_list=models,
            md_panel_path=md_panel_path,
            qd_panel_path=qd_panel_path,
            mixed_freq_excluded_years=[],
        )

        forecast_rows: list[dict[str, Any]] = []
        for asof_date in asof_dates_ts:
            try:
                selected = select_training_vintage(origin_date=asof_date, vintage_calendar=vintage_calendar)
            except ValueError:
                continue

            ds = build_origin_datasets(
                origin_quarter=target_quarter,
                vintage_id=str(selected["vintage"]),
                release_table=release_table,
                vintage_panel=vintage_panel,
                target_col=args.target_col,
                train_window="expanding",
                rolling_size=None,
                cutoff_quarter=target_quarter,
            )
            train_df = ds["train_df"].copy()
            future_target_mask = pd.PeriodIndex(train_df["quarter"], freq="Q-DEC") > observed_quarter
            if future_target_mask.any():
                train_df.loc[future_target_mask, args.target_col] = np.nan

            md_vintage_id = asof_to_vintage.get(asof_date)
            if md_vintage_id is None:
                continue
            train_df["__origin_vintage"] = md_vintage_id
            train_df["__origin_schedule"] = "daily_asof"

            target_series = pd.to_numeric(train_df[args.target_col], errors="coerce").dropna()
            if target_series.empty:
                continue
            y_prev = float(target_series.iloc[-1])

            covariate_cols = select_covariate_columns(train_df=train_df, target_col=args.target_col)
            past_data, future_data, task = train_df_to_datasets(
                train_df=train_df,
                target_col=args.target_col,
                horizon=1,
                covariate_cols=covariate_cols,
            )
            task.train_df = train_df

            for model in models:
                try:
                    pred_ds = model.predict(past_data=past_data, future_data=future_data, task=task)
                    path = np.asarray(pred_ds["predictions"][0], dtype=float)
                    y_hat = float(path[0]) if path.size >= 1 else np.nan
                    g_hat = to_saar_growth(y_hat, y_prev)
                    error = ""
                except Exception as exc:  # noqa: BLE001
                    y_hat = np.nan
                    g_hat = np.nan
                    error = f"{type(exc).__name__}: {exc}"

                forecast_rows.append(
                    {
                        "asof_date": asof_date.date().isoformat(),
                        "first_release_date": first_release_date.date().isoformat(),
                        "target_quarter": str(target_quarter),
                        "observed_quarter": str(observed_quarter),
                        "model": str(getattr(model, "name", "")),
                        "origin_vintage_panel": str(selected["vintage"]),
                        "origin_vintage_asof_date": pd.Timestamp(selected["asof_date"]).date().isoformat(),
                        "md_asof_vintage_id": md_vintage_id,
                        "last_observed_level": y_prev,
                        "y_hat_level": y_hat,
                        "g_hat_saar": g_hat,
                        "actual_first_release_level": actual_first_release_level,
                        "actual_first_release_qoq_saar": actual_outturn_saar,
                        "error": error,
                    }
                )

        forecast_df = pd.DataFrame(forecast_rows)
        if forecast_df.empty:
            raise ValueError("No forecasts were produced.")

        forecast_df["asof_date"] = pd.to_datetime(forecast_df["asof_date"], errors="coerce")
        forecast_df = forecast_df.sort_values(["model", "asof_date"]).reset_index(drop=True)

        timeseries_csv = output_dir / "forecasts_timeseries.csv"
        forecast_df.to_csv(timeseries_csv, index=False)

        pre_release = forecast_df.loc[forecast_df["asof_date"] < first_release_date].copy()
        final_df = (
            pre_release.sort_values(["model", "asof_date"])
            .groupby("model", as_index=False, sort=True)
            .tail(1)
            .sort_values("model")
            .reset_index(drop=True)
        )
        final_csv = output_dir / "final_forecast_pre_first_release.csv"
        final_df.to_csv(final_csv, index=False)

        actual_outturn_csv = output_dir / "actual_outturn_first_release.csv"
        pd.DataFrame(
            [
                {
                    "target_quarter": str(target_quarter),
                    "first_release_date": first_release_date.date().isoformat(),
                    "actual_first_release_level": actual_first_release_level,
                    "actual_first_release_qoq_saar": actual_outturn_saar,
                }
            ]
        ).to_csv(actual_outturn_csv, index=False)

        plot_png = output_dir / "forecasts_timeseries.png"
        _make_forecast_plot(
            forecast_df=forecast_df,
            release_date=first_release_date,
            actual_outturn_saar=actual_outturn_saar,
            target_quarter=str(target_quarter),
            output_png=plot_png,
        )

        asof_dates_csv = output_dir / "asof_dates_used.csv"
        pd.DataFrame({"asof_date": [d.date().isoformat() for d in asof_dates_ts]}).to_csv(asof_dates_csv, index=False)

        print(f"Target quarter: {target_quarter} | first release date: {first_release_date.date().isoformat()}")
        print(f"As-of runs: {len(asof_dates_ts)}")
        print(f"Wrote MD as-of processed panel: {md_panel_path}")
        print(f"Wrote forecast time series CSV: {timeseries_csv}")
        print(f"Wrote final pre-release forecast CSV: {final_csv}")
        print(f"Wrote actual outturn CSV: {actual_outturn_csv}")
        print(f"Wrote forecast plot PNG: {plot_png}")
        if np.isfinite(actual_outturn_saar):
            print(f"Actual first-release outturn (q/q SAAR): {actual_outturn_saar:.6f}")
        if not final_df.empty:
            print("Final forecast(s) before first release:")
            print(final_df[["model", "asof_date", "g_hat_saar", "y_hat_level"]].to_string(index=False))

    finally:
        store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
