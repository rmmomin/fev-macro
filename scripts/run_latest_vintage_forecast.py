#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.models import available_models  # noqa: E402
from fev_macro.realtime_feeds import select_covariate_columns, train_df_to_datasets  # noqa: E402
from fev_macro.realtime_oos import (  # noqa: E402
    BUILTIN_MODELS,
    build_origin_datasets,
    load_release_table,
    load_vintage_panel,
    resolve_models,
    to_saar_growth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train models on a single chosen FRED-QD vintage and produce level/qoq/SAAR forecasts "
            "for a requested target quarter."
        )
    )
    parser.add_argument(
        "--release_csv",
        type=str,
        default="",
        help="Optional release table path (used for compatibility with shared dataset builders).",
    )
    parser.add_argument(
        "--vintage_panel",
        type=str,
        default="",
        help="Path to FRED-QD vintage panel parquet. Defaults to data/panels/fred_qd_vintage_panel.parquet.",
    )
    parser.add_argument(
        "--vintage",
        type=str,
        default="latest",
        help="Vintage id YYYY-MM. Use 'latest' to pick the max available vintage.",
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
        default="2025Q4",
        help="Target quarter in YYYYQ# format.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "rw_drift_log",
            "ar4_growth",
            "auto_ets",
            "theta",
            "drift",
            "auto_arima",
            "local_trend_ssm",
            "random_forest",
            "xgboost",
            "chronos2",
            "factor_pca_qd",
            "bvar_minnesota_8",
            "bvar_minnesota_20",
            "mixed_freq_dfm_md",
            "ensemble_avg_top3",
            "ensemble_weighted_top5",
            "naive_last",
        ],
        help=f"Model names. Available: {sorted(set(BUILTIN_MODELS) | set(available_models()))}",
    )
    parser.add_argument("--train_window", type=str, default="expanding", choices=["expanding", "rolling"])
    parser.add_argument("--rolling_size", type=int, default=None)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/realtime_latest_vintage_forecast.csv",
        help="Output CSV with model forecasts.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    release_path = Path(args.release_csv).resolve() if args.release_csv else None
    panel_path = Path(args.vintage_panel).resolve() if args.vintage_panel else None
    return release_path, panel_path


def _resolve_vintage(vintage_panel: pd.DataFrame, vintage_arg: str) -> str:
    available = sorted(vintage_panel["vintage"].astype(str).dropna().unique())
    if not available:
        raise ValueError("No vintages found in vintage panel.")
    if vintage_arg.lower() == "latest":
        return str(available[-1])
    if vintage_arg not in set(available):
        raise ValueError(f"Requested vintage '{vintage_arg}' not found. Latest available is '{available[-1]}'.")
    return vintage_arg


def _latest_observed_quarter(vintage_slice: pd.DataFrame, target_col: str) -> pd.Period:
    valid = vintage_slice.loc[pd.to_numeric(vintage_slice[target_col], errors="coerce").notna(), "quarter"]
    if valid.empty:
        raise ValueError("No observed target levels in the selected vintage.")
    return pd.Period(valid.max(), freq="Q-DEC")


def main() -> int:
    args = parse_args()
    release_path, panel_path = _resolve_paths(args)

    release_table = load_release_table(path=release_path)
    vintage_panel = load_vintage_panel(path=panel_path, target_col=args.target_col)

    vintage_id = _resolve_vintage(vintage_panel=vintage_panel, vintage_arg=str(args.vintage))
    target_q = pd.Period(str(args.target_quarter), freq="Q-DEC")

    vintage_slice = vintage_panel.loc[vintage_panel["vintage"].astype(str) == vintage_id].copy()
    observed_q = _latest_observed_quarter(vintage_slice=vintage_slice, target_col=args.target_col)
    if target_q <= observed_q:
        raise ValueError(
            f"target_quarter={target_q} must be after latest observed quarter in vintage {vintage_id}: {observed_q}"
        )

    horizon = int(target_q.ordinal - observed_q.ordinal)
    ds = build_origin_datasets(
        origin_quarter=observed_q + 1,
        vintage_id=vintage_id,
        release_table=release_table,
        vintage_panel=vintage_panel,
        target_col=args.target_col,
        train_window=args.train_window,
        rolling_size=args.rolling_size,
        cutoff_quarter=target_q,
    )
    train_df = ds["train_df"].copy()
    train_df["__origin_vintage"] = vintage_id
    train_df["__origin_schedule"] = "manual_latest_vintage"
    last_observed = float(ds["last_observed_level"])

    model_list = resolve_models(args.models)
    run_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    rows: list[dict[str, object]] = []
    for model in model_list:
        try:
            covariate_cols = select_covariate_columns(train_df=train_df, target_col=args.target_col)
            past_data, future_data, task_shim = train_df_to_datasets(
                train_df=train_df,
                target_col=args.target_col,
                horizon=horizon,
                covariate_cols=covariate_cols,
            )
            task_shim.train_df = train_df
            pred_ds = model.predict(past_data=past_data, future_data=future_data, task=task_shim)
            path = np.asarray(pred_ds["predictions"][0], dtype=float)
            if path.size != horizon:
                raise ValueError(f"expected horizon={horizon}, got {path.size}")
            if not np.isfinite(path).all():
                raise ValueError("non-finite values in forecast path")
            error_msg = ""
        except Exception as exc:
            path = np.full(horizon, np.nan, dtype=float)
            error_msg = f"{type(exc).__name__}: {exc}"

        for h in range(1, horizon + 1):
            tq = observed_q + h
            y_hat = float(path[h - 1]) if np.isfinite(path[h - 1]) else np.nan
            y_prev = float(last_observed if h == 1 else path[h - 2]) if np.isfinite(y_hat) else np.nan
            qoq = float((y_hat / y_prev - 1.0) * 100.0) if np.isfinite(y_hat) and np.isfinite(y_prev) and y_prev > 0 else np.nan
            saar = float(to_saar_growth(y_hat, y_prev)) if np.isfinite(y_hat) and np.isfinite(y_prev) else np.nan
            rows.append(
                {
                    "run_timestamp_utc": run_ts,
                    "vintage": vintage_id,
                    "observed_max_quarter": str(observed_q),
                    "target_quarter_requested": str(target_q),
                    "model": model.name,
                    "horizon": h,
                    "target_quarter": str(tq),
                    "last_observed_level": last_observed,
                    "y_hat_level": y_hat,
                    "y_hat_prev_level": y_prev,
                    "qoq_pct": qoq,
                    "g_hat_saar": saar,
                    "error": error_msg,
                }
            )

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    final = out_df.loc[out_df["target_quarter"] == str(target_q)].copy()
    final = final.sort_values("g_hat_saar", ascending=True, na_position="last")

    print(
        "Run config: "
        f"vintage={vintage_id}, observed_max_quarter={observed_q}, target_quarter={target_q}, "
        f"horizon={horizon}, models={len(model_list)}, covariate_cutoff={train_df['quarter'].max()}"
    )
    print(f"Wrote forecast table: {output_path} ({len(out_df)} rows)")
    if not final.empty:
        print("\nTarget-quarter forecasts:")
        cols = ["model", "y_hat_level", "qoq_pct", "g_hat_saar", "error"]
        print(final[cols].to_string(index=False))
    else:
        print("No rows found for requested target quarter in output.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
