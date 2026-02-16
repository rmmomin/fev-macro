#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.models import MODEL_REGISTRY, available_models, build_models  # noqa: E402
from fev_macro.realtime_feeds import select_covariate_columns, train_df_to_datasets  # noqa: E402
from fev_macro.realtime_oos import (  # noqa: E402
    BUILTIN_MODELS,
    _RealtimeModelAdapter,
    apply_model_runtime_options,
    build_origin_datasets,
    compute_vintage_asof_date,
    load_release_table,
    load_vintage_panel,
    to_saar_growth,
)
from fev_macro.realtime_runner import (  # noqa: E402
    normalize_mode,
    resolve_latest_output_path,
    resolve_md_panel_path,
    resolve_models as resolve_mode_models,
    resolve_qd_panel_path,
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
        "--mode",
        type=str,
        choices=["unprocessed", "processed"],
        default="processed",
        help="Realtime mode. Controls default panel/model/output paths.",
    )
    parser.add_argument(
        "--vintage_panel",
        type=str,
        default="",
        help="Path to FRED-QD vintage panel parquet. Defaults by --mode.",
    )
    parser.add_argument(
        "--md_vintage_panel",
        type=str,
        default="",
        help=(
            "Optional path to FRED-MD vintage panel parquet used by MD-feature models "
            "(random_forest, xgboost, mixed_freq_dfm_md). Defaults by --mode."
        ),
    )
    parser.add_argument(
        "--processed_qd_csv",
        type=str,
        default="",
        help=(
            "Optional processed single-vintage QD CSV (e.g., data/processed/fred_qd_2026m1_processed.csv). "
            "When set, this overrides --vintage_panel input."
        ),
    )
    parser.add_argument(
        "--processed_md_csv",
        type=str,
        default="",
        help=(
            "Optional processed single-vintage MD CSV (e.g., data/processed/fred_md_2026m1_processed.csv). "
            "When set, a temporary one-vintage MD panel parquet is built and injected into MD-feature models."
        ),
    )
    parser.add_argument(
        "--processed_vintage",
        type=str,
        default="",
        help=(
            "Vintage label (YYYY-MM) for processed CSV inputs. If omitted, inferred from filename; "
            "fallback is --vintage when that is not 'latest'."
        ),
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
        default=None,
        help=f"Model names. Available: {sorted(set(BUILTIN_MODELS) | set(available_models()))}",
    )
    parser.add_argument("--train_window", type=str, default="expanding", choices=["expanding", "rolling"])
    parser.add_argument("--rolling_size", type=int, default=None)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output CSV with model forecasts. Defaults by --mode.",
    )
    parser.add_argument(
        "--exclude_years",
        type=int,
        nargs="*",
        default=[],
        help="Optional calendar years to exclude inside mixed-frequency MD factor models (default: none).",
    )
    parser.add_argument(
        "--per_model_timeout_sec",
        type=int,
        default=180,
        help="Timeout in seconds per model prediction attempt (best effort on POSIX).",
    )
    return parser.parse_args()


def _infer_vintage_from_path(path: Path) -> str | None:
    text = path.name.lower()
    m = re.search(r"(\d{4})m(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    m = re.search(r"(\d{4})-(\d{2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    return None


def _resolve_processed_vintage(args: argparse.Namespace, *paths: Path) -> str:
    if args.processed_vintage:
        return str(pd.Period(str(args.processed_vintage), freq="M"))
    if str(args.vintage).lower() != "latest":
        return str(pd.Period(str(args.vintage), freq="M"))
    for path in paths:
        inferred = _infer_vintage_from_path(path)
        if inferred:
            return inferred
    raise ValueError(
        "Unable to resolve processed vintage label. Provide --processed_vintage YYYY-MM "
        "or set --vintage explicitly."
    )


def _load_processed_single_vintage_csv(
    csv_path: Path,
    vintage_id: str,
    target_col: str | None,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    if "date" not in raw.columns:
        raise ValueError(f"Processed CSV must contain 'date' column: {csv_path}")
    if target_col and target_col not in raw.columns:
        raise ValueError(f"Processed CSV missing target column '{target_col}': {csv_path}")

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop(columns=["date"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df.insert(0, "vintage", str(vintage_id))
    df.insert(1, "vintage_timestamp", pd.Timestamp(pd.Period(vintage_id, freq="M").start_time))
    df["quarter"] = pd.PeriodIndex(df["timestamp"], freq="Q-DEC")
    df["asof_date"] = compute_vintage_asof_date(vintage_id)
    if target_col:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df


def _load_vintage_panel_from_args(
    args: argparse.Namespace,
    panel_path: Path | None,
) -> tuple[pd.DataFrame, str]:
    if not args.processed_qd_csv:
        panel = load_vintage_panel(path=panel_path, target_col=args.target_col)
        vintage_id = _resolve_vintage(vintage_panel=panel, vintage_arg=str(args.vintage))
        return panel, vintage_id

    qd_csv_path = Path(args.processed_qd_csv).expanduser().resolve()
    if not qd_csv_path.exists():
        raise FileNotFoundError(f"Processed QD CSV not found: {qd_csv_path}")

    vintage_id = _resolve_processed_vintage(args, qd_csv_path)
    panel = _load_processed_single_vintage_csv(
        csv_path=qd_csv_path,
        vintage_id=vintage_id,
        target_col=args.target_col,
    )
    return panel, vintage_id


def _build_md_panel_from_processed_csv(
    args: argparse.Namespace,
    output_csv_path: Path,
    default_md_panel_path: Path,
) -> Path | None:
    if not args.processed_md_csv:
        return default_md_panel_path

    md_csv_path = Path(args.processed_md_csv).expanduser().resolve()
    if not md_csv_path.exists():
        raise FileNotFoundError(f"Processed MD CSV not found: {md_csv_path}")

    vintage_id = _resolve_processed_vintage(args, md_csv_path)
    md_df = _load_processed_single_vintage_csv(
        csv_path=md_csv_path,
        vintage_id=vintage_id,
        target_col=None,
    )
    md_df = md_df.drop(columns=["quarter", "asof_date"], errors="ignore")

    out_path = output_csv_path.parent / f"md_panel_from_processed_{vintage_id.replace('-', '')}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_df.to_parquet(out_path, index=False)
    return out_path


def _resolve_models_prefer_builtin(models: list[str]) -> list:
    resolved: list = []
    for model_name in models:
        key = str(model_name).strip().lower()
        if key in BUILTIN_MODELS:
            resolved.append(_RealtimeModelAdapter(BUILTIN_MODELS[key]()))
            continue
        if key in MODEL_REGISTRY:
            resolved.append(build_models([key], seed=0)[key])
            continue
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(set(BUILTIN_MODELS) | set(MODEL_REGISTRY))}")
    return resolved


class _ModelTimeoutError(TimeoutError):
    pass


def _run_predict_with_timeout(
    model,
    *,
    past_data,
    future_data,
    task_shim,
    timeout_sec: int,
):
    # Best-effort timeout guard for long-running model calls on POSIX.
    if timeout_sec <= 0 or not hasattr(signal, "SIGALRM"):
        return model.predict(past_data=past_data, future_data=future_data, task=task_shim)

    def _alarm_handler(signum, frame):
        raise _ModelTimeoutError(f"Model prediction exceeded {timeout_sec} seconds")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(timeout_sec))
    try:
        return model.predict(past_data=past_data, future_data=future_data, task=task_shim)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


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
    mode = normalize_mode(args.mode)
    release_path = Path(args.release_csv).resolve() if args.release_csv else None
    panel_path = resolve_qd_panel_path(mode=mode, explicit_path=args.vintage_panel or None)
    output_path = resolve_latest_output_path(mode=mode, explicit_path=args.output_csv or None)
    md_panel_default = resolve_md_panel_path(mode=mode, explicit_path=args.md_vintage_panel or None)
    requested_models = resolve_mode_models(mode=mode, requested_models=args.models)

    release_table = load_release_table(path=release_path)
    vintage_panel, vintage_id = _load_vintage_panel_from_args(args=args, panel_path=panel_path)
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

    model_list = _resolve_models_prefer_builtin(list(requested_models))
    md_panel_override = _build_md_panel_from_processed_csv(
        args=args,
        output_csv_path=output_path,
        default_md_panel_path=md_panel_default,
    )
    apply_model_runtime_options(
        model_list=model_list,
        md_panel_path=md_panel_override,
        mixed_freq_excluded_years=args.exclude_years,
    )
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
            pred_ds = _run_predict_with_timeout(
                model,
                past_data=past_data,
                future_data=future_data,
                task_shim=task_shim,
                timeout_sec=int(args.per_model_timeout_sec),
            )
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    final = out_df.loc[out_df["target_quarter"] == str(target_q)].copy()
    final = final.sort_values("g_hat_saar", ascending=True, na_position="last")

    print(
        "Run config: "
        f"mode={mode}, vintage={vintage_id}, observed_max_quarter={observed_q}, target_quarter={target_q}, "
        f"horizon={horizon}, models={len(model_list)}, covariate_cutoff={train_df['quarter'].max()}"
    )
    if args.processed_qd_csv:
        print(f"Processed QD input: {Path(args.processed_qd_csv).expanduser().resolve()}")
    if args.processed_md_csv:
        print(f"Processed MD input: {Path(args.processed_md_csv).expanduser().resolve()}")
    if md_panel_override is not None:
        print(f"MD panel used by MD-feature models: {md_panel_override}")
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
