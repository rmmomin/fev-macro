#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import (  # noqa: E402
    DEFAULT_SOURCE_SERIES_CANDIDATES,
    DEFAULT_TARGET_SERIES_NAME,
    DEFAULT_TARGET_TRANSFORM,
    HistoricalQuarterlyVintageProvider,
    SUPPORTED_TARGET_TRANSFORMS,
    build_reindexed_to_actual_timestamp_map,
    build_real_gdp_target_series,
    exclude_years,
    export_local_dataset_parquet,
    load_fred_qd_transform_codes,
    load_fev_dataset,
    reindex_to_regular_frequency,
)
from fev_macro.models import build_models  # noqa: E402
from fev_macro.models.base import get_task_horizon, get_task_target_column, get_task_timestamp_column  # noqa: E402
from fev_macro.tasks import make_gdp_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot train/OOS real GDP q/q SAAR growth from log-real-GDP forecasts "
            "for top-K models in the leaderboard."
        )
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--leaderboard", type=str, default="results/leaderboard.csv")
    parser.add_argument("--dataset_path", type=str, default="autogluon/fev_datasets")
    parser.add_argument("--dataset_config", type=str, default="fred_qd_2025")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_SERIES_NAME)
    parser.add_argument(
        "--target_transform",
        type=str,
        default=DEFAULT_TARGET_TRANSFORM,
        choices=sorted(SUPPORTED_TARGET_TRANSFORMS),
        help="Forecast target transform used by the benchmark (must be log_level or level for this plot).",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--num_windows", type=int, default=80)
    parser.add_argument("--metric", type=str, default="RMSE")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude_years",
        type=int,
        nargs="*",
        default=[2020],
        help="Calendar years excluded from model training/evaluation windows.",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="log_real_gdp",
        help="Task prefix used in benchmark task names.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/log_real_gdp_growth_oos_top5.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default="results/log_real_gdp_growth_oos_top5.csv",
        help="Output OOS predictions csv path",
    )
    parser.add_argument(
        "--source_series_candidates",
        nargs="+",
        default=list(DEFAULT_SOURCE_SERIES_CANDIDATES),
    )
    parser.add_argument(
        "--historical_qd_dir",
        type=str,
        default="data/historical/qd/vintages_2018_2026",
        help="Path to historical FRED-QD vintage CSVs for vintage-correct training windows.",
    )
    parser.add_argument(
        "--disable_historical_vintages",
        action="store_true",
        help="Disable historical-vintage training and use finalized data for all windows.",
    )
    parser.add_argument(
        "--disable_fred_transforms",
        action="store_true",
        help="Disable FRED-MD/QD tcode transforms for covariates during dataset construction.",
    )
    parser.add_argument(
        "--fred_transform_vintage",
        type=str,
        default=None,
        help="Optional transform-code vintage month (YYYY-MM); default uses latest available historical vintage.",
    )
    parser.add_argument(
        "--vintage_fallback_to_earliest",
        action="store_true",
        help="Allow pre-vintage windows to use earliest available vintage instead of strict coverage.",
    )
    return parser.parse_args()


def load_top_models(leaderboard_path: Path, top_k: int) -> list[str]:
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    leaderboard = pd.read_csv(leaderboard_path)
    if "model_name" not in leaderboard.columns:
        raise ValueError(f"Leaderboard must include model_name column. Found columns={list(leaderboard.columns)}")

    if "win_rate" in leaderboard.columns and "test_error" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(["win_rate", "test_error"], ascending=[False, True])
    elif "win_rate" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(["win_rate"], ascending=[False])
    elif "test_error" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(["test_error"], ascending=[True])

    top_models = leaderboard["model_name"].head(top_k).tolist()
    if not top_models:
        raise ValueError("No models found in leaderboard")
    return top_models


def collect_oos_predictions(
    task,
    model_names: list[str],
    seed: int,
    past_data_adapter: Callable | None = None,
) -> pd.DataFrame:
    models = build_models(model_names=model_names, seed=seed)
    horizon = get_task_horizon(task)
    ts_col = get_task_timestamp_column(task)
    target_col = get_task_target_column(task)

    data: dict[pd.Timestamp, dict[str, list[float]]] = {}

    for window_idx, window in enumerate(task.iter_windows()):
        input_data = window.get_input_data()
        if isinstance(input_data, tuple) and len(input_data) == 2:
            past_data, future_data = input_data
        else:
            past_data = input_data
            future_data = window.get_future_data()

        if past_data_adapter is not None:
            past_data = past_data_adapter(past_data, future_data, task, window_idx)

        truth = window.get_ground_truth()
        ts_values = pd.to_datetime(pd.Series(truth[0][ts_col]), errors="coerce")
        actual_values = pd.to_numeric(pd.Series(truth[0][target_col]), errors="coerce")

        model_preds: dict[str, np.ndarray] = {}
        for model_name, model in models.items():
            pred_ds = model.predict(past_data=past_data, future_data=future_data, task=task)
            arr = np.asarray(pred_ds["predictions"][0], dtype=float).reshape(-1)
            if arr.size != horizon:
                raise ValueError(f"{model_name} produced horizon={arr.size}, expected={horizon}")
            if np.isnan(arr).any() or not np.isfinite(arr).all():
                raise ValueError(f"{model_name} produced NaN/inf forecasts")
            model_preds[model_name] = arr

        for i in range(horizon):
            ts = ts_values.iloc[i]
            actual = float(actual_values.iloc[i])
            if pd.isna(ts) or not np.isfinite(actual):
                continue

            bucket = data.setdefault(pd.Timestamp(ts), {"actual": []})
            bucket["actual"].append(actual)
            for model_name in model_names:
                bucket.setdefault(model_name, []).append(float(model_preds[model_name][i]))

    rows: list[dict[str, float]] = []
    for ts in sorted(data.keys()):
        row: dict[str, float] = {"timestamp": ts, "actual": float(np.mean(data[ts]["actual"]))}
        for model_name in model_names:
            preds = data[ts].get(model_name, [])
            row[model_name] = float(np.mean(preds)) if preds else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _to_log_series(values: pd.Series, target_transform: str) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    if target_transform == "log_level":
        return series
    if target_transform == "level":
        return np.log(series.where(series > 0, np.nan))
    raise ValueError(
        "Plot conversion to q/q SAAR from log GDP requires target_transform in {'log_level', 'level'}. "
        f"Got: {target_transform}"
    )


def _log_diff_to_saar(log_diff: pd.Series) -> pd.Series:
    diff = pd.to_numeric(log_diff, errors="coerce")
    return 100.0 * (np.exp(4.0 * diff) - 1.0)


def make_growth_view(
    filtered_target_df: pd.DataFrame,
    full_reference_df: pd.DataFrame,
    oos_raw_df: pd.DataFrame,
    model_names: list[str],
    target_transform: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = filtered_target_df[["timestamp", "target"]].copy()
    full["timestamp"] = pd.to_datetime(full["timestamp"], errors="coerce")
    full = full.dropna(subset=["timestamp", "target"]).sort_values("timestamp").reset_index(drop=True)

    ref = full_reference_df[["timestamp", "target"]].copy()
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], errors="coerce")
    ref = ref.dropna(subset=["timestamp", "target"]).sort_values("timestamp").reset_index(drop=True)
    ref["actual_log"] = _to_log_series(ref["target"], target_transform=target_transform)
    ref["prev_log"] = ref["actual_log"].shift(1)

    full["actual_log"] = _to_log_series(full["target"], target_transform=target_transform)
    prev_lookup = ref[["timestamp", "prev_log"]].copy()
    full = full.merge(prev_lookup, on="timestamp", how="left")
    full["actual_growth_saar"] = _log_diff_to_saar(full["actual_log"] - full["prev_log"])

    oos = oos_raw_df.copy()
    oos["timestamp"] = pd.to_datetime(oos["timestamp"], errors="coerce")
    oos = oos.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    oos = oos.merge(prev_lookup, on="timestamp", how="left")
    oos["actual_log"] = _to_log_series(oos["actual"], target_transform=target_transform)
    oos["actual_growth_saar"] = _log_diff_to_saar(oos["actual_log"] - oos["prev_log"])

    for model_name in model_names:
        if model_name not in oos.columns:
            continue
        oos[f"{model_name}_log"] = _to_log_series(oos[model_name], target_transform=target_transform)
        oos[f"{model_name}_growth_saar"] = _log_diff_to_saar(oos[f"{model_name}_log"] - oos["prev_log"])

    first_test_ts = pd.to_datetime(oos["timestamp"]).min()
    history = full.loc[full["timestamp"] < first_test_ts, ["timestamp", "actual_growth_saar"]].dropna().copy()

    return history, oos


def make_plot(
    history_df: pd.DataFrame,
    oos_growth_df: pd.DataFrame,
    model_names: list[str],
    output_path: Path,
    horizon: int,
    num_windows: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    first_test_ts = pd.to_datetime(oos_growth_df["timestamp"]).min()

    ax.plot(
        history_df["timestamp"],
        history_df["actual_growth_saar"],
        color="black",
        linewidth=2.2,
        label="Actual (train)",
    )

    test_actual = oos_growth_df[["timestamp", "actual_growth_saar"]].dropna().sort_values("timestamp")
    ax.plot(
        test_actual["timestamp"],
        test_actual["actual_growth_saar"],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Actual (test)",
    )

    for model_name in model_names:
        col = f"{model_name}_growth_saar"
        if col not in oos_growth_df.columns:
            continue
        ax.plot(
            oos_growth_df["timestamp"],
            oos_growth_df[col],
            linewidth=1.8,
            alpha=0.9,
            label=f"{model_name} (OOS)",
        )

    ax.axvline(first_test_ts, color="gray", linestyle=":", linewidth=1.6, label="Test start")

    ax.set_title(
        (
            "US Real GDP Growth (q/q SAAR from log real GDP): "
            f"Train History + OOS Forecasts (Top {len(model_names)} Models, h={horizon}, windows={num_windows})"
        )
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth (percent, SAAR)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    leaderboard_path = Path(args.leaderboard)
    output_path = Path(args.output)
    predictions_csv = Path(args.predictions_csv)

    top_models = load_top_models(leaderboard_path=leaderboard_path, top_k=args.top_k)

    dataset = load_fev_dataset(
        config=args.dataset_config,
        dataset_path=args.dataset_path,
        dataset_revision=args.dataset_revision,
    )

    apply_fred_transforms = not args.disable_fred_transforms
    transform_vintage = pd.Period(args.fred_transform_vintage, freq="M") if args.fred_transform_vintage else None
    fred_transform_codes: dict[str, int] = {}
    if apply_fred_transforms:
        try:
            fred_transform_codes = load_fred_qd_transform_codes(
                historical_qd_dir=args.historical_qd_dir,
                vintage_period=transform_vintage,
            )
        except Exception as err:
            print(f"Warning: unable to load FRED transform codes ({err}); continuing without transforms.")
            apply_fred_transforms = False

    gdp_full_df, gdp_meta = build_real_gdp_target_series(
        dataset=dataset,
        target_series_name=args.target,
        target_transform=args.target_transform,
        source_series_candidates=args.source_series_candidates,
        include_covariates=True,
        apply_fred_transforms=apply_fred_transforms,
        fred_transform_codes=fred_transform_codes,
    )
    years_to_exclude = sorted({int(y) for y in (args.exclude_years or [])})
    gdp_df = exclude_years(gdp_full_df, years=years_to_exclude)
    gdp_filtered_actual = gdp_df.copy()
    freq = pd.infer_freq(pd.to_datetime(gdp_full_df["timestamp"], errors="coerce").dropna().sort_values()) or "QS-DEC"
    gdp_task_df = reindex_to_regular_frequency(gdp_df, freq=freq)
    timestamp_mapping = build_reindexed_to_actual_timestamp_map(
        actual_df=gdp_filtered_actual,
        reindexed_df=gdp_task_df,
        id_col="item_id",
        timestamp_col="timestamp",
    )
    covariate_columns = list(gdp_meta.get("covariate_columns", []))

    local_dataset_path = results_dir / f"{args.target.lower()}_dataset.parquet"
    export_local_dataset_parquet(gdp_task_df, output_path=local_dataset_path)

    vintage_provider: HistoricalQuarterlyVintageProvider | None = None
    past_data_adapter = None
    if not args.disable_historical_vintages:
        vintage_provider = HistoricalQuarterlyVintageProvider(
            historical_qd_dir=args.historical_qd_dir,
            target_series_name=args.target,
            target_transform=args.target_transform,
            source_series_candidates=args.source_series_candidates,
            covariate_columns=covariate_columns,
            include_covariates=True,
            apply_fred_transforms=apply_fred_transforms,
            exclude_years_list=years_to_exclude,
            timestamp_mapping=timestamp_mapping,
            strict=not args.vintage_fallback_to_earliest,
            fallback_to_earliest=args.vintage_fallback_to_earliest,
        )
        print(
            "Historical FRED-QD vintages enabled for plotting: "
            f"{vintage_provider.available_range_str()} ({len(vintage_provider.vintage_periods)} files) "
            f"from {vintage_provider.historical_qd_dir}"
        )

    task = make_gdp_tasks(
        dataset_path=str(local_dataset_path),
        dataset_config=None,
        horizons=[args.horizon],
        num_windows=args.num_windows,
        metric=args.metric,
        id_col="id",
        timestamp_col="timestamp",
        target_col="target",
        known_dynamic_columns=covariate_columns,
        task_prefix=args.task_prefix,
    )[0]

    if vintage_provider is not None:
        if not args.vintage_fallback_to_earliest:
            compatible_windows = vintage_provider.compatible_window_count(task)
            if compatible_windows <= 0:
                raise ValueError(
                    "No compatible windows for historical-vintage plotting at "
                    f"horizon {args.horizon}. Reduce --num_windows or enable --vintage_fallback_to_earliest."
                )
            if compatible_windows < int(args.num_windows):
                print(
                    f"Reducing plotting windows from {args.num_windows} to {compatible_windows} "
                    "to enforce strict historical-vintage coverage."
                )
                task = make_gdp_tasks(
                    dataset_path=str(local_dataset_path),
                    dataset_config=None,
                    horizons=[args.horizon],
                    num_windows=int(compatible_windows),
                    metric=args.metric,
                    id_col="id",
                    timestamp_col="timestamp",
                    target_col="target",
                    known_dynamic_columns=covariate_columns,
                    task_prefix=args.task_prefix,
                )[0]

        def past_data_adapter(past_data, future_data, task, window_idx):  # type: ignore[no-redef]
            _ = future_data
            _ = window_idx
            return vintage_provider.adapt_past_data(past_data=past_data, task=task)

    effective_num_windows = int(getattr(task, "num_windows", args.num_windows))

    oos_raw_df = collect_oos_predictions(
        task=task,
        model_names=top_models,
        seed=args.seed,
        past_data_adapter=past_data_adapter,
    )
    if oos_raw_df.empty:
        raise ValueError("No OOS predictions were generated")
    ts_map = timestamp_mapping
    oos_raw_df["timestamp"] = pd.to_datetime(oos_raw_df["timestamp"], errors="coerce").map(ts_map)
    oos_raw_df = oos_raw_df.dropna(subset=["timestamp"]).reset_index(drop=True)

    history_df, oos_growth_df = make_growth_view(
        filtered_target_df=gdp_df,
        full_reference_df=gdp_full_df,
        oos_raw_df=oos_raw_df,
        model_names=top_models,
        target_transform=args.target_transform,
    )

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    oos_growth_df.to_csv(predictions_csv, index=False)

    make_plot(
        history_df=history_df,
        oos_growth_df=oos_growth_df,
        model_names=top_models,
        output_path=output_path,
        horizon=args.horizon,
        num_windows=effective_num_windows,
    )

    print(f"Top models: {top_models}")
    print(
        "Target used: "
        f"{gdp_meta['target_series']} "
        f"(transform={gdp_meta.get('target_transform', args.target_transform)}, "
        f"computed={gdp_meta['computed']}, source={gdp_meta['source_series']})"
    )
    print(f"Excluded years: {years_to_exclude}")
    print(f"Covariates used: {len(covariate_columns)}")
    print(
        "FRED transforms enabled: "
        f"{apply_fred_transforms}; transform codes loaded={len(fred_transform_codes)}; "
        f"covariates transformed={len(gdp_meta.get('transformed_covariates', []))}"
    )
    print(f"Saved OOS predictions: {predictions_csv}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
