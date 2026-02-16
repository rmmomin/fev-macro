from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import numpy as np
import pandas as pd

from .data import (
    DEFAULT_SOURCE_SERIES_CANDIDATES,
    DEFAULT_TARGET_SERIES_NAME,
    HistoricalQuarterlyVintageProvider,
    RELEASE_STAGE_TO_REALTIME_SAAR_COLUMN,
    SUPPORTED_TARGET_TRANSFORMS,
    apply_gdpc1_release_truth_target,
    build_covariate_df,
    build_release_target_scaffold,
    build_reindexed_to_actual_timestamp_map,
    exclude_years,
    export_local_dataset_parquet,
    reindex_to_regular_frequency,
)
from .eval import run_models_on_tasks_with_records, save_summaries, save_timing_report
from .model_sets import MODELS_G_PROCESSED, MODELS_LL_UNPROCESSED, resolve_model_names
from .models import available_models, build_models
from .report import generate_reports, infer_metric_column
from .tasks import make_gdp_tasks

CovariateMode = Literal["unprocessed", "processed"]
ModelSet = Literal["auto", "ll", "g"]
Profile = Literal["smoke", "standard", "full"]

PROFILE_CHOICES: tuple[Profile, ...] = ("smoke", "standard", "full")

FULL_PROFILE_MODELS: list[str] = [
    "naive_last",
    "mean",
    "ar4",
    "drift",
    "seasonal_naive",
    "random_normal",
    "random_uniform",
    "random_permutation",
    "random_forest",
    "xgboost",
    "local_trend_ssm",
    "bvar_minnesota_8",
    "bvar_minnesota_20",
    "factor_pca_qd",
    "mixed_freq_dfm_md",
    "ensemble_avg_top3",
    "ensemble_weighted_top5",
    "auto_arima",
    "nyfed_nowcast_mqdfm",
    "ecb_nowcast_mqdfm",
    "chronos2",
]

SMOKE_PROFILE_MODELS: list[str] = [
    "naive_last",
    "drift",
    "auto_arima",
]

UNPROCESSED_STANDARD_MODELS: list[str] = [
    "naive_last",
    "drift",
    "auto_arima",
    "theta",
    "auto_ets",
    "local_trend_ssm",
    "random_forest",
    "xgboost",
    "bvar_minnesota_8",
    "factor_pca_qd",
    "mixed_freq_dfm_md",
]

PROCESSED_STANDARD_MODELS: list[str] = [
    "naive_last",
    "mean",
    "ar4",
    "auto_arima",
    "local_trend_ssm",
    "random_forest",
    "xgboost",
    "factor_pca_qd",
    "mixed_freq_dfm_md",
    "nyfed_nowcast_mqdfm",
    "ecb_nowcast_mqdfm",
]

TASK_META_FIELDS: tuple[str, ...] = (
    "release_stage",
    "release_metric",
    "target_transform",
    "target_units",
    "truth_item_id",
    "covariate_mode",
    "dataset_path",
)


def build_eval_arg_parser(
    *,
    description: str,
    default_target_transform: str,
    default_results_dir: str,
    default_model_set: ModelSet = "auto",
    default_models: Sequence[str] | None = None,
    default_task_prefix: str = "log_real_gdp",
    default_qd_vintage_panel: str = "data/panels/fred_qd_vintage_panel.parquet",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(PROFILE_CHOICES),
        default="standard",
        help="Preset run profile. Defaults to 'standard'.",
    )
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_SERIES_NAME)
    parser.add_argument(
        "--target_transform",
        type=str,
        default=default_target_transform,
        choices=sorted(SUPPORTED_TARGET_TRANSFORMS),
        help="Target transform used for model training targets.",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--num_windows", type=int, default=100)
    parser.add_argument("--metric", type=str, default="RMSE")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(default_models) if default_models is not None else None,
        help="Optional explicit model list. If omitted, --model_set defaults are used.",
    )
    parser.add_argument(
        "--model_set",
        type=str,
        choices=["auto", "ll", "g"],
        default=default_model_set,
        help="Model family to use when --models is not provided.",
    )
    parser.add_argument("--baseline_model", type=str, default="naive_last")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default=default_results_dir)
    parser.add_argument(
        "--exclude_years",
        type=int,
        nargs="*",
        default=[],
        help="Calendar years excluded from training/evaluation dataset. Default keeps all years.",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default=default_task_prefix,
        help="Prefix used for fev task names, e.g., log_real_gdp_h1.",
    )
    parser.add_argument(
        "--disable_covariates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable covariates and run univariate targets only.",
    )
    parser.add_argument(
        "--max_covariates",
        type=int,
        default=120,
        help="Max selected covariate columns passed to models.",
    )
    parser.add_argument(
        "--fast_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Speed-oriented mode: removes heavy default models (chronos2 and ensembles) "
            "and reduces heavy model hyperparameters."
        ),
    )
    parser.add_argument(
        "--source_series_candidates",
        nargs="+",
        default=list(DEFAULT_SOURCE_SERIES_CANDIDATES),
        help="Candidate real-GDP level series IDs used in historical vintage reconstruction.",
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
        help="Disable historical-vintage training and use a single finalized dataset for all windows.",
    )
    parser.add_argument(
        "--qd_vintage_panel",
        type=str,
        default=default_qd_vintage_panel,
        help="Fallback QD vintage panel parquet used when historical CSV vintages are unavailable.",
    )
    parser.add_argument(
        "--covariate_vintage",
        type=str,
        default="",
        help="Optional covariate vintage (YYYY-MM) to use for static covariate scaffold; default is latest.",
    )
    parser.add_argument(
        "--vintage_fallback_to_earliest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow windows earlier than the first vintage to fall back to the earliest available vintage "
            "instead of enforcing strict vintage coverage. Enabled by default; use "
            "--no-vintage_fallback_to_earliest for strict mode."
        ),
    )
    parser.add_argument(
        "--eval_release_metric",
        type=str,
        choices=["auto", "realtime_qoq_saar", "level"],
        default="auto",
        help="Release-table column family used for truth. 'auto' infers from target transform.",
    )
    parser.add_argument(
        "--eval_release_csv",
        type=str,
        default="data/panels/gdpc1_releases_first_second_third.csv",
        help="Path to gdpc1 release table CSV used as the benchmark truth source.",
    )
    parser.add_argument(
        "--eval_release_stage",
        type=str,
        choices=["first", "second", "third", "latest"],
        default=None,
        help="Legacy single-stage selector. If set, overrides --eval_release_stages.",
    )
    parser.add_argument(
        "--eval_release_stages",
        nargs="+",
        choices=["first", "second", "third", "latest"],
        default=None,
        help=(
            "Release stages to evaluate. "
            "Default is first/second/third for realtime_qoq_saar and first for level."
        ),
    )
    parser.add_argument(
        "--tail_periods",
        type=int,
        default=4,
        help="Tail quarters used for processed-covariate diagnostics.",
    )
    parser.add_argument(
        "--strict_truth_validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Validate that y_true matches the release-table truth column for first/second/third realtime SAAR. "
            "Defaults to enabled for processed runs and disabled otherwise."
        ),
    )
    return parser


def parse_args_with_provenance(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    provided_dests = _collect_provided_dests(parser=parser, argv=raw_argv)
    args = parser.parse_args(raw_argv)
    setattr(args, "_provided_dests", provided_dests)
    return args


def run_eval_pipeline(
    *,
    covariate_mode: CovariateMode,
    default_target_transform: Literal["log_level", "saar_growth"],
    model_set: ModelSet,
    cli_args: argparse.Namespace,
) -> None:
    mode = _normalize_covariate_mode(covariate_mode)
    args = cli_args
    profile = _apply_profile_defaults(args=args, covariate_mode=mode)

    np.random.seed(int(args.seed))

    target_transform = str(getattr(args, "target_transform", "") or default_target_transform)
    if target_transform not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(
            f"Unsupported target_transform={target_transform!r}. Supported={sorted(SUPPORTED_TARGET_TRANSFORMS)}"
        )

    eval_release_metric = _resolve_eval_release_metric(
        requested_metric=str(getattr(args, "eval_release_metric", "auto") or "auto"),
        target_transform=target_transform,
    )
    if eval_release_metric == "realtime_qoq_saar" and target_transform != "saar_growth":
        raise ValueError(
            "--eval_release_metric realtime_qoq_saar requires --target_transform saar_growth "
            "to keep training targets and evaluation truth on the same scale."
        )

    requested_models = resolve_model_names(
        getattr(args, "models", None),
        covariate_mode=mode,
        model_set=_normalize_model_set(str(getattr(args, "model_set", model_set) or model_set)),
    )
    fast_mode = bool(getattr(args, "fast_mode", False))
    if fast_mode:
        requested_models, dropped_models = _apply_fast_mode_model_list(requested_models)
        if dropped_models:
            print(f"Fast mode dropped models: {', '.join(dropped_models)}")
        if not requested_models:
            raise ValueError("Fast mode removed all selected models; provide at least one non-dropped model.")

    unknown_models = [m for m in requested_models if m not in available_models()]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Available models: {available_models()}")

    stages = _resolve_eval_release_stages(
        eval_release_metric=eval_release_metric,
        eval_release_stage=getattr(args, "eval_release_stage", None),
        eval_release_stages=getattr(args, "eval_release_stages", None),
    )

    results_dir = Path(str(args.results_dir)).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    years_to_exclude = sorted({int(y) for y in (getattr(args, "exclude_years", None) or [])})
    strict_truth_validation_arg = getattr(args, "strict_truth_validation", None)
    strict_truth_validation = (
        mode == "processed" if strict_truth_validation_arg is None else bool(strict_truth_validation_arg)
    )

    base_gdp_df, base_meta = build_release_target_scaffold(
        release_csv_path=args.eval_release_csv,
        target_series_name=args.target,
    )

    covariate_df = pd.DataFrame(columns=["timestamp"])
    covariate_columns: list[str] = []
    covariate_meta: dict[str, Any] = {}
    if not bool(getattr(args, "disable_covariates", False)):
        requested_vintage = str(getattr(args, "covariate_vintage", "") or "").strip()
        vintage_period = pd.Period(requested_vintage, freq="M") if requested_vintage else None

        covariate_df, covariate_meta = build_covariate_df(
            historical_qd_dir=args.historical_qd_dir,
            qd_panel_path=args.qd_vintage_panel,
            covariate_mode=mode,
            target_series_name=args.target,
            source_series_candidates=args.source_series_candidates,
            vintage_period=vintage_period,
        )
        covariate_df["timestamp"] = pd.to_datetime(covariate_df["timestamp"], errors="coerce")

    print(f"Base release-table dataset: {base_meta['release_csv_path']} (rows={base_meta['rows']})")
    print(f"Target series used: {args.target}")
    print(f"Profile: {profile}")
    print(f"Target transform: {target_transform}")
    print(f"Covariate mode: {mode}")
    print(f"Strict truth validation: {'enabled' if strict_truth_validation else 'disabled'}")
    print(f"Model set: {getattr(args, 'model_set', model_set)}")
    if requested_models == MODELS_LL_UNPROCESSED:
        print("Using default model list: MODELS_LL_UNPROCESSED")
    elif requested_models == MODELS_G_PROCESSED:
        print("Using default model list: MODELS_G_PROCESSED")
    else:
        print(f"Using explicit model list ({len(requested_models)} models)")

    if covariate_meta:
        print(
            "Covariate source: "
            f"{covariate_meta.get('source')} vintage={covariate_meta.get('selected_vintage')} "
            f"count={covariate_meta.get('covariate_count')}"
        )
    if fast_mode:
        print("Fast mode: enabled")

    all_tasks = []
    task_provider_by_name: dict[str, HistoricalQuarterlyVintageProvider] = {}
    task_meta_by_task_name: dict[str, dict[str, Any]] = {}

    for stage in stages:
        gdp_df, eval_meta = apply_gdpc1_release_truth_target(
            dataset_df=base_gdp_df,
            release_csv_path=args.eval_release_csv,
            release_stage=stage,
            release_metric=eval_release_metric,
            target_transform=target_transform,
        )

        gdp_df = gdp_df.copy()
        gdp_df["timestamp"] = pd.to_datetime(gdp_df["timestamp"], errors="coerce")

        if not bool(getattr(args, "disable_covariates", False)) and not covariate_df.empty:
            gdp_df = _merge_covariates_by_quarter(target_df=gdp_df, covariate_df=covariate_df)

        if not covariate_columns:
            covariate_columns = _select_covariate_columns(
                frame=gdp_df,
                target_col="target",
                max_covariates=int(getattr(args, "max_covariates", 120)),
            )

        keep_cols = ["item_id", "timestamp", "target", *covariate_columns]
        keep_cols = [c for c in keep_cols if c in gdp_df.columns]
        gdp_df = gdp_df[keep_cols].copy()

        original_freq = pd.infer_freq(pd.to_datetime(gdp_df["timestamp"], errors="coerce").dropna().sort_values())
        timestamp_mapping: dict[pd.Timestamp, pd.Timestamp] = {}

        before_n = len(gdp_df)
        gdp_filtered_actual = gdp_df.copy()
        if years_to_exclude:
            gdp_filtered_actual = exclude_years(gdp_filtered_actual, years=years_to_exclude)
        after_n = len(gdp_filtered_actual)

        gdp_task_df = gdp_filtered_actual.copy()
        if years_to_exclude:
            gdp_task_df = reindex_to_regular_frequency(gdp_task_df, freq=original_freq or "QS-DEC")
            timestamp_mapping = build_reindexed_to_actual_timestamp_map(
                actual_df=gdp_filtered_actual,
                reindexed_df=gdp_task_df,
                id_col="item_id",
                timestamp_col="timestamp",
            )

        min_ts, max_ts, has_2020 = _series_date_span(gdp_filtered_actual)
        print(
            f"Dataset span [{stage}]: {min_ts} -> {max_ts}, rows={len(gdp_filtered_actual)}, "
            f"contains_2020={has_2020}, excluded_years={years_to_exclude} (rows {before_n} -> {after_n})"
        )

        dataset_id = str(eval_meta.get("item_id", "") or str(args.target).strip().lower())
        local_dataset_path = results_dir / f"{dataset_id}_dataset.parquet"
        export_local_dataset_parquet(
            gdp_task_df,
            output_path=local_dataset_path,
            covariate_mode=mode,
        )

        print(
            "Evaluation truth source"
            f"{f' [{stage}]' if len(stages) > 1 else ''}: "
            f"{eval_meta['source']} (metric={eval_meta.get('release_metric')}, "
            f"stage={eval_meta.get('release_stage')}, column={eval_meta.get('release_column')})"
        )
        print(
            "Local task dataset parquet"
            f"{f' [{stage}]' if len(stages) > 1 else ''}: "
            f"{local_dataset_path}"
        )

        task_prefix = args.task_prefix if len(stages) == 1 else f"{args.task_prefix}_{stage}"
        tasks = make_gdp_tasks(
            dataset_path=str(local_dataset_path),
            dataset_config=None,
            id_col="id",
            timestamp_col="timestamp",
            target_col="target",
            known_dynamic_columns=[],
            past_dynamic_columns=covariate_columns,
            horizons=args.horizons,
            num_windows=args.num_windows,
            metric=args.metric,
            task_prefix=task_prefix,
        )

        vintage_provider: HistoricalQuarterlyVintageProvider | None = None
        if not bool(getattr(args, "disable_historical_vintages", False)):
            vintage_provider = HistoricalQuarterlyVintageProvider(
                historical_qd_dir=args.historical_qd_dir,
                target_series_name=args.target,
                target_transform=target_transform,
                source_series_candidates=args.source_series_candidates,
                covariate_columns=covariate_columns,
                include_covariates=bool(covariate_columns),
                apply_fred_transforms=(mode == "processed"),
                covariate_mode=mode,
                exclude_years_list=years_to_exclude,
                timestamp_mapping=timestamp_mapping,
                strict=not args.vintage_fallback_to_earliest,
                fallback_to_earliest=args.vintage_fallback_to_earliest,
                qd_panel_path=args.qd_vintage_panel,
            )
            print(
                "Historical FRED-QD vintages enabled"
                f"{f' [{stage}]' if len(stages) > 1 else ''}: "
                f"{vintage_provider.available_range_str()} ({len(vintage_provider.vintage_periods)} files) "
                f"from {vintage_provider.historical_qd_dir}"
            )

            adjusted_tasks = []
            for horizon, task in zip(args.horizons, tasks):
                compatible_windows = vintage_provider.compatible_window_count(task)
                if compatible_windows <= 0:
                    raise ValueError(
                        "No compatible windows for historical-vintage training at horizon "
                        f"{horizon} (eval={stage}). Try reducing --num_windows or disabling historical vintages."
                    )

                if compatible_windows < int(args.num_windows):
                    reason = (
                        "strict historical-vintage coverage and valid task windows"
                        if not args.vintage_fallback_to_earliest
                        else "valid task windows"
                    )
                    print(
                        f"Reducing eval={stage} horizon h={horizon} windows from "
                        f"{args.num_windows} to {compatible_windows} to enforce {reason}."
                    )
                    adjusted_task = make_gdp_tasks(
                        dataset_path=str(local_dataset_path),
                        dataset_config=None,
                        id_col="id",
                        timestamp_col="timestamp",
                        target_col="target",
                        known_dynamic_columns=[],
                        past_dynamic_columns=covariate_columns,
                        horizons=[int(horizon)],
                        num_windows=int(compatible_windows),
                        metric=args.metric,
                        task_prefix=task_prefix,
                    )[0]
                    adjusted_tasks.append(adjusted_task)
                else:
                    adjusted_tasks.append(task)
            tasks = adjusted_tasks

        all_tasks.extend(tasks)
        stage_for_meta = str(eval_meta.get("release_stage", stage))
        task_meta = {
            "release_stage": stage_for_meta,
            "release_metric": eval_meta.get("release_metric"),
            "target_transform": target_transform,
            "target_units": eval_meta.get("target_units"),
            "truth_item_id": eval_meta.get("item_id"),
            "covariate_mode": mode,
            "dataset_path": str(local_dataset_path),
        }
        for task in tasks:
            task_meta_by_task_name[str(task.task_name)] = dict(task_meta)

        if vintage_provider is not None:
            for task in tasks:
                task_provider_by_name[str(task.task_name)] = vintage_provider

        if mode == "processed" and covariate_columns:
            _print_processed_covariate_diagnostics(
                frame=gdp_filtered_actual,
                covariate_columns=covariate_columns,
                tail_periods=int(getattr(args, "tail_periods", 4)),
                stage=stage,
            )

    if not all_tasks:
        raise ValueError("No tasks were created for evaluation.")

    past_data_adapter = None
    if task_provider_by_name:

        def past_data_adapter(past_data, future_data, task, window_idx):  # type: ignore[no-redef]
            _ = future_data
            _ = window_idx
            provider = task_provider_by_name.get(str(getattr(task, "task_name", "")))
            if provider is None:
                return past_data
            return provider.adapt_past_data(past_data=past_data, task=task)

    models = build_models(model_names=requested_models, seed=int(args.seed))
    if fast_mode:
        _apply_fast_mode_hyperparameters(models=models)
    summaries, prediction_records = run_models_on_tasks_with_records(
        tasks=all_tasks,
        models=models,
        num_windows=int(args.num_windows),
        past_data_adapter=past_data_adapter,
    )

    _enrich_summaries_with_task_meta(summaries=summaries, task_meta_by_task_name=task_meta_by_task_name)

    summaries_jsonl = results_dir / "summaries.jsonl"
    summaries_csv = results_dir / "summaries.csv"
    save_summaries(summaries=summaries, jsonl_path=summaries_jsonl, csv_path=summaries_csv)
    timing_csv = results_dir / "timing.csv"
    timing_df = save_timing_report(summaries=summaries, output_csv=timing_csv)

    leaderboard_df, pairwise_df = generate_reports(
        summaries=summaries,
        results_dir=results_dir,
        baseline_model=args.baseline_model,
        seed=int(args.seed),
    )

    metric_col = infer_metric_column(pd.DataFrame(summaries))

    records_df = pd.DataFrame(prediction_records)
    if not records_df.empty:
        records_df["timestamp"] = pd.to_datetime(records_df["timestamp"], errors="coerce")
        records_df = records_df.dropna(subset=["timestamp"]).sort_values(
            ["task_name", "model_name", "window_idx", "item_id", "horizon_step"]
        )
    records_df = _enrich_prediction_records_with_task_meta(
        records_df=records_df,
        task_meta_by_task_name=task_meta_by_task_name,
    )
    validate_y_true_matches_release_table(
        records_df=records_df,
        release_csv_path=args.eval_release_csv,
        tolerance=1e-6,
        strict=strict_truth_validation,
    )

    predictions_csv = results_dir / "predictions_per_window.csv"
    records_df.to_csv(predictions_csv, index=False)

    kpi_df, kpi_subperiod_df = _compute_kpi_tables(
        records_df=records_df,
        target_transform=target_transform,
    )
    kpi_csv = results_dir / "kpi_metrics.csv"
    kpi_subperiod_csv = results_dir / "kpi_subperiod_metrics.csv"
    kpi_df.to_csv(kpi_csv, index=False)
    kpi_subperiod_df.to_csv(kpi_subperiod_csv, index=False)

    print(f"Wrote summaries: {summaries_jsonl}")
    print(f"Wrote summaries CSV: {summaries_csv}")
    print(f"Wrote timing report: {timing_csv}")
    print(f"Wrote leaderboard: {results_dir / 'leaderboard.csv'}")
    print(f"Wrote pairwise: {results_dir / 'pairwise.csv'}")
    print(f"Wrote prediction records: {predictions_csv}")
    print(f"Wrote KPI metrics: {kpi_csv}")
    print(f"Wrote KPI subperiod metrics: {kpi_subperiod_csv}")

    if not timing_df.empty:
        slow_models = (
            timing_df.groupby("model_name", as_index=False, dropna=False)
            .agg(inference_time_s=("inference_time_s", "sum"))
            .sort_values("inference_time_s", ascending=False)
            .head(10)
        )
        if not slow_models.empty:
            print("Top 10 slowest models by total inference time (seconds):")
            print(slow_models.to_string(index=False))

    if not leaderboard_df.empty and metric_col in leaderboard_df.columns:
        display_cols = [
            c for c in ["model_name", metric_col, "win_rate", "skill_vs_baseline"] if c in leaderboard_df.columns
        ]
        print("Top leaderboard rows:")
        print(leaderboard_df[display_cols].head(10).to_string(index=False))

    if not pairwise_df.empty:
        print("Top pairwise rows:")
        print(pairwise_df.head(10).to_string(index=False))


def _enrich_summaries_with_task_meta(
    summaries: list[dict[str, Any]],
    task_meta_by_task_name: dict[str, dict[str, Any]],
) -> None:
    for summary in summaries:
        task_name = str(summary.get("task_name", ""))
        task_meta = task_meta_by_task_name.get(task_name, {})
        for field in TASK_META_FIELDS:
            summary[field] = task_meta.get(field)


def _enrich_prediction_records_with_task_meta(
    records_df: pd.DataFrame,
    task_meta_by_task_name: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    out = records_df.copy()
    if "task_name" not in out.columns:
        for field in TASK_META_FIELDS:
            out[field] = pd.Series(index=out.index, dtype=object)
        return out

    task_meta = out["task_name"].map(lambda task_name: task_meta_by_task_name.get(str(task_name), {}))
    for field in TASK_META_FIELDS:
        out[field] = task_meta.map(lambda meta: meta.get(field) if isinstance(meta, dict) else None)
    return out


def validate_y_true_matches_release_table(
    records_df: pd.DataFrame,
    release_csv_path: str | Path,
    tolerance: float = 1e-6,
    strict: bool = True,
) -> dict[str, dict[str, float | int]]:
    if records_df.empty:
        print("Truth check skipped: prediction records are empty.")
        return {}

    missing_cols = [c for c in ["timestamp", "y_true", "release_stage"] if c not in records_df.columns]
    if missing_cols:
        msg = f"Truth check skipped: prediction records missing required columns {missing_cols}."
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}")
        return {}

    eval_rows = records_df.copy()
    if "release_metric" in eval_rows.columns:
        eval_rows = eval_rows.loc[eval_rows["release_metric"].astype(str) == "realtime_qoq_saar"].copy()
    if eval_rows.empty:
        print("Truth check skipped: no realtime_qoq_saar prediction rows.")
        return {}

    eval_rows["release_stage"] = eval_rows["release_stage"].astype(str).str.strip().str.lower()
    stage_order = ["first", "second", "third"]
    stages = [stage for stage in stage_order if stage in set(eval_rows["release_stage"].tolist())]
    if not stages:
        print("Truth check skipped: no first/second/third release stages in prediction rows.")
        return {}

    csv_path = Path(release_csv_path).expanduser().resolve()
    release_df = pd.read_csv(csv_path)
    if "observation_date" not in release_df.columns:
        msg = f"Release CSV missing required column 'observation_date': {csv_path}"
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}")
        return {}

    release_df = release_df.copy()
    release_df["observation_date"] = pd.to_datetime(release_df["observation_date"], errors="coerce")
    release_df = release_df.loc[release_df["observation_date"].notna()].copy()
    if release_df.empty:
        msg = f"Release CSV has no valid observation_date rows: {csv_path}"
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}")
        return {}

    release_df["quarter"] = pd.PeriodIndex(release_df["observation_date"], freq="Q-DEC")
    release_df = release_df.sort_values("observation_date").drop_duplicates(subset=["quarter"], keep="last")

    stats: dict[str, dict[str, float | int]] = {}
    failures: list[str] = []
    for stage in stages:
        expected_col = RELEASE_STAGE_TO_REALTIME_SAAR_COLUMN[stage]
        if expected_col not in release_df.columns:
            failures.append(f"stage={stage} missing release CSV column {expected_col!r}")
            continue

        stage_rows = eval_rows.loc[eval_rows["release_stage"] == stage].copy()
        if stage_rows.empty:
            continue

        stage_rows["timestamp"] = pd.to_datetime(stage_rows["timestamp"], errors="coerce")
        stage_rows = stage_rows.loc[stage_rows["timestamp"].notna()].copy()
        stage_rows["quarter"] = pd.PeriodIndex(stage_rows["timestamp"], freq="Q-DEC")

        expected = release_df[["quarter", expected_col]].copy()
        expected[expected_col] = pd.to_numeric(expected[expected_col], errors="coerce")
        merged = stage_rows.merge(expected.rename(columns={expected_col: "expected_y_true"}), on="quarter", how="left")

        actual = pd.to_numeric(merged["y_true"], errors="coerce").to_numpy(dtype=float)
        expected_values = pd.to_numeric(merged["expected_y_true"], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(actual) & np.isfinite(expected_values)
        abs_diff = np.abs(actual[valid] - expected_values[valid]) if valid.any() else np.array([], dtype=float)
        max_abs_diff = float(abs_diff.max()) if abs_diff.size else float("nan")
        mean_abs_diff = float(abs_diff.mean()) if abs_diff.size else float("nan")
        num_bad_diff = int((abs_diff > tolerance).sum()) if abs_diff.size else 0
        num_missing = int((~valid).sum())
        num_bad = num_bad_diff + num_missing

        stats[stage] = {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "num_bad": num_bad,
            "num_rows": int(len(stage_rows)),
        }

        max_abs_diff_str = "nan" if not np.isfinite(max_abs_diff) else f"{max_abs_diff:.12g}"
        mean_abs_diff_str = "nan" if not np.isfinite(mean_abs_diff) else f"{mean_abs_diff:.12g}"
        print(f"Truth check {stage}: max_abs_diff={max_abs_diff_str}, mean_abs_diff={mean_abs_diff_str}, bad={num_bad}")

        if num_bad > 0:
            failures.append(
                f"stage={stage} has {num_bad} mismatches/missing rows "
                f"(max_abs_diff={max_abs_diff_str}, tolerance={tolerance:g})"
            )

    if failures:
        msg = "Release-truth validation failed: " + "; ".join(failures)
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}")

    return stats


def _compute_kpi_tables(
    records_df: pd.DataFrame,
    target_transform: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if records_df.empty:
        cols = [
            "model_name",
            "task_name",
            "horizon",
            "sample_count",
            "rmse_target",
            "mae_target",
            "rmse_kpi_saar",
            "mae_kpi_saar",
        ]
        sub_cols = [*cols[:3], "period", *cols[3:]]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=sub_cols)

    df = records_df.copy()
    df["horizon"] = df["task_name"].map(_parse_task_horizon)
    df["target_error"] = pd.to_numeric(df["y_pred"], errors="coerce") - pd.to_numeric(df["y_true"], errors="coerce")

    if target_transform == "saar_growth":
        df["kpi_true_saar"] = pd.to_numeric(df["y_true"], errors="coerce")
        df["kpi_pred_saar"] = pd.to_numeric(df["y_pred"], errors="coerce")
    elif target_transform in {"log_level", "level"}:
        df = _attach_level_based_kpi(df=df, target_transform=target_transform)
    else:
        raise ValueError(f"Unsupported target_transform={target_transform!r}")

    df["kpi_error"] = pd.to_numeric(df["kpi_pred_saar"], errors="coerce") - pd.to_numeric(df["kpi_true_saar"], errors="coerce")
    df["period"] = _assign_period_slice(df["timestamp"])

    overall = _aggregate_metrics(df=df, group_cols=["model_name", "task_name", "horizon"])
    if not overall.empty:
        overall = overall.sort_values(["horizon", "rmse_kpi_saar", "model_name"], na_position="last").reset_index(
            drop=True
        )

    by_period = _aggregate_metrics(df=df, group_cols=["model_name", "task_name", "horizon", "period"])
    if not by_period.empty:
        by_period = by_period.sort_values(
            ["horizon", "period", "rmse_kpi_saar", "model_name"],
            na_position="last",
        ).reset_index(drop=True)

    return overall, by_period


def _attach_level_based_kpi(df: pd.DataFrame, target_transform: str) -> pd.DataFrame:
    out = df.copy()

    if target_transform == "log_level":
        out["true_level"] = np.exp(pd.to_numeric(out["y_true"], errors="coerce"))
        out["pred_level"] = np.exp(pd.to_numeric(out["y_pred"], errors="coerce"))
        out["last_level"] = np.exp(pd.to_numeric(out["last_observed_target"], errors="coerce"))
    else:
        out["true_level"] = pd.to_numeric(out["y_true"], errors="coerce")
        out["pred_level"] = pd.to_numeric(out["y_pred"], errors="coerce")
        out["last_level"] = pd.to_numeric(out["last_observed_target"], errors="coerce")

    rows: list[pd.DataFrame] = []
    group_cols = ["task_name", "model_name", "window_idx", "item_id"]
    for _, grp in out.groupby(group_cols, sort=False):
        g = grp.sort_values("horizon_step").copy()

        true_levels = pd.to_numeric(g["true_level"], errors="coerce").to_numpy(dtype=float)
        pred_levels = pd.to_numeric(g["pred_level"], errors="coerce").to_numpy(dtype=float)
        last_level = float(pd.to_numeric(g["last_level"], errors="coerce").iloc[0])

        prev_true = np.concatenate([[last_level], true_levels[:-1]]) if true_levels.size else np.array([], dtype=float)
        prev_pred = np.concatenate([[last_level], pred_levels[:-1]]) if pred_levels.size else np.array([], dtype=float)

        g["kpi_true_saar"] = _to_saar_growth_vector(current=true_levels, previous=prev_true)
        g["kpi_pred_saar"] = _to_saar_growth_vector(current=pred_levels, previous=prev_pred)
        rows.append(g)

    if not rows:
        out["kpi_true_saar"] = np.nan
        out["kpi_pred_saar"] = np.nan
        return out

    return pd.concat(rows, axis=0, ignore_index=True)


def _metric_row(group: pd.DataFrame) -> pd.Series:
    target_err = pd.to_numeric(group["target_error"], errors="coerce").to_numpy(dtype=float)
    kpi_err = pd.to_numeric(group["kpi_error"], errors="coerce").to_numpy(dtype=float)

    return pd.Series(
        {
            "sample_count": int(np.isfinite(target_err).sum()),
            "rmse_target": _rmse(target_err),
            "mae_target": _mae(target_err),
            "rmse_kpi_saar": _rmse(kpi_err),
            "mae_kpi_saar": _mae(kpi_err),
        }
    )


def _aggregate_metrics(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(list(group_cols), dropna=False, sort=False)
    for key, grp in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(group_cols, key_tuple))
        row.update(_metric_row(grp).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def _rmse(err: np.ndarray) -> float:
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def _mae(err: np.ndarray) -> float:
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.abs(finite)))


def _to_saar_growth_vector(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    cur = np.asarray(current, dtype=float)
    prev = np.asarray(previous, dtype=float)
    out = np.full(cur.shape[0], np.nan, dtype=float)

    valid = np.isfinite(cur) & np.isfinite(prev) & (prev > 0) & (cur > 0)
    out[valid] = (np.power(cur[valid] / prev[valid], 4.0) - 1.0) * 100.0
    return out


def _assign_period_slice(ts: pd.Series) -> pd.Series:
    ts_vals = pd.to_datetime(ts, errors="coerce")
    quarters = pd.PeriodIndex(ts_vals, freq="Q-DEC")

    pre_cut = pd.Period("2019Q4", freq="Q-DEC")
    covid_start = pd.Period("2020Q1", freq="Q-DEC")
    covid_end = pd.Period("2021Q1", freq="Q-DEC")

    labels: list[str] = []
    for q in quarters:
        if pd.isna(q):
            labels.append("unknown")
            continue
        if q <= pre_cut:
            labels.append("pre_covid")
            continue
        if covid_start <= q <= covid_end:
            labels.append("covid_2020Q1_2021Q1")
            continue
        labels.append("post_covid")

    return pd.Series(labels, index=ts.index)


def _resolve_eval_release_metric(requested_metric: str, target_transform: str) -> str:
    metric_norm = str(requested_metric).strip().lower()
    if metric_norm == "auto":
        return "realtime_qoq_saar" if target_transform == "saar_growth" else "level"
    if metric_norm not in {"realtime_qoq_saar", "level"}:
        raise ValueError(f"Unsupported eval_release_metric={requested_metric!r}")
    return metric_norm


def _resolve_eval_release_stages(
    *,
    eval_release_metric: str,
    eval_release_stage: str | None,
    eval_release_stages: Sequence[str] | None,
) -> list[str]:
    if eval_release_stage:
        stages = [str(eval_release_stage).strip().lower()]
    elif eval_release_stages:
        stages = [str(s).strip().lower() for s in eval_release_stages]
    else:
        stages = ["first", "second", "third"] if eval_release_metric == "realtime_qoq_saar" else ["first"]

    stages = list(dict.fromkeys(stages))
    if eval_release_metric == "realtime_qoq_saar":
        invalid = [s for s in stages if s not in {"first", "second", "third"}]
        if invalid:
            raise ValueError(
                "Realtime qoq SAAR evaluation only supports release stages first/second/third. "
                f"Received: {invalid}"
            )
    return stages


def _merge_covariates_by_quarter(target_df: pd.DataFrame, covariate_df: pd.DataFrame) -> pd.DataFrame:
    if target_df.empty or covariate_df.empty:
        return target_df.copy()

    target = target_df.copy()
    cov = covariate_df.copy()

    target["timestamp"] = pd.to_datetime(target["timestamp"], errors="coerce")
    cov["timestamp"] = pd.to_datetime(cov["timestamp"], errors="coerce")

    target = target.dropna(subset=["timestamp"])
    cov = cov.dropna(subset=["timestamp"])
    if target.empty or cov.empty:
        return target

    target["quarter"] = pd.PeriodIndex(target["timestamp"], freq="Q-DEC")
    cov["quarter"] = pd.PeriodIndex(cov["timestamp"], freq="Q-DEC")
    cov = cov.sort_values("timestamp").groupby("quarter", as_index=False).last()

    drop_cols = {"timestamp", "quarter"}
    cov_cols = [c for c in cov.columns if c not in drop_cols]
    merged = target.merge(cov[["quarter", *cov_cols]], on="quarter", how="left")
    merged = merged.drop(columns=["quarter"])
    return merged


def _select_covariate_columns(frame: pd.DataFrame, target_col: str, max_covariates: int) -> list[str]:
    exclude = {"item_id", "timestamp", target_col}
    candidates: list[str] = []
    for col in frame.columns:
        if col in exclude:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() < 8:
            continue
        candidates.append(col)
    ordered = sorted(candidates, key=lambda v: str(v))
    return ordered[: max(0, int(max_covariates))]


def _apply_fast_mode_model_list(model_names: Sequence[str]) -> tuple[list[str], list[str]]:
    # Keep speed-focused defaults by dropping known-heavy models.
    drop_set = {"chronos2", "ensemble_avg_top3", "ensemble_weighted_top5"}
    kept: list[str] = []
    dropped: list[str] = []
    for name in model_names:
        if name in drop_set:
            dropped.append(name)
            continue
        kept.append(name)
    return kept, dropped


def _apply_fast_mode_hyperparameters(models: dict[str, Any]) -> None:
    if "random_forest" in models and hasattr(models["random_forest"], "n_estimators"):
        models["random_forest"].n_estimators = 50
    if "xgboost" in models and hasattr(models["xgboost"], "n_estimators"):
        models["xgboost"].n_estimators = 100
    if "local_trend_ssm" in models and hasattr(models["local_trend_ssm"], "maxiter"):
        models["local_trend_ssm"].maxiter = 50


def _collect_provided_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            option_to_dest[opt] = str(action.dest)

    provided: set[str] = set()
    for token in argv:
        if not token.startswith("-") or token == "-":
            continue
        opt = token.split("=", 1)[0]
        dest = option_to_dest.get(opt)
        if dest:
            provided.add(dest)
    return provided


def _profile_results_dir(covariate_mode: CovariateMode, profile: Profile) -> str:
    if profile == "full":
        return "results"
    mode_prefix = "unprocessed" if covariate_mode == "unprocessed" else "processed"
    return f"results/{mode_prefix}_{profile}"


def _apply_profile_defaults(args: argparse.Namespace, covariate_mode: CovariateMode) -> Profile:
    raw_profile = str(getattr(args, "profile", "standard") or "standard").strip().lower()
    if raw_profile not in PROFILE_CHOICES:
        raise ValueError(f"Unsupported profile={raw_profile!r}. Supported={PROFILE_CHOICES}.")
    profile = cast(Profile, raw_profile)

    provided = set(getattr(args, "_provided_dests", set()) or set())

    def _set_if_not_provided(dest: str, value: Any) -> None:
        if dest in provided:
            return
        setattr(args, dest, value)

    if profile == "smoke":
        _set_if_not_provided("horizons", [1, 4])
        _set_if_not_provided("num_windows", 10)
        _set_if_not_provided("metric", "RMSE")
        _set_if_not_provided("models", list(SMOKE_PROFILE_MODELS))
        _set_if_not_provided("disable_historical_vintages", True)

    elif profile == "standard":
        _set_if_not_provided("horizons", [1, 2, 4])
        _set_if_not_provided("metric", "RMSE")
        _set_if_not_provided("disable_historical_vintages", True)
        if covariate_mode == "unprocessed":
            _set_if_not_provided("target_transform", "log_level")
            _set_if_not_provided("num_windows", 60)
            _set_if_not_provided("models", list(UNPROCESSED_STANDARD_MODELS))
        else:
            _set_if_not_provided("target_transform", "saar_growth")
            _set_if_not_provided("num_windows", 40)
            _set_if_not_provided("models", list(PROCESSED_STANDARD_MODELS))

    else:  # profile == "full"
        _set_if_not_provided("horizons", [1, 2, 3, 4])
        _set_if_not_provided("num_windows", 80)
        _set_if_not_provided("metric", "RMSE")
        _set_if_not_provided("models", list(FULL_PROFILE_MODELS))

    _set_if_not_provided("results_dir", _profile_results_dir(covariate_mode=covariate_mode, profile=profile))
    return profile


def _parse_task_horizon(task_name: str) -> int:
    m = re.search(r"_h(\d+)$", str(task_name))
    if not m:
        return -1
    return int(m.group(1))


def _normalize_covariate_mode(mode: str) -> CovariateMode:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"unprocessed", "processed"}:
        raise ValueError("covariate_mode must be one of {'unprocessed', 'processed'}")
    return cast(CovariateMode, mode_norm)


def _normalize_model_set(value: str) -> ModelSet:
    v = str(value).strip().lower()
    if v not in {"auto", "ll", "g"}:
        raise ValueError("model_set must be one of {'auto','ll','g'}")
    return cast(ModelSet, v)


def _series_date_span(df: pd.DataFrame) -> tuple[str, str, bool]:
    if df.empty or "timestamp" not in df.columns:
        return "NA", "NA", False
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna().sort_values()
    if ts.empty:
        return "NA", "NA", False
    has_2020 = bool((ts.dt.year == 2020).any())
    return ts.iloc[0].date().isoformat(), ts.iloc[-1].date().isoformat(), has_2020


def _print_processed_covariate_diagnostics(
    *,
    frame: pd.DataFrame,
    covariate_columns: Sequence[str],
    tail_periods: int,
    stage: str,
) -> None:
    if not covariate_columns:
        return

    ordered = frame.sort_values("timestamp").copy()
    if ordered.empty:
        return

    tail_n = max(1, int(tail_periods))
    tail = ordered.tail(tail_n)
    tail_missing_pct = (
        tail[list(covariate_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .to_numpy(dtype=float)
        .mean()
        * 100.0
    )
    print(
        f"Processed covariate diagnostic [{stage}]: tail_missing_pct={tail_missing_pct:.2f}% "
        f"over last {tail_n} quarter(s) across {len(covariate_columns)} covariates"
    )

    extremes: list[tuple[str, float, float]] = []
    for col in covariate_columns:
        s = pd.to_numeric(ordered[col], errors="coerce")
        hist = s.dropna()
        if hist.shape[0] < 20:
            continue
        tail_val = float(hist.iloc[-1])
        med = float(hist.median())
        mad = float(np.median(np.abs(hist.to_numpy(dtype=float) - med)))
        scale = mad if mad > 1e-9 else float(hist.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-9:
            continue
        z = abs((tail_val - med) / scale)
        if np.isfinite(z) and z >= 8.0:
            extremes.append((str(col), float(z), tail_val))

    if extremes:
        extremes = sorted(extremes, key=lambda row: row[1], reverse=True)[:10]
        rendered = ", ".join([f"{c}(z~{z:.1f}, tail={v:.3g})" for c, z, v in extremes])
        print(f"Processed covariate extreme tails [{stage}]: {rendered}")
