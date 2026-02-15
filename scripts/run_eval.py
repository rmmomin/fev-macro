#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import (  # noqa: E402
    DEFAULT_SOURCE_SERIES_CANDIDATES,
    DEFAULT_TARGET_SERIES_NAME,
    HistoricalQuarterlyVintageProvider,
    SUPPORTED_TARGET_TRANSFORMS,
    apply_gdpc1_release_truth_target,
    build_reindexed_to_actual_timestamp_map,
    build_real_gdp_target_series,
    exclude_years,
    export_local_dataset_parquet,
    find_gdp_column_candidates,
    load_fred_qd_transform_codes,
    load_fev_dataset,
    reindex_to_regular_frequency,
)
from fev_macro.eval import run_models_on_tasks, save_summaries  # noqa: E402
from fev_macro.models import available_models, build_models  # noqa: E402
from fev_macro.report import generate_reports, infer_metric_column  # noqa: E402
from fev_macro.tasks import make_gdp_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible macro-forecast eval harness (US real GDP).")
    parser.add_argument("--dataset_path", type=str, default="autogluon/fev_datasets")
    parser.add_argument("--dataset_config", type=str, default="fred_qd_2025")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_SERIES_NAME)
    parser.add_argument(
        "--target_transform",
        type=str,
        default="saar_growth",
        choices=sorted(SUPPORTED_TARGET_TRANSFORMS),
        help="Target transform when target series is not directly available.",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--num_windows", type=int, default=100)
    parser.add_argument("--metric", type=str, default="RMSE")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "naive_last",
            "mean",
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
            "chronos2",
        ],
    )
    parser.add_argument("--baseline_model", type=str, default="naive_last")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--exclude_years",
        type=int,
        nargs="*",
        default=[2020],
        help="Calendar years excluded from training/evaluation dataset.",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="log_real_gdp",
        help="Prefix used for fev task names, e.g., log_real_gdp_h1.",
    )
    parser.add_argument(
        "--disable_covariates",
        action="store_true",
        help="If set, run univariate-only target forecasting without extra macro covariates.",
    )
    parser.add_argument(
        "--source_series_candidates",
        nargs="+",
        default=list(DEFAULT_SOURCE_SERIES_CANDIDATES),
        help="Candidate real-GDP level series IDs used when target series is absent.",
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
        default="data/panels/fred_qd_vintage_panel.parquet",
        help=(
            "Fallback QD vintage panel parquet used when --historical_qd_dir CSV vintages are unavailable. "
            "Useful after running make panel-qd."
        ),
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
        help=(
            "Allow windows earlier than the first vintage to fall back to the earliest available vintage "
            "instead of enforcing strict vintage coverage."
        ),
    )
    parser.add_argument(
        "--eval_truth_source",
        type=str,
        choices=["release_csv", "dataset"],
        default="release_csv",
        help=(
            "Ground-truth source for future-window scoring. "
            "'release_csv' uses gdpc1 release table columns; "
            "'dataset' keeps the finalized HF target path."
        ),
    )
    parser.add_argument(
        "--eval_release_metric",
        type=str,
        choices=["realtime_qoq_saar", "level"],
        default="realtime_qoq_saar",
        help=(
            "Release-table column family used for truth when --eval_truth_source=release_csv. "
            "'realtime_qoq_saar' maps to qoq_saar_growth_realtime_{first,second,third}_pct."
        ),
    )
    parser.add_argument(
        "--eval_release_csv",
        type=str,
        default="data/panels/gdpc1_releases_first_second_third.csv",
        help="Path to gdpc1 release table CSV used when --eval_truth_source=release_csv.",
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
    return parser.parse_args()


def _resolve_eval_release_stages(args: argparse.Namespace) -> list[str]:
    if args.eval_release_stage:
        stages = [str(args.eval_release_stage).strip().lower()]
    elif args.eval_release_stages:
        stages = [str(s).strip().lower() for s in args.eval_release_stages]
    else:
        if args.eval_release_metric == "realtime_qoq_saar":
            stages = ["first", "second", "third"]
        else:
            stages = ["first"]

    stages = list(dict.fromkeys(stages))
    if args.eval_release_metric == "realtime_qoq_saar":
        invalid = [s for s in stages if s not in {"first", "second", "third"}]
        if invalid:
            raise ValueError(
                "Realtime qoq SAAR evaluation only supports release stages first/second/third. "
                f"Received: {invalid}"
            )
    return stages


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    requested_models = args.models
    unknown_models = [m for m in requested_models if m not in available_models()]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Available models: {available_models()}")

    if args.eval_truth_source == "release_csv" and args.eval_release_metric == "realtime_qoq_saar":
        if args.target_transform != "saar_growth":
            raise ValueError(
                "--eval_release_metric realtime_qoq_saar requires --target_transform saar_growth "
                "to keep training targets and evaluation truth on the same scale."
            )

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

    candidates = find_gdp_column_candidates(dataset)
    base_gdp_df, gdp_meta = build_real_gdp_target_series(
        dataset=dataset,
        target_series_name=args.target,
        target_transform=args.target_transform,
        source_series_candidates=args.source_series_candidates,
        include_covariates=not args.disable_covariates,
        apply_fred_transforms=apply_fred_transforms,
        fred_transform_codes=fred_transform_codes,
    )

    eval_datasets: list[dict[str, object]] = [{"label": "dataset", "df": base_gdp_df, "meta": {"source": "dataset"}}]
    if args.eval_truth_source == "release_csv":
        stages = _resolve_eval_release_stages(args)
        eval_datasets = []
        for stage in stages:
            stage_df, stage_meta = apply_gdpc1_release_truth_target(
                dataset_df=base_gdp_df,
                release_csv_path=args.eval_release_csv,
                release_stage=stage,
                release_metric=args.eval_release_metric,
                target_transform=args.target_transform,
            )
            eval_datasets.append({"label": stage, "df": stage_df, "meta": stage_meta})

    covariate_columns = list(gdp_meta.get("covariate_columns", []))
    years_to_exclude = sorted({int(y) for y in (args.exclude_years or [])})
    results_dir = Path(args.results_dir)

    print(f"Base dataset: {args.dataset_path} (config={args.dataset_config})")
    print(f"Target series used: {gdp_meta['target_series']}")
    print(f"Target transform: {gdp_meta.get('target_transform', args.target_transform)}")
    print(f"Computed from level series: {gdp_meta['computed']} (source={gdp_meta['source_series']})")
    print(f"GDP candidate IDs: {candidates['id_candidates'][:20]}")
    print(f"Covariates enabled: {not args.disable_covariates}; count={len(covariate_columns)}")
    print(
        "FRED transforms enabled: "
        f"{apply_fred_transforms}; transform codes loaded={len(fred_transform_codes)}; "
        f"covariates transformed={len(gdp_meta.get('transformed_covariates', []))}"
    )

    all_tasks = []
    task_provider_by_name: dict[str, HistoricalQuarterlyVintageProvider] = {}

    for spec in eval_datasets:
        eval_label = str(spec["label"])
        eval_meta = dict(spec.get("meta", {}))
        gdp_df = pd.DataFrame(spec["df"]).copy()

        original_freq = pd.infer_freq(pd.to_datetime(gdp_df["timestamp"], errors="coerce").dropna().sort_values())
        timestamp_mapping: dict[pd.Timestamp, pd.Timestamp] = {}
        if years_to_exclude:
            before_n = len(gdp_df)
            gdp_df = exclude_years(gdp_df, years=years_to_exclude)
            gdp_filtered_actual = gdp_df.copy()
            if original_freq:
                gdp_df = reindex_to_regular_frequency(gdp_df, freq=original_freq)
                timestamp_mapping = build_reindexed_to_actual_timestamp_map(
                    actual_df=gdp_filtered_actual,
                    reindexed_df=gdp_df,
                    id_col="item_id",
                    timestamp_col="timestamp",
                )
            after_n = len(gdp_df)
        else:
            before_n = after_n = len(gdp_df)

        dataset_suffix = "" if len(eval_datasets) == 1 else f"_{eval_label}"
        local_dataset_path = results_dir / f"{args.target.lower()}_dataset{dataset_suffix}.parquet"
        export_local_dataset_parquet(gdp_df, output_path=local_dataset_path)

        if eval_meta.get("source") == "gdpc1_release_csv":
            print(
                "Evaluation truth source"
                f"{f' [{eval_label}]' if len(eval_datasets) > 1 else ''}: "
                f"{eval_meta['source']} (metric={eval_meta.get('release_metric')}, "
                f"stage={eval_meta.get('release_stage')}, column={eval_meta.get('release_column')})"
            )
            print(
                "Evaluation release table"
                f"{f' [{eval_label}]' if len(eval_datasets) > 1 else ''}: "
                f"{eval_meta.get('release_csv_path')} "
                f"(quarters={eval_meta.get('release_quarters_available')}, "
                f"rows_with_release_target={eval_meta.get('rows_with_release_target')})"
            )
        else:
            print(
                "Evaluation truth source"
                f"{f' [{eval_label}]' if len(eval_datasets) > 1 else ''}: "
                "dataset finalized target path"
            )

        print(
            "Local task dataset parquet"
            f"{f' [{eval_label}]' if len(eval_datasets) > 1 else ''}: "
            f"{local_dataset_path}"
        )
        print(
            "Excluded years"
            f"{f' [{eval_label}]' if len(eval_datasets) > 1 else ''}: "
            f"{years_to_exclude} (rows {before_n} -> {after_n})"
        )
        if years_to_exclude:
            print(
                "Reindexed filtered series to regular frequency"
                f"{f' [{eval_label}]' if len(eval_datasets) > 1 else ''}: "
                f"{original_freq or 'QS-DEC'}"
            )

        task_prefix = args.task_prefix if len(eval_datasets) == 1 else f"{args.task_prefix}_{eval_label}"
        tasks = make_gdp_tasks(
            dataset_path=str(local_dataset_path),
            dataset_config=None,
            id_col="id",
            timestamp_col="timestamp",
            target_col="target",
            known_dynamic_columns=covariate_columns,
            horizons=args.horizons,
            num_windows=args.num_windows,
            metric=args.metric,
            task_prefix=task_prefix,
        )

        vintage_provider: HistoricalQuarterlyVintageProvider | None = None
        if not args.disable_historical_vintages:
            vintage_provider = HistoricalQuarterlyVintageProvider(
                historical_qd_dir=args.historical_qd_dir,
                target_series_name=args.target,
                target_transform=args.target_transform,
                source_series_candidates=args.source_series_candidates,
                covariate_columns=covariate_columns,
                include_covariates=not args.disable_covariates,
                apply_fred_transforms=apply_fred_transforms,
                exclude_years_list=years_to_exclude,
                timestamp_mapping=timestamp_mapping,
                strict=not args.vintage_fallback_to_earliest,
                fallback_to_earliest=args.vintage_fallback_to_earliest,
                qd_panel_path=args.qd_vintage_panel,
            )
            print(
                "Historical FRED-QD vintages enabled"
                f"{f' [{eval_label}]' if len(eval_datasets) > 1 else ''}: "
                f"{vintage_provider.available_range_str()} ({len(vintage_provider.vintage_periods)} files) "
                f"from {vintage_provider.historical_qd_dir}"
            )

            if not args.vintage_fallback_to_earliest:
                adjusted_tasks = []
                for horizon, task in zip(args.horizons, tasks):
                    compatible_windows = vintage_provider.compatible_window_count(task)
                    if compatible_windows <= 0:
                        raise ValueError(
                            "No compatible windows for historical-vintage training at horizon "
                            f"{horizon} (eval={eval_label}). Try reducing --num_windows or enable "
                            "--vintage_fallback_to_earliest."
                        )

                    if compatible_windows < int(args.num_windows):
                        print(
                            f"Reducing eval={eval_label} horizon h={horizon} windows from "
                            f"{args.num_windows} to {compatible_windows} to enforce strict "
                            "historical-vintage coverage."
                        )
                        adjusted_task = make_gdp_tasks(
                            dataset_path=str(local_dataset_path),
                            dataset_config=None,
                            id_col="id",
                            timestamp_col="timestamp",
                            target_col="target",
                            known_dynamic_columns=covariate_columns,
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
        if vintage_provider is not None:
            for task in tasks:
                task_provider_by_name[str(task.task_name)] = vintage_provider

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

    models = build_models(model_names=requested_models, seed=args.seed)

    summaries = run_models_on_tasks(
        tasks=all_tasks,
        models=models,
        num_windows=args.num_windows,
        past_data_adapter=past_data_adapter,
    )

    summaries_jsonl = results_dir / "summaries.jsonl"
    summaries_csv = results_dir / "summaries.csv"
    save_summaries(summaries=summaries, jsonl_path=summaries_jsonl, csv_path=summaries_csv)

    leaderboard_df, pairwise_df = generate_reports(
        summaries=summaries,
        results_dir=results_dir,
        baseline_model=args.baseline_model,
        seed=args.seed,
    )

    metric_col = infer_metric_column(pd.DataFrame(summaries))

    print(f"Wrote summaries: {summaries_jsonl}")
    print(f"Wrote summaries CSV: {summaries_csv}")
    print(f"Wrote leaderboard: {results_dir / 'leaderboard.csv'}")
    print(f"Wrote pairwise: {results_dir / 'pairwise.csv'}")

    if not leaderboard_df.empty and metric_col in leaderboard_df.columns:
        display_cols = [c for c in ["model_name", metric_col, "win_rate", "skill_vs_baseline"] if c in leaderboard_df.columns]
        print("Top leaderboard rows:")
        print(leaderboard_df[display_cols].head(10).to_string(index=False))

    if not pairwise_df.empty:
        print("Top pairwise rows:")
        print(pairwise_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
