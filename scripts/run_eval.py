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
    build_gdp_saar_growth_series,
    export_local_dataset_parquet,
    find_gdp_column_candidates,
    load_fev_dataset,
)
from fev_macro.eval import run_models_on_tasks, save_summaries  # noqa: E402
from fev_macro.models import available_models, build_models  # noqa: E402
from fev_macro.report import generate_reports, infer_metric_column  # noqa: E402
from fev_macro.tasks import make_gdp_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible macro-forecast eval harness (US real GDP q/q SAAR).")
    parser.add_argument("--dataset_path", type=str, default="autogluon/fev_datasets")
    parser.add_argument("--dataset_config", type=str, default="fred_qd_2025")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--target", type=str, default="GDP_SAAR")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num_windows", type=int, default=80)
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
            "auto_arima",
            "chronos2",
        ],
    )
    parser.add_argument("--baseline_model", type=str, default="naive_last")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    requested_models = args.models
    unknown_models = [m for m in requested_models if m not in available_models()]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Available models: {available_models()}")

    dataset = load_fev_dataset(
        config=args.dataset_config,
        dataset_path=args.dataset_path,
        dataset_revision=args.dataset_revision,
    )

    candidates = find_gdp_column_candidates(dataset)
    gdp_df, gdp_meta = build_gdp_saar_growth_series(
        dataset=dataset,
        target_series_name=args.target,
        source_series_candidates=args.source_series_candidates,
        include_covariates=not args.disable_covariates,
    )
    covariate_columns = list(gdp_meta.get("covariate_columns", []))

    results_dir = Path(args.results_dir)
    local_dataset_path = results_dir / f"{args.target.lower()}_dataset.parquet"
    export_local_dataset_parquet(gdp_df, output_path=local_dataset_path)

    print(f"Target series used: {gdp_meta['target_series']}")
    print(f"Computed from level series: {gdp_meta['computed']} (source={gdp_meta['source_series']})")
    print(f"Local task dataset parquet: {local_dataset_path}")
    print(f"GDP candidate IDs: {candidates['id_candidates'][:20]}")
    print(f"Covariates enabled: {not args.disable_covariates}; count={len(covariate_columns)}")

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
    )

    models = build_models(model_names=requested_models, seed=args.seed)

    summaries = run_models_on_tasks(tasks=tasks, models=models, num_windows=args.num_windows)

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
