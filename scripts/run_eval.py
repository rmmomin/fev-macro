#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.eval_runner import build_eval_arg_parser, run_eval_pipeline  # noqa: E402


LEGACY_RUN_EVAL_MODELS = [
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
]


def parse_args():
    parser = build_eval_arg_parser(
        description="Run reproducible macro-forecast eval harness (US real GDP).",
        default_target_transform="saar_growth",
        default_results_dir="results",
        default_model_set="auto",
        default_models=LEGACY_RUN_EVAL_MODELS,
        default_task_prefix="log_real_gdp",
        default_qd_vintage_panel="data/panels/fred_qd_vintage_panel_processed.parquet",
    )

    # Preserve historical defaults for run_eval.py while delegating to shared runner.
    parser.set_defaults(disable_covariates=True)
    parser.set_defaults(eval_release_metric="realtime_qoq_saar")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eval_pipeline(
        covariate_mode="processed",
        default_target_transform="saar_growth",
        model_set="auto",
        cli_args=args,
    )


if __name__ == "__main__":
    main()
