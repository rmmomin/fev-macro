#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.eval_runner import (  # noqa: E402
    FULL_PROFILE_MODELS,
    build_eval_arg_parser,
    parse_args_with_provenance,
    run_eval_pipeline,
)


LEGACY_RUN_EVAL_MODELS = list(FULL_PROFILE_MODELS)


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

    return parse_args_with_provenance(parser)


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
