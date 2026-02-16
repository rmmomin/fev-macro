#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.eval_runner import build_eval_arg_parser, parse_args_with_provenance, run_eval_pipeline  # noqa: E402


def parse_args():
    parser = build_eval_arg_parser(
        description=(
            "Run evaluation with UNPROCESSED covariates (raw FRED-QD values) "
            "and log-level GDP target by default."
        ),
        default_target_transform="log_level",
        default_results_dir="results/bench_unprocessed_ll",
        default_model_set="auto",
        default_models=None,
        default_task_prefix="log_real_gdp",
        default_qd_vintage_panel="data/panels/fred_qd_vintage_panel.parquet",
    )
    return parse_args_with_provenance(parser)


def main() -> None:
    args = parse_args()
    run_eval_pipeline(
        covariate_mode="unprocessed",
        default_target_transform="log_level",
        model_set="auto",
        cli_args=args,
    )


if __name__ == "__main__":
    main()
