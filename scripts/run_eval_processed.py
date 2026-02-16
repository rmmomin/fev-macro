#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.eval_runner import build_eval_arg_parser, run_eval_pipeline  # noqa: E402


def parse_args():
    parser = build_eval_arg_parser(
        description=(
            "Run evaluation with PROCESSED covariates (FRED transform codes / stationary-ish inputs) "
            "and GDP q/q SAAR target by default."
        ),
        default_target_transform="saar_growth",
        default_results_dir="results/bench_processed_g",
        default_model_set="auto",
        default_models=None,
        default_task_prefix="gdp_saar",
        default_qd_vintage_panel="data/panels/fred_qd_vintage_panel_processed.parquet",
    )
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
