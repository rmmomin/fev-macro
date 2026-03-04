from __future__ import annotations

import argparse
from pathlib import Path

from .boe_adapter import export_to_boe_schema
from .boe_eval import run_boe_eval
from .boe_plots import make_plots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BoE forecast-evaluation helpers for fev-macro outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export fev-macro predictions to BoE schema CSV files.")
    export_parser.add_argument("--predictions_csv", required=True, type=Path)
    export_parser.add_argument("--release_table_csv", default=None, type=Path)
    export_parser.add_argument("--out_dir", default=Path("boe_export"), type=Path)
    export_parser.add_argument("--truth", default="first", choices=["first", "second", "third"])
    export_parser.add_argument("--variable", default="GDPC1")
    export_parser.add_argument("--metric", default="levels")
    export_parser.add_argument("--forecast_value_col", default="y_hat_level")

    eval_parser = subparsers.add_parser("eval", help="Run BoE accuracy, DM, rolling DM, and fluctuation tests.")
    eval_parser.add_argument("--forecasts_csv", required=True, type=Path)
    eval_parser.add_argument("--outturns_csv", required=True, type=Path)
    eval_parser.add_argument("--out_dir", default=Path("boe_results"), type=Path)
    eval_parser.add_argument("--k", type=int, default=0)
    eval_parser.add_argument("--benchmark_model", required=True)
    eval_parser.add_argument("--variable", default=None)
    eval_parser.add_argument("--same_date_range", action="store_true")
    eval_parser.add_argument("--add_boe_random_walk", action="store_true")
    eval_parser.add_argument("--add_boe_ar_p", action="store_true")

    plots_parser = subparsers.add_parser("plots", help="Generate BoE diagnostic plot PNG files.")
    plots_parser.add_argument("--forecasts_csv", required=True, type=Path)
    plots_parser.add_argument("--outturns_csv", required=True, type=Path)
    plots_parser.add_argument("--out_dir", default=Path("boe_plots"), type=Path)
    plots_parser.add_argument("--variable", required=True)
    plots_parser.add_argument("--source", required=True)
    plots_parser.add_argument("--metric", default="levels", choices=["levels", "pop", "yoy"])
    plots_parser.add_argument("--frequency", default="Q", choices=["Q", "M"])
    plots_parser.add_argument("--k", type=int, default=0)
    plots_parser.add_argument("--horizon", type=int, default=0)
    plots_parser.add_argument("--ma_window", type=int, default=4)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "export":
        forecasts_path, outturns_path = export_to_boe_schema(
            predictions_csv=args.predictions_csv,
            release_table_csv=args.release_table_csv,
            out_dir=args.out_dir,
            truth=args.truth,
            variable=args.variable,
            metric=args.metric,
            forecast_value_col=args.forecast_value_col,
        )
        print(f"Wrote forecasts -> {forecasts_path}")
        print(f"Wrote outturns  -> {outturns_path}")
        return 0

    if args.command == "eval":
        run_boe_eval(
            forecasts_csv=args.forecasts_csv,
            outturns_csv=args.outturns_csv,
            out_dir=args.out_dir,
            k=args.k,
            benchmark_model=args.benchmark_model,
            variable=args.variable,
            same_date_range=args.same_date_range,
            add_boe_random_walk=args.add_boe_random_walk,
            add_boe_ar_p=args.add_boe_ar_p,
        )
        print(f"Wrote evaluation outputs to {args.out_dir}")
        return 0

    if args.command == "plots":
        make_plots(
            forecasts_csv=args.forecasts_csv,
            outturns_csv=args.outturns_csv,
            out_dir=args.out_dir,
            variable=args.variable,
            source=args.source,
            metric=args.metric,
            frequency=args.frequency,
            k=args.k,
            horizon=args.horizon,
            ma_window=args.ma_window,
        )
        print(f"Wrote plot outputs to {args.out_dir}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
