#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.realtime_oos import (  # noqa: E402
    BUILTIN_MODELS,
    compute_metrics,
    load_release_table,
    load_vintage_panel,
    run_backtest,
)
from fev_macro.models import available_models  # noqa: E402
from fev_macro.realtime_runner import (  # noqa: E402
    normalize_mode,
    resolve_baseline_model,
    resolve_md_panel_path,
    resolve_models as resolve_mode_models,
    resolve_oos_output_dir,
    resolve_qd_panel_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run vintage-correct rolling OOS GDP evaluation scored against first-release GDP levels "
            "and SAAR growth derived from first-release levels."
        )
    )
    parser.add_argument(
        "--release_csv",
        type=str,
        default="",
        help=(
            "Path to gdpc1 release table CSV. "
            "If omitted, tries data/gdpc1_releases_first_second_third.csv then data/panels/..."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["unprocessed", "processed"],
        default="unprocessed",
        help="Realtime mode. Controls default panel/model/output paths.",
    )
    parser.add_argument(
        "--vintage_panel",
        type=str,
        default="",
        help=(
            "Path to FRED-QD vintage panel parquet. "
            "Defaults by --mode when omitted."
        ),
    )
    parser.add_argument(
        "--md_vintage_panel",
        type=str,
        default="",
        help="Path to FRED-MD vintage panel parquet used by MD-feature realtime models. Defaults by --mode.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model names. Available: {sorted(set(BUILTIN_MODELS) | set(available_models()))}",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--score_releases",
        nargs="+",
        choices=["first", "second", "third"],
        default=None,
        help=(
            "Truth releases to score against. Defaults to first for quarterly origins and "
            "first+second+third for monthly origins."
        ),
    )
    parser.add_argument("--target_col", type=str, default="GDPC1")
    parser.add_argument("--baseline_model", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--origin_schedule", type=str, default="quarterly", choices=["quarterly", "monthly"])
    parser.add_argument(
        "--group_by_months_to_first_release_bucket",
        action="store_true",
        help="If set, metrics are reported by months-to-first-release bucket.",
    )
    parser.add_argument("--train_window", type=str, default="expanding", choices=["expanding", "rolling"])
    parser.add_argument("--rolling_size", type=int, default=None)
    parser.add_argument("--max_origins", type=int, default=None)
    parser.add_argument(
        "--min_target_quarter",
        type=str,
        default="2018Q1",
        help="Minimum target quarter included in scoring (default: 2018Q1).",
    )
    parser.add_argument("--min_train_observations", type=int, default=24)
    parser.add_argument(
        "--exclude_years",
        type=int,
        nargs="*",
        default=[],
        help="Optional calendar years to exclude inside mixed-frequency MD factor models (default: none).",
    )
    parser.add_argument("--min_first_release_lag_days", type=int, default=60)
    parser.add_argument("--max_first_release_lag_days", type=int, default=200)
    parser.add_argument(
        "--smoke_run",
        action="store_true",
        help="Fast smoke run: default 3 models and 10 origins unless overridden.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = normalize_mode(args.mode)

    requested_models = resolve_mode_models(mode=mode, requested_models=args.models)
    baseline_model = resolve_baseline_model(mode=mode, explicit_baseline=args.baseline_model or None)

    if args.smoke_run:
        if args.models is None:
            requested_models = requested_models[:3]
        if args.max_origins is None:
            args.max_origins = 10
        if args.origin_schedule == "monthly" and args.horizons == [1, 2, 3, 4]:
            args.horizons = [1, 2, 3, 4]

    output_dir = resolve_oos_output_dir(mode=mode, explicit_dir=args.output_dir or None)
    output_dir.mkdir(parents=True, exist_ok=True)

    release_path = Path(args.release_csv).resolve() if args.release_csv else None
    panel_path = resolve_qd_panel_path(mode=mode, explicit_path=args.vintage_panel or None)
    md_panel_path = resolve_md_panel_path(mode=mode, explicit_path=args.md_vintage_panel or None)

    release_table = load_release_table(
        path=release_path,
        min_first_release_lag_days=args.min_first_release_lag_days,
        max_first_release_lag_days=args.max_first_release_lag_days,
    )
    vintage_panel = load_vintage_panel(path=panel_path, target_col=args.target_col)

    valid_origins = int(release_table["valid_origin"].sum())
    print(f"Release table rows: {len(release_table)} | valid origins: {valid_origins}")
    print(
        "Vintage panel rows: "
        f"{len(vintage_panel)} | vintages: {vintage_panel['vintage'].nunique()} "
        f"({vintage_panel['vintage'].min()}..{vintage_panel['vintage'].max()})"
    )
    print(
        "Run config: "
        f"mode={mode}, models={requested_models}, baseline_model={baseline_model}, horizons={args.horizons}, "
        f"origin_schedule={args.origin_schedule}, max_origins={args.max_origins}, "
        f"score_releases={args.score_releases}, min_target_quarter={args.min_target_quarter}, "
        f"md_vintage_panel={md_panel_path}, exclude_years={sorted({int(y) for y in args.exclude_years})}"
    )

    predictions = run_backtest(
        models=requested_models,
        release_table=release_table,
        vintage_panel=vintage_panel,
        horizons=args.horizons,
        target_col=args.target_col,
        train_window=args.train_window,
        rolling_size=args.rolling_size,
        max_origins=args.max_origins,
        min_train_observations=args.min_train_observations,
        origin_schedule=args.origin_schedule,
        release_stages=args.score_releases,
        min_target_quarter=args.min_target_quarter,
        md_panel_path=md_panel_path,
        mixed_freq_excluded_years=args.exclude_years,
    )

    group_by_bucket = args.group_by_months_to_first_release_bucket or args.origin_schedule == "monthly"
    metrics = compute_metrics(
        predictions=predictions,
        baseline_model=baseline_model,
        group_by_bucket=group_by_bucket,
    )

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(f"Wrote predictions: {predictions_path} ({len(predictions)} rows)")
    print(f"Wrote metrics: {metrics_path} ({len(metrics)} rows)")

    if not predictions.empty:
        by_model = predictions.groupby("model").size().sort_values(ascending=False)
        print("Prediction rows by model:")
        for model_name, n_rows in by_model.items():
            print(f"  {model_name}: {int(n_rows)}")

    if not metrics.empty:
        print("Top metrics rows (sorted by horizon / release / bucket / RMSE):")
        sort_cols = [c for c in ["horizon", "release_stage", "months_to_first_release_bucket", "rmse", "model"] if c in metrics.columns]
        top = metrics.sort_values(sort_cols).head(20)
        print(top.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
