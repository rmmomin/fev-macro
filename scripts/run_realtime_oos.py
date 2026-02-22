#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


def _infer_skill_column(leaderboard: pd.DataFrame) -> str:
    if "skill_score" in leaderboard.columns:
        return "skill_score"
    if "skill_vs_baseline" in leaderboard.columns:
        return "skill_vs_baseline"
    raise ValueError("Leaderboard is missing both 'skill_score' and 'skill_vs_baseline' columns.")


def _is_ensemble_row(model_name: str) -> bool:
    return "ensemble" in str(model_name).strip().lower()


def _map_eval_model_to_realtime_growth(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key.endswith("_growth") or key.startswith("bvar_minnesota_growth_"):
        return key

    mapping = {
        "naive_last": "naive_last_growth",
        "mean": "mean_growth",
        "ar4": "ar4_growth",
        "auto_arima": "auto_arima_growth",
        "auto_ets": "auto_ets_growth",
        "theta": "theta_growth",
        "local_trend_ssm": "local_trend_ssm_growth",
        "random_forest": "random_forest_growth",
        "xgboost": "xgboost_growth",
        "factor_pca_qd": "factor_pca_qd_growth",
        "mixed_freq_dfm_md": "mixed_freq_dfm_md_growth",
        "bvar_minnesota_8": "bvar_minnesota_growth_8",
        "bvar_minnesota_20": "bvar_minnesota_growth_20",
        "chronos2": "chronos2_growth",
        # keep log-drift as the realtime baseline when drift appears in eval outputs
        "drift": "rw_drift_log",
    }
    return mapping.get(key, key)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _select_models_from_leaderboard(
    *,
    leaderboard_path: Path,
    skill_threshold: float,
    exclude_chronos2: bool,
) -> dict[str, Any]:
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    leaderboard = pd.read_csv(leaderboard_path)
    if "model_name" not in leaderboard.columns:
        raise ValueError(f"Leaderboard missing model_name column: {leaderboard_path}")

    skill_col = _infer_skill_column(leaderboard)
    work = leaderboard.copy()
    work[skill_col] = pd.to_numeric(work[skill_col], errors="coerce")
    work["model_name"] = work["model_name"].astype(str)

    work = work.loc[work[skill_col].notna()]
    work = work.loc[work[skill_col] > float(skill_threshold)]
    work = work.loc[~work["model_name"].str.lower().eq("naive_last")]
    work = work.loc[~work["model_name"].map(_is_ensemble_row)]
    if exclude_chronos2:
        work = work.loc[~work["model_name"].str.lower().eq("chronos2")]
    work = work.sort_values(skill_col, ascending=False).reset_index(drop=True)

    eligible_eval = work["model_name"].tolist()
    top3_eval = eligible_eval[:3]
    top5_eval = eligible_eval[:5]

    available = set(BUILTIN_MODELS) | set(available_models())

    def _map_and_filter(names: list[str]) -> tuple[list[str], list[str]]:
        mapped = [_map_eval_model_to_realtime_growth(name) for name in names]
        deduped = _dedupe(mapped)
        keep = [name for name in deduped if name in available]
        dropped = [name for name in deduped if name not in available]
        return keep, dropped

    eligible_rt, dropped_rt = _map_and_filter(eligible_eval)
    top3_rt, dropped_top3 = _map_and_filter(top3_eval)
    top5_rt, dropped_top5 = _map_and_filter(top5_eval)

    return {
        "skill_column_used": skill_col,
        "threshold": float(skill_threshold),
        "eligible_models": eligible_eval,
        "top3_models": top3_eval,
        "top5_models": top5_eval,
        "eligible_models_realtime": eligible_rt,
        "top3_models_realtime": top3_rt,
        "top5_models_realtime": top5_rt,
        "dropped_realtime_models": _dedupe(dropped_rt + dropped_top3 + dropped_top5),
    }


def _default_selection_trace_path(skill_threshold: float) -> Path:
    threshold = f"{float(skill_threshold):g}".replace(".", "p").replace("-", "m")
    return Path("results/processed_standard") / f"selected_models_growth_skill_gt_{threshold}.json"


def _write_selection_trace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_leaderboard_ensembles(
    predictions: pd.DataFrame,
    *,
    top3_models_realtime: list[str],
    top5_models_realtime: list[str],
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    if "g_hat_saar" not in predictions.columns:
        return predictions

    group_cols = [
        c
        for c in [
            "origin_date",
            "origin_quarter",
            "target_quarter",
            "horizon",
            "release_stage",
            "origin_schedule",
            "origin_observed_quarter",
            "origin_vintage",
            "origin_vintage_asof_date",
            "training_min_quarter",
            "training_max_quarter",
            "covariate_cutoff_quarter",
            "months_to_first_release_bucket",
        ]
        if c in predictions.columns
    ]
    if not group_cols:
        return predictions

    ensemble_specs = [
        ("ensemble_leaderboard_top3", _dedupe(top3_models_realtime)),
        ("ensemble_leaderboard_top5", _dedupe(top5_models_realtime)),
    ]

    out = predictions.copy()
    available_models_in_preds = set(out["model"].astype(str).unique())
    extras: list[pd.DataFrame] = []

    for ensemble_name, members in ensemble_specs:
        use_members = [m for m in members if m in available_models_in_preds]
        if not use_members:
            continue

        subset = out.loc[out["model"].isin(use_members)].copy()
        if subset.empty:
            continue

        ens = (
            subset.groupby(group_cols, dropna=False, sort=False)["g_hat_saar"]
            .mean()
            .reset_index()
        )
        ens["model"] = ensemble_name

        passthrough_cols = [
            c
            for c in subset.columns
            if c not in set(group_cols + ["model", "g_hat_saar", "y_hat_level", "y_hat_prev_level", "log_y_hat"])
        ]
        if passthrough_cols:
            passthrough = (
                subset.groupby(group_cols, dropna=False, sort=False)[passthrough_cols]
                .first()
                .reset_index()
            )
            ens = ens.merge(passthrough, on=group_cols, how="left")

        # Growth-space ensembles are authoritative for realtime scoring.
        # Level/log outputs are reconstructed only when previous level is available.
        if "y_hat_prev_level" in subset.columns:
            prev_levels = (
                subset.groupby(group_cols, dropna=False, sort=False)["y_hat_prev_level"]
                .mean()
                .reset_index()
            )
            ens = ens.merge(prev_levels, on=group_cols, how="left")
            prev = pd.to_numeric(ens["y_hat_prev_level"], errors="coerce")
        else:
            ens["y_hat_prev_level"] = np.nan
            prev = pd.Series(np.nan, index=ens.index, dtype=float)

        growth = pd.to_numeric(ens["g_hat_saar"], errors="coerce")
        quarter_factor = 1.0 + growth / 100.0
        ens["y_hat_level"] = np.where(
            prev.notna() & np.isfinite(prev) & (prev > 0.0) & quarter_factor.notna() & np.isfinite(quarter_factor) & (quarter_factor > 0.0),
            prev * np.power(quarter_factor, 0.25),
            np.nan,
        )

        if "y_hat_level" in ens.columns:
            y_hat = pd.to_numeric(ens["y_hat_level"], errors="coerce")
            ens["log_y_hat"] = np.where(y_hat > 0, np.log(y_hat), np.nan)
        elif "log_y_hat" not in ens.columns:
            ens["log_y_hat"] = np.nan

        extras.append(ens)

    if not extras:
        return out
    return pd.concat([out, *extras], axis=0, ignore_index=True)


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
    parser.add_argument("--horizons", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--score_releases",
        nargs="+",
        choices=["first", "second", "third"],
        default=None,
        help=(
            "Truth releases to score against. Defaults to first vintage only."
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
    parser.add_argument(
        "--select_models_from_leaderboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use processed growth leaderboard skill thresholding to choose realtime models. "
            "When enabled, --models is ignored."
        ),
    )
    parser.add_argument(
        "--leaderboard_csv",
        type=str,
        default="results/processed_standard/leaderboard.csv",
        help="Processed growth leaderboard used for skill-threshold model selection.",
    )
    parser.add_argument(
        "--skill_threshold",
        type=float,
        default=0.18,
        help="Leaderboard skill threshold used when --select_models_from_leaderboard is enabled.",
    )
    parser.add_argument(
        "--exclude_chronos2_from_selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude chronos2 from leaderboard-selected model sets.",
    )
    parser.add_argument(
        "--selection_trace_json",
        type=str,
        default="",
        help=(
            "Optional path to write selected model metadata JSON. "
            "Defaults to results/processed_standard/selected_models_growth_skill_gt_<threshold>.json."
        ),
    )
    parser.add_argument(
        "--add_leaderboard_ensembles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append post-hoc ensemble rows for leaderboard top-3 and top-5 members.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = normalize_mode(args.mode)

    requested_models = resolve_mode_models(mode=mode, requested_models=args.models)
    baseline_model = resolve_baseline_model(mode=mode, explicit_baseline=args.baseline_model or None)
    selection_info: dict[str, Any] | None = None

    if args.smoke_run:
        if args.models is None:
            requested_models = requested_models[:3]
        if args.max_origins is None:
            args.max_origins = 10

    output_dir = resolve_oos_output_dir(mode=mode, explicit_dir=args.output_dir or None)
    output_dir.mkdir(parents=True, exist_ok=True)

    release_path = Path(args.release_csv).resolve() if args.release_csv else None
    panel_path = resolve_qd_panel_path(mode=mode, explicit_path=args.vintage_panel or None)
    md_panel_path = resolve_md_panel_path(mode=mode, explicit_path=args.md_vintage_panel or None)

    if args.select_models_from_leaderboard:
        leaderboard_path = Path(args.leaderboard_csv).expanduser().resolve()
        selection_info = _select_models_from_leaderboard(
            leaderboard_path=leaderboard_path,
            skill_threshold=float(args.skill_threshold),
            exclude_chronos2=bool(args.exclude_chronos2_from_selection),
        )
        requested_models = list(selection_info["eligible_models_realtime"])

        trace_path = (
            Path(args.selection_trace_json).expanduser().resolve()
            if str(args.selection_trace_json).strip()
            else _default_selection_trace_path(skill_threshold=float(args.skill_threshold)).resolve()
        )
        _write_selection_trace(trace_path, selection_info)
        print(f"Wrote selection trace: {trace_path}")
        if selection_info["dropped_realtime_models"]:
            print(f"Dropped unavailable mapped models: {selection_info['dropped_realtime_models']}")
        if not requested_models:
            raise ValueError(
                "Leaderboard selection produced no runnable realtime models. "
                "Lower --skill_threshold or inspect --leaderboard_csv."
            )

    if baseline_model not in requested_models:
        requested_models = _dedupe([baseline_model, *requested_models])

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
        f"md_vintage_panel={md_panel_path}, qd_vintage_panel={panel_path}, "
        f"exclude_years={sorted({int(y) for y in args.exclude_years})}, "
        f"select_models_from_leaderboard={bool(args.select_models_from_leaderboard)}"
    )
    if selection_info is not None:
        print(
            "Leaderboard selection: "
            f"skill_column={selection_info['skill_column_used']}, threshold={selection_info['threshold']}, "
            f"eligible_eval={selection_info['eligible_models']}, "
            f"eligible_realtime={selection_info['eligible_models_realtime']}, "
            f"top3_realtime={selection_info['top3_models_realtime']}, "
            f"top5_realtime={selection_info['top5_models_realtime']}"
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
        qd_panel_path=panel_path,
        mixed_freq_excluded_years=args.exclude_years,
    )
    if selection_info is not None and args.add_leaderboard_ensembles:
        predictions = _append_leaderboard_ensembles(
            predictions=predictions,
            top3_models_realtime=list(selection_info["top3_models_realtime"]),
            top5_models_realtime=list(selection_info["top5_models_realtime"]),
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
