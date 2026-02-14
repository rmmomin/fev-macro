#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    load_fev_dataset,
)
from fev_macro.models import build_models  # noqa: E402
from fev_macro.models.base import get_task_horizon, get_task_target_column, get_task_timestamp_column  # noqa: E402
from fev_macro.tasks import make_gdp_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train history and OOS forecasts for top-K leaderboard models on GDP SAAR growth."
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--leaderboard", type=str, default="results/leaderboard.csv")
    parser.add_argument("--dataset_path", type=str, default="autogluon/fev_datasets")
    parser.add_argument("--dataset_config", type=str, default="fred_qd_2025")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--target", type=str, default="GDP_SAAR")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--num_windows", type=int, default=80)
    parser.add_argument("--metric", type=str, default="RMSE")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="results/gdp_saar_oos_top5.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default="results/gdp_saar_oos_top5.csv",
        help="Output OOS predictions csv path",
    )
    parser.add_argument(
        "--source_series_candidates",
        nargs="+",
        default=list(DEFAULT_SOURCE_SERIES_CANDIDATES),
    )
    return parser.parse_args()


def load_top_models(leaderboard_path: Path, top_k: int) -> list[str]:
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    leaderboard = pd.read_csv(leaderboard_path)
    if "model_name" not in leaderboard.columns:
        raise ValueError(f"Leaderboard must include model_name column. Found columns={list(leaderboard.columns)}")

    sort_cols = [c for c in ["win_rate", "skill_score"] if c in leaderboard.columns]
    if sort_cols:
        leaderboard = leaderboard.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    top_models = leaderboard["model_name"].head(top_k).tolist()
    if not top_models:
        raise ValueError("No models found in leaderboard")
    return top_models


def collect_oos_predictions(task, model_names: list[str], seed: int) -> pd.DataFrame:
    models = build_models(model_names=model_names, seed=seed)
    horizon = get_task_horizon(task)
    ts_col = get_task_timestamp_column(task)
    target_col = get_task_target_column(task)

    # timestamp -> {actuals: [...], model_name: [preds]}
    data: dict[pd.Timestamp, dict[str, list[float]]] = {}

    for window in task.iter_windows():
        input_data = window.get_input_data()
        if isinstance(input_data, tuple) and len(input_data) == 2:
            past_data, future_data = input_data
        else:
            past_data = input_data
            future_data = window.get_future_data()

        truth = window.get_ground_truth()
        ts_values = pd.to_datetime(pd.Series(truth[0][ts_col]), errors="coerce")
        actual_values = pd.to_numeric(pd.Series(truth[0][target_col]), errors="coerce")

        model_preds: dict[str, np.ndarray] = {}
        for model_name, model in models.items():
            pred_ds = model.predict(past_data=past_data, future_data=future_data, task=task)
            arr = np.asarray(pred_ds["predictions"][0], dtype=float).reshape(-1)
            if arr.size != horizon:
                raise ValueError(f"{model_name} produced horizon={arr.size}, expected={horizon}")
            if np.isnan(arr).any() or not np.isfinite(arr).all():
                raise ValueError(f"{model_name} produced NaN/inf forecasts")
            model_preds[model_name] = arr

        for i in range(horizon):
            ts = ts_values.iloc[i]
            actual = float(actual_values.iloc[i])
            if pd.isna(ts) or not np.isfinite(actual):
                continue

            bucket = data.setdefault(pd.Timestamp(ts), {"actual": []})
            bucket["actual"].append(actual)
            for model_name in model_names:
                bucket.setdefault(model_name, []).append(float(model_preds[model_name][i]))

    rows: list[dict[str, float]] = []
    for ts in sorted(data.keys()):
        row: dict[str, float] = {"timestamp": ts, "actual": float(np.mean(data[ts]["actual"]))}
        for model_name in model_names:
            preds = data[ts].get(model_name, [])
            row[model_name] = float(np.mean(preds)) if preds else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def make_plot(
    history_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    model_names: list[str],
    output_path: Path,
    horizon: int,
    num_windows: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    first_test_ts = pd.to_datetime(oos_df["timestamp"]).min()

    ax.plot(
        history_df["timestamp"],
        history_df["target"],
        color="black",
        linewidth=2.2,
        label="Actual (train)",
    )

    test_actual = oos_df[["timestamp", "actual"]].dropna().sort_values("timestamp")
    ax.plot(
        test_actual["timestamp"],
        test_actual["actual"],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Actual (test)",
    )

    for model_name in model_names:
        if model_name not in oos_df.columns:
            continue
        ax.plot(
            oos_df["timestamp"],
            oos_df[model_name],
            linewidth=1.8,
            alpha=0.9,
            label=f"{model_name} (OOS)",
        )

    ax.axvline(first_test_ts, color="gray", linestyle=":", linewidth=1.6, label="Test start")

    ax.set_title(
        f"US Real GDP Growth (q/q SAAR): Train History + OOS Forecasts (Top {len(model_names)} Models, h={horizon}, windows={num_windows})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth (percent, SAAR)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    leaderboard_path = Path(args.leaderboard)
    output_path = Path(args.output)
    predictions_csv = Path(args.predictions_csv)

    top_models = load_top_models(leaderboard_path=leaderboard_path, top_k=args.top_k)

    dataset = load_fev_dataset(
        config=args.dataset_config,
        dataset_path=args.dataset_path,
        dataset_revision=args.dataset_revision,
    )

    gdp_df, gdp_meta = build_gdp_saar_growth_series(
        dataset=dataset,
        target_series_name=args.target,
        source_series_candidates=args.source_series_candidates,
        include_covariates=True,
    )
    covariate_columns = list(gdp_meta.get("covariate_columns", []))

    local_dataset_path = results_dir / f"{args.target.lower()}_dataset.parquet"
    export_local_dataset_parquet(gdp_df, output_path=local_dataset_path)

    task = make_gdp_tasks(
        dataset_path=str(local_dataset_path),
        dataset_config=None,
        horizons=[args.horizon],
        num_windows=args.num_windows,
        metric=args.metric,
        id_col="id",
        timestamp_col="timestamp",
        target_col="target",
        known_dynamic_columns=covariate_columns,
    )[0]

    oos_df = collect_oos_predictions(task=task, model_names=top_models, seed=args.seed)
    if oos_df.empty:
        raise ValueError("No OOS predictions were generated")

    oos_df = oos_df.sort_values("timestamp").reset_index(drop=True)
    first_test_ts = pd.to_datetime(oos_df["timestamp"]).min()

    full_actual = gdp_df[["timestamp", "target"]].copy()
    full_actual["timestamp"] = pd.to_datetime(full_actual["timestamp"], errors="coerce")
    full_actual = full_actual.dropna(subset=["timestamp", "target"]).sort_values("timestamp")

    history_df = full_actual.loc[full_actual["timestamp"] < first_test_ts].copy()

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    oos_df.to_csv(predictions_csv, index=False)

    make_plot(
        history_df=history_df,
        oos_df=oos_df,
        model_names=top_models,
        output_path=output_path,
        horizon=args.horizon,
        num_windows=args.num_windows,
    )

    print(f"Top models: {top_models}")
    print(f"Target used: {gdp_meta['target_series']} (computed={gdp_meta['computed']}, source={gdp_meta['source_series']})")
    print(f"Covariates used: {len(covariate_columns)}")
    print(f"Saved OOS predictions: {predictions_csv}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
