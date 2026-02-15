from __future__ import annotations

import inspect
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_reports(
    summaries: pd.DataFrame | list[dict[str, Any]],
    results_dir: str | Path = "results",
    baseline_model: str = "naive_last",
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leaderboard + pairwise tables and save them to `results_dir`."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summaries_df = summaries if isinstance(summaries, pd.DataFrame) else pd.DataFrame(summaries)
    if summaries_df.empty:
        raise ValueError("No summaries available for reporting")

    metric_col = infer_metric_column(summaries_df)

    leaderboard_df: pd.DataFrame
    pairwise_df: pd.DataFrame

    try:
        import fev

        leaderboard_df = _call_fev_utility(
            fn=fev.leaderboard,
            summaries_df=summaries_df,
            baseline_model=baseline_model,
            metric_col=metric_col,
        )
        pairwise_df = _call_fev_utility(
            fn=fev.pairwise_comparison,
            summaries_df=summaries_df,
            baseline_model=baseline_model,
            metric_col=metric_col,
        )
    except Exception:
        leaderboard_df = _fallback_leaderboard(
            summaries_df=summaries_df,
            metric_col=metric_col,
            baseline_model=baseline_model,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        pairwise_df = _fallback_pairwise(
            summaries_df=summaries_df,
            metric_col=metric_col,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )

    leaderboard_path = results_dir / "leaderboard.csv"
    pairwise_path = results_dir / "pairwise.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)

    return leaderboard_df, pairwise_df


def infer_metric_column(summaries_df: pd.DataFrame) -> str:
    if "test_error" in summaries_df.columns:
        return "test_error"

    metric_cols = [c for c in summaries_df.columns if c.startswith("fev-")]
    if metric_cols:
        return metric_cols[0]

    if "eval_metric" in summaries_df.columns:
        eval_metric = str(summaries_df["eval_metric"].iloc[0])
        if eval_metric in summaries_df.columns:
            return eval_metric

    numeric_cols = summaries_df.select_dtypes(include=["number"]).columns.tolist()
    ignored = {
        "horizon",
        "num_windows",
        "initial_cutoff",
        "window_step_size",
        "min_context_length",
        "max_context_length",
        "seasonality",
        "training_time_s",
        "inference_time_s",
        "num_forecasts",
    }
    candidates = [c for c in numeric_cols if c not in ignored]
    if candidates:
        return candidates[0]

    raise ValueError("Could not infer metric column from summaries")


def _call_fev_utility(
    fn,
    summaries_df: pd.DataFrame,
    baseline_model: str,
    metric_col: str,
) -> pd.DataFrame:
    params = inspect.signature(fn).parameters
    kwargs: dict[str, Any] = {}

    if "baseline_model" in params:
        kwargs["baseline_model"] = baseline_model
    if "metric_column" in params:
        kwargs["metric_column"] = metric_col
    elif "metric_col" in params:
        kwargs["metric_col"] = metric_col

    result = fn(summaries_df, **kwargs)
    df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    return df


def _fallback_leaderboard(
    summaries_df: pd.DataFrame,
    metric_col: str,
    baseline_model: str,
    seed: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    pivot = _pivot_metric(summaries_df=summaries_df, metric_col=metric_col)
    rng = np.random.default_rng(seed)

    baseline_series = pivot[baseline_model].dropna() if baseline_model in pivot.columns else pd.Series(dtype=float)

    rows: list[dict[str, Any]] = []
    task_mins = pivot.min(axis=1)

    for model_name in pivot.columns:
        model_series = pivot[model_name].dropna()
        if model_series.empty:
            continue

        task_idx = model_series.index
        wins = (pivot.loc[task_idx, model_name] == task_mins.loc[task_idx]).astype(float)
        mean_metric = float(model_series.mean())
        win_rate = float(wins.mean()) if len(wins) else np.nan

        metric_lo, metric_hi = _bootstrap_ci(
            values=model_series.to_numpy(dtype=float),
            reducer=lambda x: float(np.mean(x)),
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )
        win_lo, win_hi = _bootstrap_ci(
            values=wins.to_numpy(dtype=float),
            reducer=lambda x: float(np.mean(x)),
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )

        skill = np.nan
        skill_lo = np.nan
        skill_hi = np.nan
        if not baseline_series.empty:
            common = pivot[[model_name, baseline_model]].dropna()
            if not common.empty:
                model_vals = common[model_name].to_numpy(dtype=float)
                baseline_vals = common[baseline_model].to_numpy(dtype=float)
                denom = baseline_vals.mean()
                skill = float(1.0 - (model_vals.mean() / denom)) if denom != 0 else np.nan
                skill_lo, skill_hi = _bootstrap_skill_ci(
                    model_values=model_vals,
                    baseline_values=baseline_vals,
                    rng=rng,
                    bootstrap_samples=bootstrap_samples,
                )

        rows.append(
            {
                "model_name": model_name,
                "n_tasks": int(model_series.shape[0]),
                metric_col: mean_metric,
                f"{metric_col}_ci_lo": metric_lo,
                f"{metric_col}_ci_hi": metric_hi,
                "win_rate": win_rate,
                "win_rate_ci_lo": win_lo,
                "win_rate_ci_hi": win_hi,
                "skill_vs_baseline": skill,
                "skill_ci_lo": skill_lo,
                "skill_ci_hi": skill_hi,
                "baseline_model": baseline_model,
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values(metric_col, ascending=True).reset_index(drop=True)
    return leaderboard


def _fallback_pairwise(
    summaries_df: pd.DataFrame,
    metric_col: str,
    seed: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    pivot = _pivot_metric(summaries_df=summaries_df, metric_col=metric_col)
    rng = np.random.default_rng(seed + 1)

    rows: list[dict[str, Any]] = []
    for model_a, model_b in permutations(pivot.columns.tolist(), 2):
        common = pivot[[model_a, model_b]].dropna()
        if common.empty:
            continue

        a_vals = common[model_a].to_numpy(dtype=float)
        b_vals = common[model_b].to_numpy(dtype=float)

        wins = (a_vals < b_vals).astype(float) + 0.5 * (a_vals == b_vals).astype(float)
        win_rate = float(np.mean(wins))
        mean_metric_diff = float(np.mean(b_vals - a_vals))

        win_lo, win_hi = _bootstrap_ci(
            values=wins,
            reducer=lambda x: float(np.mean(x)),
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )

        diff_lo, diff_hi = _bootstrap_ci(
            values=b_vals - a_vals,
            reducer=lambda x: float(np.mean(x)),
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_tasks": int(common.shape[0]),
                "win_rate_a_over_b": win_rate,
                "win_rate_ci_lo": win_lo,
                "win_rate_ci_hi": win_hi,
                "mean_metric_diff_b_minus_a": mean_metric_diff,
                "mean_metric_diff_ci_lo": diff_lo,
                "mean_metric_diff_ci_hi": diff_hi,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_a",
                "model_b",
                "n_tasks",
                "win_rate_a_over_b",
                "win_rate_ci_lo",
                "win_rate_ci_hi",
                "mean_metric_diff_b_minus_a",
                "mean_metric_diff_ci_lo",
                "mean_metric_diff_ci_hi",
            ]
        )

    pairwise = pd.DataFrame(rows).sort_values(["model_a", "win_rate_a_over_b"], ascending=[True, False])
    return pairwise.reset_index(drop=True)


def _pivot_metric(summaries_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    required = {"task_name", "model_name", metric_col}
    missing = [c for c in required if c not in summaries_df.columns]
    if missing:
        raise ValueError(f"Summaries missing required columns for reporting: {missing}")

    pivot = summaries_df.pivot_table(
        index="task_name",
        columns="model_name",
        values=metric_col,
        aggfunc="mean",
    )
    return pivot


def _bootstrap_ci(
    values: np.ndarray,
    reducer,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan

    stats = np.empty(bootstrap_samples, dtype=float)
    n = arr.size
    for i in range(bootstrap_samples):
        sample = arr[rng.integers(0, n, size=n)]
        stats[i] = reducer(sample)

    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_skill_ci(
    model_values: np.ndarray,
    baseline_values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[float, float]:
    model_values = np.asarray(model_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)

    if model_values.size == 0 or baseline_values.size == 0:
        return np.nan, np.nan

    n = min(model_values.size, baseline_values.size)
    model_values = model_values[:n]
    baseline_values = baseline_values[:n]
    stats = np.empty(bootstrap_samples, dtype=float)

    for i in range(bootstrap_samples):
        idx = rng.integers(0, n, size=n)
        m = model_values[idx].mean()
        b = baseline_values[idx].mean()
        stats[i] = np.nan if b == 0 else (1.0 - m / b)

    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return np.nan, np.nan

    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
