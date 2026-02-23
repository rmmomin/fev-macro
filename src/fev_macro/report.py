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

    leaderboard_df = _augment_leaderboard_with_legacy_fields(
        leaderboard_df=leaderboard_df,
        summaries_df=summaries_df,
        metric_col=metric_col,
        baseline_model=baseline_model,
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


def _augment_leaderboard_with_legacy_fields(
    leaderboard_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    metric_col: str,
    baseline_model: str,
) -> pd.DataFrame:
    out = leaderboard_df.copy()
    if out.empty or "model_name" not in out.columns:
        return out

    summaries_work = summaries_df.copy()
    if "model_name" in summaries_work.columns:
        summaries_work["model_name"] = summaries_work["model_name"].astype(str)
    out["model_name"] = out["model_name"].astype(str)

    runtime_cols: list[tuple[str, str]] = []
    if "training_time_s" in summaries_work.columns:
        summaries_work["training_time_s"] = pd.to_numeric(summaries_work["training_time_s"], errors="coerce")
        runtime_cols.append(("median_training_time_s", "training_time_s"))
    if "inference_time_s" in summaries_work.columns:
        summaries_work["inference_time_s"] = pd.to_numeric(summaries_work["inference_time_s"], errors="coerce")
        runtime_cols.append(("median_inference_time_s", "inference_time_s"))

    computed = pd.DataFrame(index=pd.Index([], name="model_name"))
    if runtime_cols:
        agg_map = {alias: (src, "median") for alias, src in runtime_cols}
        runtime_df = summaries_work.groupby("model_name", as_index=False, dropna=False).agg(**agg_map).set_index("model_name")
        computed = runtime_df if computed.empty else computed.join(runtime_df, how="outer")

    if "trained_on_this_dataset" in summaries_work.columns:
        overlap = (
            pd.to_numeric(summaries_work["trained_on_this_dataset"], errors="coerce")
            .groupby(summaries_work["model_name"])
            .mean()
            .rename("training_corpus_overlap")
            .to_frame()
        )
        computed = overlap if computed.empty else computed.join(overlap, how="outer")

    if metric_col in summaries_work.columns:
        metric_vals = pd.to_numeric(summaries_work[metric_col], errors="coerce")
        failures = (
            metric_vals.isna()
            .groupby(summaries_work["model_name"])
            .sum()
            .astype(int)
            .rename("num_failures")
            .to_frame()
        )
        computed = failures if computed.empty else computed.join(failures, how="outer")

    if not computed.empty:
        out = out.set_index("model_name", drop=False)
        for col in computed.columns:
            computed_col = pd.to_numeric(computed[col], errors="coerce")
            if col in out.columns:
                out_col = pd.to_numeric(out[col], errors="coerce")
                out[col] = out_col.combine_first(computed_col)
            else:
                out[col] = computed_col
        out = out.reset_index(drop=True)

    if "win_rate" not in out.columns:
        win_rate = _compute_win_rate_from_summaries(summaries_df=summaries_work, metric_col=metric_col)
        out = out.merge(win_rate, on="model_name", how="left")

    if "skill_score" not in out.columns:
        if "skill_vs_baseline" in out.columns:
            out["skill_score"] = pd.to_numeric(out["skill_vs_baseline"], errors="coerce")
        elif metric_col in out.columns:
            metric_series = pd.to_numeric(out[metric_col], errors="coerce")
            baseline_rows = out.loc[out["model_name"] == str(baseline_model)].copy()
            baseline_value = np.nan
            if not baseline_rows.empty and metric_col in baseline_rows.columns:
                baseline_metric = pd.to_numeric(baseline_rows.iloc[0][metric_col], errors="coerce")
                if np.isfinite(baseline_metric) and baseline_metric != 0:
                    baseline_value = float(baseline_metric)
            if np.isfinite(baseline_value):
                out["skill_score"] = 1.0 - (metric_series / baseline_value)
            else:
                out["skill_score"] = np.nan

    if "median_training_time_s" not in out.columns:
        out["median_training_time_s"] = np.nan
    if "median_inference_time_s" not in out.columns:
        out["median_inference_time_s"] = np.nan
    if "training_corpus_overlap" not in out.columns:
        out["training_corpus_overlap"] = 0.0
    if "num_failures" not in out.columns:
        out["num_failures"] = 0

    out["num_failures"] = pd.to_numeric(out["num_failures"], errors="coerce").fillna(0).astype(int)
    out["training_corpus_overlap"] = pd.to_numeric(out["training_corpus_overlap"], errors="coerce").fillna(0.0)

    legacy_cols = [
        "model_name",
        "win_rate",
        "skill_score",
        "median_training_time_s",
        "median_inference_time_s",
        "training_corpus_overlap",
        "num_failures",
    ]
    remaining_cols = [c for c in out.columns if c not in legacy_cols]
    ordered = out[legacy_cols + remaining_cols].copy()

    if "win_rate" in ordered.columns:
        ordered["win_rate"] = pd.to_numeric(ordered["win_rate"], errors="coerce")
    if "skill_score" in ordered.columns:
        ordered["skill_score"] = pd.to_numeric(ordered["skill_score"], errors="coerce")
    if "win_rate" in ordered.columns:
        ordered = ordered.sort_values(
            by=["win_rate", "skill_score", "model_name"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    return ordered


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
                model_vals = common.iloc[:, 0].to_numpy(dtype=float)
                baseline_vals = common.iloc[:, -1].to_numpy(dtype=float)
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


def _compute_win_rate_from_summaries(summaries_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    try:
        pivot = _pivot_metric(summaries_df=summaries_df, metric_col=metric_col)
    except Exception:
        return pd.DataFrame(columns=["model_name", "win_rate"])

    if pivot.empty:
        return pd.DataFrame(columns=["model_name", "win_rate"])

    task_mins = pivot.min(axis=1)
    rows: list[dict[str, Any]] = []
    for model_name in pivot.columns:
        series = pivot[model_name].dropna()
        if series.empty:
            continue
        idx = series.index
        wins = (pivot.loc[idx, model_name] == task_mins.loc[idx]).astype(float)
        rows.append({"model_name": str(model_name), "win_rate": float(wins.mean())})
    return pd.DataFrame(rows, columns=["model_name", "win_rate"])


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
    model_values = np.asarray(model_values, dtype=float).reshape(-1)
    baseline_values = np.asarray(baseline_values, dtype=float).reshape(-1)

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
