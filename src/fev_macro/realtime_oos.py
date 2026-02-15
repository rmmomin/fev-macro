from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BMonthEnd

DEFAULT_MIN_FIRST_RELEASE_LAG_DAYS = 60
DEFAULT_MAX_FIRST_RELEASE_LAG_DAYS = 200
DEFAULT_TARGET_COL = "GDPC1"
DEFAULT_HORIZONS = (1, 2, 3, 4)
RELEASE_STAGE_TO_COL = {
    "first": "first_release",
    "second": "second_release",
    "third": "third_release",
}


def to_quarter_period(values: pd.Series | Sequence[object]) -> pd.PeriodIndex:
    ts = pd.to_datetime(values, errors="coerce")
    return pd.PeriodIndex(ts, freq="Q-DEC")


def compute_vintage_asof_date(vintage_label: str | pd.Period | pd.Timestamp) -> pd.Timestamp:
    if isinstance(vintage_label, pd.Period):
        month_start = vintage_label.to_timestamp(how="start")
    elif isinstance(vintage_label, pd.Timestamp):
        month_start = pd.Timestamp(vintage_label.year, vintage_label.month, 1)
    else:
        period = pd.Period(str(vintage_label), freq="M")
        month_start = period.to_timestamp(how="start")

    return (month_start + BMonthEnd(1)).normalize()


def to_saar_growth(current_level: float, previous_level: float) -> float:
    if not np.isfinite(current_level) or not np.isfinite(previous_level):
        return float("nan")
    if current_level <= 0 or previous_level <= 0:
        return float("nan")
    return float(100.0 * ((current_level / previous_level) ** 4 - 1.0))


def months_to_first_release(origin_date: pd.Timestamp, first_release_date: pd.Timestamp | float) -> float:
    if pd.isna(first_release_date):
        return float("nan")
    o = pd.Timestamp(origin_date)
    f = pd.Timestamp(first_release_date)
    return float((f.to_period("M") - o.to_period("M")).n)


def months_to_first_release_bucket(months: float) -> str:
    if not np.isfinite(months):
        return "unknown"
    m = int(months)
    if m <= 0:
        return "<=0"
    if m == 1:
        return "1"
    if m <= 3:
        return "2-3"
    if m <= 6:
        return "4-6"
    return "7+"


def _normalize_release_stages(release_stages: Sequence[str] | None, origin_schedule: str) -> list[str]:
    if release_stages is None:
        stages = ["first", "second", "third"] if origin_schedule == "monthly" else ["first"]
    else:
        stages = [str(s).strip().lower() for s in release_stages]

    if not stages:
        raise ValueError("release_stages cannot be empty.")

    invalid = [s for s in stages if s not in RELEASE_STAGE_TO_COL]
    if invalid:
        raise ValueError(f"Invalid release stages: {invalid}. Valid: {sorted(RELEASE_STAGE_TO_COL)}")

    return list(dict.fromkeys(stages))


def _default_release_path() -> Path:
    candidates = [
        Path("data/gdpc1_releases_first_second_third.csv"),
        Path("data/panels/gdpc1_releases_first_second_third.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_vintage_panel_path() -> Path:
    candidates = [
        Path("data/fred_qd_vintage_panel.parquet"),
        Path("data/panels/fred_qd_vintage_panel.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_release_table(
    path: str | Path | None = None,
    min_first_release_lag_days: int = DEFAULT_MIN_FIRST_RELEASE_LAG_DAYS,
    max_first_release_lag_days: int = DEFAULT_MAX_FIRST_RELEASE_LAG_DAYS,
) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else _default_release_path()
    df = pd.read_csv(csv_path)

    required = {"observation_date", "first_release_date", "first_release"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Release table missing required columns: {missing}")

    df = df.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["first_release_date"] = pd.to_datetime(df["first_release_date"], errors="coerce")
    df["first_release"] = pd.to_numeric(df["first_release"], errors="coerce")

    df["quarter"] = to_quarter_period(df["observation_date"])

    if "first_release_lag_days" not in df.columns:
        df["first_release_lag_days"] = (df["first_release_date"] - df["observation_date"]).dt.days
    else:
        df["first_release_lag_days"] = pd.to_numeric(df["first_release_lag_days"], errors="coerce")

    df = df.sort_values("quarter").reset_index(drop=True)

    y = df["first_release"].astype(float)
    y_prev = y.shift(1)
    df["g_true_saar_first"] = [to_saar_growth(c, p) for c, p in zip(y, y_prev)]

    valid_origin = (
        df["first_release_date"].notna()
        & df["first_release"].notna()
        & df["first_release_lag_days"].between(min_first_release_lag_days, max_first_release_lag_days, inclusive="both")
    )
    df["valid_origin"] = valid_origin

    return df


def load_vintage_panel(
    path: str | Path | None = None,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    panel_path = Path(path) if path is not None else _default_vintage_panel_path()
    df = pd.read_parquet(panel_path)

    if "timestamp" not in df.columns:
        raise ValueError("Vintage panel must contain a 'timestamp' column.")

    if "vintage" not in df.columns:
        if "vintage_timestamp" not in df.columns:
            raise ValueError("Vintage panel must contain either 'vintage' or 'vintage_timestamp'.")
        df["vintage"] = pd.to_datetime(df["vintage_timestamp"], errors="coerce").dt.strftime("%Y-%m")

    if target_col not in df.columns:
        raise ValueError(f"Vintage panel missing target column: {target_col}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["quarter"] = to_quarter_period(out["timestamp"])
    out["vintage"] = out["vintage"].astype(str)

    unique_vintages = sorted(out["vintage"].dropna().unique())
    asof_map = {v: compute_vintage_asof_date(v) for v in unique_vintages}
    out["asof_date"] = out["vintage"].map(asof_map)

    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")

    out = out.dropna(subset=["quarter", "asof_date"]).sort_values(["vintage", "quarter"]).reset_index(drop=True)
    return out


def build_vintage_calendar(vintage_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"vintage", "asof_date"}
    missing = sorted(required.difference(vintage_panel.columns))
    if missing:
        raise ValueError(f"Vintage panel missing required columns for calendar: {missing}")

    calendar = vintage_panel[["vintage", "asof_date"]].drop_duplicates().sort_values("asof_date").reset_index(drop=True)
    calendar["next_asof_date"] = calendar["asof_date"].shift(-1)
    return calendar


def select_training_vintage(origin_date: pd.Timestamp, vintage_calendar: pd.DataFrame) -> dict[str, object]:
    if vintage_calendar.empty:
        raise ValueError("Vintage calendar is empty.")

    origin_ts = pd.Timestamp(origin_date).normalize()
    eligible = vintage_calendar.loc[vintage_calendar["asof_date"] <= origin_ts]
    if eligible.empty:
        earliest = vintage_calendar.iloc[0]
        raise ValueError(
            "No vintage available for origin date "
            f"{origin_ts.date()} (earliest as-of: {pd.Timestamp(earliest['asof_date']).date()})."
        )

    selected = eligible.iloc[-1]
    return {
        "vintage": str(selected["vintage"]),
        "asof_date": pd.Timestamp(selected["asof_date"]),
        "next_asof_date": pd.Timestamp(selected["next_asof_date"]) if pd.notna(selected["next_asof_date"]) else pd.NaT,
    }


def build_origin_datasets(
    origin_quarter: pd.Period,
    vintage_id: str,
    release_table: pd.DataFrame,
    vintage_panel: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    train_window: str = "expanding",
    rolling_size: int | None = None,
    cutoff_quarter: pd.Period | None = None,
) -> dict[str, object]:
    if train_window not in {"expanding", "rolling"}:
        raise ValueError("train_window must be one of {'expanding', 'rolling'}")

    if "quarter" not in vintage_panel.columns:
        raise ValueError("Vintage panel must include 'quarter'.")

    origin_period = pd.Period(origin_quarter, freq="Q-DEC")
    cutoff = pd.Period(cutoff_quarter, freq="Q-DEC") if cutoff_quarter is not None else origin_period - 1

    vintage_slice = vintage_panel.loc[vintage_panel["vintage"].astype(str) == str(vintage_id)].copy()
    if vintage_slice.empty:
        raise ValueError(f"No rows found for vintage '{vintage_id}'.")

    train_df = vintage_slice.loc[vintage_slice["quarter"] <= cutoff].copy()
    train_df = train_df.sort_values("quarter").reset_index(drop=True)

    if train_window == "rolling" and rolling_size is not None and rolling_size > 0 and len(train_df) > rolling_size:
        train_df = train_df.iloc[-rolling_size:].reset_index(drop=True)

    y_train = train_df[target_col].astype(float)
    y_train = y_train[y_train.notna()]
    if y_train.empty:
        raise ValueError(
            f"No non-missing {target_col} values in training data for vintage '{vintage_id}' and cutoff {cutoff}."
        )

    observed_quarters = train_df.loc[train_df[target_col].notna(), "quarter"]
    if observed_quarters.empty:
        raise ValueError("Training data has no observed target quarters.")

    training_max_quarter = observed_quarters.max()
    if training_max_quarter > cutoff:
        raise AssertionError(
            "Leakage detected: training includes data after cutoff quarter. "
            f"training_max={training_max_quarter}, cutoff={cutoff}"
        )

    return {
        "origin_quarter": origin_period,
        "cutoff_quarter": cutoff,
        "train_df": train_df,
        "training_min_quarter": observed_quarters.min(),
        "training_max_quarter": training_max_quarter,
        "last_observed_level": float(y_train.iloc[-1]),
        "n_train": int(y_train.shape[0]),
    }


class RealtimeModel(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        raise NotImplementedError


class NaiveLastModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="naive_last")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = pd.to_numeric(train_df[target_col], errors="coerce").dropna().to_numpy(dtype=float)
        if y.size == 0:
            raise ValueError("No target history for naive_last model.")
        return np.repeat(y[-1], horizon)


class RandomWalkDriftLogModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="rw_drift_log")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = pd.to_numeric(train_df[target_col], errors="coerce").dropna().to_numpy(dtype=float)
        if y.size == 0:
            raise ValueError("No target history for rw_drift_log model.")
        if np.any(y <= 0):
            raise ValueError("rw_drift_log requires strictly positive levels.")

        ly = np.log(y)
        if ly.size < 2:
            drift = 0.0
        else:
            drift = float((ly[-1] - ly[0]) / (ly.size - 1))

        steps = np.arange(1, horizon + 1, dtype=float)
        ly_hat = ly[-1] + drift * steps
        return np.exp(ly_hat)


class AR4GrowthModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="ar4_growth")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = pd.to_numeric(train_df[target_col], errors="coerce").dropna().to_numpy(dtype=float)
        if y.size == 0:
            raise ValueError("No target history for ar4_growth model.")
        if np.any(y <= 0):
            raise ValueError("ar4_growth requires strictly positive levels.")

        if y.size < 8:
            return np.repeat(y[-1], horizon)

        ly = np.log(y)
        g = np.diff(ly)
        p = 4
        if g.size <= p:
            return np.repeat(y[-1], horizon)

        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        for t in range(p, g.size):
            row = [1.0]
            row.extend(float(g[t - lag]) for lag in range(1, p + 1))
            x_rows.append(row)
            y_rows.append(float(g[t]))

        x = np.asarray(x_rows, dtype=float)
        y_target = np.asarray(y_rows, dtype=float)

        coef, *_ = np.linalg.lstsq(x, y_target, rcond=None)

        hist = list(g.astype(float))
        g_forecast: list[float] = []
        for _ in range(horizon):
            row = np.asarray([1.0] + [hist[-lag] for lag in range(1, p + 1)], dtype=float)
            pred = float(np.dot(row, coef))
            g_forecast.append(pred)
            hist.append(pred)

        cumulative = np.cumsum(np.asarray(g_forecast, dtype=float))
        return y[-1] * np.exp(cumulative)


BUILTIN_MODELS: dict[str, type[RealtimeModel]] = {
    "naive_last": NaiveLastModel,
    "rw_drift_log": RandomWalkDriftLogModel,
    "ar4_growth": AR4GrowthModel,
}


def resolve_models(models: Sequence[str | RealtimeModel]) -> list[RealtimeModel]:
    resolved: list[RealtimeModel] = []
    for m in models:
        if isinstance(m, RealtimeModel):
            resolved.append(m)
            continue

        key = str(m).strip().lower()
        if key not in BUILTIN_MODELS:
            raise ValueError(f"Unknown model '{m}'. Available: {sorted(BUILTIN_MODELS)}")
        resolved.append(BUILTIN_MODELS[key]())

    if not resolved:
        raise ValueError("No models were provided.")

    return resolved


def _build_quarterly_origins(
    release_table: pd.DataFrame,
    vintage_calendar: pd.DataFrame,
    max_origins: int | None = None,
) -> pd.DataFrame:
    origin_rows = release_table.loc[release_table["valid_origin"]].copy()
    origin_rows = origin_rows.sort_values("quarter").reset_index(drop=True)
    if max_origins is not None and max_origins > 0 and len(origin_rows) > max_origins:
        origin_rows = origin_rows.iloc[-max_origins:].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for origin in origin_rows.itertuples(index=False):
        q = pd.Period(origin.quarter, freq="Q-DEC")
        origin_date = pd.Timestamp(origin.first_release_date) - pd.Timedelta(days=1)
        try:
            selected = select_training_vintage(origin_date=origin_date, vintage_calendar=vintage_calendar)
        except ValueError:
            # Earliest release origins may predate available vintages; skip those origins.
            continue
        rows.append(
            {
                "origin_schedule": "quarterly",
                "origin_date": pd.Timestamp(origin_date),
                "origin_vintage": str(selected["vintage"]),
                "origin_vintage_asof_date": pd.Timestamp(selected["asof_date"]),
                "origin_quarter": q,
                "observed_quarter": q - 1,
            }
        )

    return pd.DataFrame(rows)


def _build_monthly_origins(
    vintage_panel: pd.DataFrame,
    vintage_calendar: pd.DataFrame,
    target_col: str,
    max_origins: int | None = None,
) -> pd.DataFrame:
    cal = vintage_calendar.sort_values("asof_date").reset_index(drop=True)
    if max_origins is not None and max_origins > 0 and len(cal) > max_origins:
        cal = cal.iloc[-max_origins:].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for row in cal.itertuples(index=False):
        vintage_id = str(row.vintage)
        origin_date = pd.Timestamp(row.asof_date)

        vint_df = vintage_panel.loc[vintage_panel["vintage"].astype(str) == vintage_id]
        observed_quarters = vint_df.loc[vint_df[target_col].notna(), "quarter"]
        if observed_quarters.empty:
            continue

        observed_q = pd.Period(observed_quarters.max(), freq="Q-DEC")
        rows.append(
            {
                "origin_schedule": "monthly",
                "origin_date": origin_date,
                "origin_vintage": vintage_id,
                "origin_vintage_asof_date": origin_date,
                "origin_quarter": observed_q,
                "observed_quarter": observed_q,
            }
        )

    return pd.DataFrame(rows)


def run_backtest(
    models: Sequence[str | RealtimeModel],
    release_table: pd.DataFrame,
    vintage_panel: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    target_col: str = DEFAULT_TARGET_COL,
    train_window: str = "expanding",
    rolling_size: int | None = None,
    max_origins: int | None = None,
    min_train_observations: int = 24,
    origin_schedule: str = "quarterly",
    release_stages: Sequence[str] | None = None,
    min_target_quarter: str | pd.Period | None = "2018Q1",
) -> pd.DataFrame:
    if origin_schedule not in {"quarterly", "monthly"}:
        raise ValueError("origin_schedule must be one of {'quarterly', 'monthly'}")

    horizon_list = sorted({int(h) for h in horizons if int(h) >= 1})
    if not horizon_list:
        raise ValueError("At least one horizon >= 1 is required.")

    stage_list = _normalize_release_stages(release_stages=release_stages, origin_schedule=origin_schedule)
    min_target_period = pd.Period(min_target_quarter, freq="Q-DEC") if min_target_quarter is not None else None
    model_list = resolve_models(models)

    if "valid_origin" not in release_table.columns:
        raise ValueError("Release table must include 'valid_origin'. Use load_release_table().")
    if "quarter" not in release_table.columns:
        raise ValueError("Release table must include 'quarter'.")

    releases = release_table.sort_values("quarter").reset_index(drop=True)
    first_release_date_map = releases.set_index("quarter")["first_release_date"].to_dict()
    truth_maps: dict[str, dict[pd.Period, float]] = {}
    for stage in stage_list:
        col = RELEASE_STAGE_TO_COL[stage]
        if col not in releases.columns:
            raise ValueError(f"Release table missing required column for stage '{stage}': {col}")
        truth_df = releases.loc[releases[col].notna(), ["quarter", col]].copy()
        truth_maps[stage] = {pd.Period(q, freq="Q-DEC"): float(v) for q, v in zip(truth_df["quarter"], truth_df[col])}

    vintage_calendar = build_vintage_calendar(vintage_panel)
    if origin_schedule == "quarterly":
        origins = _build_quarterly_origins(
            release_table=releases,
            vintage_calendar=vintage_calendar,
            max_origins=max_origins,
        )
    else:
        origins = _build_monthly_origins(
            vintage_panel=vintage_panel,
            vintage_calendar=vintage_calendar,
            target_col=target_col,
            max_origins=max_origins,
        )

    max_h = max(horizon_list)
    rows: list[dict[str, object]] = []

    for origin in origins.itertuples(index=False):
        origin_date = pd.Timestamp(origin.origin_date)
        origin_quarter = pd.Period(origin.origin_quarter, freq="Q-DEC")
        observed_quarter = pd.Period(origin.observed_quarter, freq="Q-DEC")
        vintage_id = str(origin.origin_vintage)

        try:
            ds = build_origin_datasets(
                origin_quarter=observed_quarter + 1,
                vintage_id=vintage_id,
                release_table=releases,
                vintage_panel=vintage_panel,
                target_col=target_col,
                train_window=train_window,
                rolling_size=rolling_size,
                cutoff_quarter=observed_quarter,
            )
        except ValueError:
            # Skip origins where selected vintage/cutoff does not provide a usable training sample.
            continue

        train_df = ds["train_df"]
        n_train = int(ds["n_train"])
        if n_train < int(min_train_observations):
            continue

        last_observed = float(ds["last_observed_level"])
        training_min_quarter = pd.Period(ds["training_min_quarter"], freq="Q-DEC")
        training_max_quarter = pd.Period(ds["training_max_quarter"], freq="Q-DEC")

        for model in model_list:
            path = np.asarray(model.forecast_levels(train_df=train_df, horizon=max_h, target_col=target_col), dtype=float)
            if path.size != max_h:
                raise ValueError(
                    f"Model '{model.name}' produced {path.size} forecasts, expected {max_h}."
                )
            if np.isnan(path).any() or np.isinf(path).any():
                raise ValueError(f"Model '{model.name}' produced non-finite forecasts.")

            for h in horizon_list:
                target_quarter = observed_quarter + h
                if min_target_period is not None and target_quarter < min_target_period:
                    continue
                if training_max_quarter >= target_quarter:
                    raise AssertionError(
                        "Leakage detected: training_max_quarter must be strictly before target_quarter. "
                        f"training_max={training_max_quarter}, target={target_quarter}"
                    )

                y_hat = float(path[h - 1])
                y_hat_prev = float(last_observed if h == 1 else path[h - 2])

                g_hat = to_saar_growth(y_hat, y_hat_prev)
                log_hat = float(np.log(y_hat)) if y_hat > 0 else float("nan")

                first_release_date_target = first_release_date_map.get(target_quarter, pd.NaT)
                m2fr = months_to_first_release(origin_date=origin_date, first_release_date=first_release_date_target)
                m2fr_bucket = months_to_first_release_bucket(m2fr)

                y_true_first = truth_maps.get("first", {}).get(target_quarter, np.nan)
                y_true_prev_first = truth_maps.get("first", {}).get(target_quarter - 1, np.nan)
                g_true_first = to_saar_growth(float(y_true_first), float(y_true_prev_first))

                for stage in stage_list:
                    stage_map = truth_maps[stage]
                    y_true = stage_map.get(target_quarter, np.nan)
                    y_true_prev = stage_map.get(target_quarter - 1, np.nan)
                    g_true = to_saar_growth(float(y_true), float(y_true_prev))
                    log_true = float(np.log(y_true)) if np.isfinite(y_true) and y_true > 0 else float("nan")

                    rows.append(
                        {
                            "model": model.name,
                            "origin_schedule": origin.origin_schedule,
                            "origin_date": origin_date.date().isoformat(),
                            "origin_quarter": str(origin_quarter),
                            "origin_observed_quarter": str(observed_quarter),
                            "origin_vintage": vintage_id,
                            "origin_vintage_asof_date": pd.Timestamp(origin.origin_vintage_asof_date).date().isoformat(),
                            "training_min_quarter": str(training_min_quarter),
                            "training_max_quarter": str(training_max_quarter),
                            "target_quarter": str(target_quarter),
                            "horizon": int(h),
                            "release_stage": stage,
                            "months_to_first_release": m2fr,
                            "months_to_first_release_bucket": m2fr_bucket,
                            "y_hat_level": y_hat,
                            "y_hat_prev_level": y_hat_prev,
                            "y_true_level": float(y_true) if np.isfinite(y_true) else np.nan,
                            "y_true_prev_level": float(y_true_prev) if np.isfinite(y_true_prev) else np.nan,
                            "y_true_level_first": float(y_true_first) if np.isfinite(y_true_first) else np.nan,
                            "y_true_prev_level_first": float(y_true_prev_first) if np.isfinite(y_true_prev_first) else np.nan,
                            "g_hat_saar": g_hat,
                            "g_true_saar": g_true,
                            "g_true_saar_first": g_true_first,
                            "log_y_hat": log_hat,
                            "log_y_true": log_true,
                        }
                    )

    return pd.DataFrame(rows)


def compute_metrics(
    predictions: pd.DataFrame,
    baseline_model: str = "rw_drift_log",
    group_by_bucket: bool = False,
) -> pd.DataFrame:
    output_cols = [
        "model",
        "horizon",
        "release_stage",
        "months_to_first_release_bucket",
        "rmse",
        "mae",
        "rmse_log_level",
        "mae_log_level",
        "rel_rmse",
        "sample_count",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=output_cols)

    group_cols = ["model", "horizon"]
    if "release_stage" in predictions.columns:
        group_cols.append("release_stage")
    if group_by_bucket and "months_to_first_release_bucket" in predictions.columns:
        group_cols.append("months_to_first_release_bucket")

    metric_rows: list[dict[str, object]] = []
    for key, grp in predictions.groupby(group_cols, sort=True):
        if len(group_cols) == 1:
            key = (key,)
        elif not isinstance(key, tuple):
            key = (key,)
        key_map = {col: key[i] for i, col in enumerate(group_cols)}

        valid = grp.dropna(subset=["g_hat_saar", "g_true_saar"])
        if valid.empty:
            rmse = np.nan
            mae = np.nan
            sample_count = 0
        else:
            err = valid["g_hat_saar"].to_numpy(dtype=float) - valid["g_true_saar"].to_numpy(dtype=float)
            if group_by_bucket and "target_quarter" in valid.columns:
                err_df = pd.DataFrame(
                    {
                        "target_quarter": valid["target_quarter"].astype(str).to_numpy(),
                        "sq_err": err**2,
                        "abs_err": np.abs(err),
                    }
                )
                by_target = err_df.groupby("target_quarter", sort=False).mean(numeric_only=True)
                rmse = float(np.sqrt(by_target["sq_err"].mean()))
                mae = float(by_target["abs_err"].mean())
                sample_count = int(by_target.shape[0])
            else:
                rmse = float(np.sqrt(np.mean(err**2)))
                mae = float(np.mean(np.abs(err)))
                sample_count = int(valid.shape[0])

        log_valid = grp.dropna(subset=["log_y_hat", "log_y_true"])
        if log_valid.empty:
            rmse_log = np.nan
            mae_log = np.nan
        else:
            log_err = log_valid["log_y_hat"].to_numpy(dtype=float) - log_valid["log_y_true"].to_numpy(dtype=float)
            if group_by_bucket and "target_quarter" in log_valid.columns:
                log_df = pd.DataFrame(
                    {
                        "target_quarter": log_valid["target_quarter"].astype(str).to_numpy(),
                        "sq_err": log_err**2,
                        "abs_err": np.abs(log_err),
                    }
                )
                by_target_log = log_df.groupby("target_quarter", sort=False).mean(numeric_only=True)
                rmse_log = float(np.sqrt(by_target_log["sq_err"].mean()))
                mae_log = float(by_target_log["abs_err"].mean())
            else:
                rmse_log = float(np.sqrt(np.mean(log_err**2)))
                mae_log = float(np.mean(np.abs(log_err)))

        metric_rows.append(
            {
                "model": str(key_map.get("model", "")),
                "horizon": int(key_map.get("horizon", -1)),
                "release_stage": str(key_map.get("release_stage", "first")),
                "months_to_first_release_bucket": str(key_map.get("months_to_first_release_bucket", "all")),
                "rmse": rmse,
                "mae": mae,
                "rmse_log_level": rmse_log,
                "mae_log_level": mae_log,
                "sample_count": sample_count,
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["horizon", "release_stage", "months_to_first_release_bucket", "model"]
    ).reset_index(drop=True)

    baseline_keys = [c for c in ["horizon", "release_stage", "months_to_first_release_bucket"] if c in metrics.columns]
    baseline_df = metrics.loc[metrics["model"] == baseline_model, baseline_keys + ["rmse"]].rename(
        columns={"rmse": "baseline_rmse"}
    )
    metrics = metrics.merge(baseline_df, on=baseline_keys, how="left")
    metrics["rel_rmse"] = np.where(
        (metrics["rmse"].notna()) & (metrics["baseline_rmse"].notna()) & (metrics["baseline_rmse"] > 0),
        metrics["rmse"] / metrics["baseline_rmse"],
        np.nan,
    )
    metrics = metrics.drop(columns=["baseline_rmse"])

    for col in output_cols:
        if col not in metrics.columns:
            metrics[col] = np.nan
    metrics = metrics[output_cols]
    metrics = metrics.loc[metrics["sample_count"] > 0].reset_index(drop=True)
    return metrics


__all__ = [
    "AR4GrowthModel",
    "BUILTIN_MODELS",
    "NaiveLastModel",
    "RandomWalkDriftLogModel",
    "RealtimeModel",
    "build_origin_datasets",
    "build_vintage_calendar",
    "compute_metrics",
    "compute_vintage_asof_date",
    "load_release_table",
    "load_vintage_panel",
    "months_to_first_release",
    "months_to_first_release_bucket",
    "resolve_models",
    "run_backtest",
    "select_training_vintage",
    "to_saar_growth",
    "to_quarter_period",
]
