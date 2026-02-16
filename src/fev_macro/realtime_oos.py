from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BMonthEnd

from .models import MODEL_REGISTRY, build_models
from .models.base import BaseModel
from .realtime_feeds import RealtimeTaskShim, select_covariate_columns, train_df_to_datasets

DEFAULT_MIN_FIRST_RELEASE_LAG_DAYS = 60
DEFAULT_MAX_FIRST_RELEASE_LAG_DAYS = 200
DEFAULT_TARGET_COL = "GDPC1"
DEFAULT_HORIZONS = (1, 2, 3, 4)
DEFAULT_MD_PANEL_PATH = Path("data/panels/fred_md_vintage_panel.parquet")
RELEASE_STAGE_TO_COL = {
    "first": "first_release",
    "second": "second_release",
    "third": "third_release",
}
RELEASE_STAGE_TO_REALTIME_SAAR_COL = {
    "first": "qoq_saar_growth_realtime_first_pct",
    "second": "qoq_saar_growth_realtime_second_pct",
    "third": "qoq_saar_growth_realtime_third_pct",
}

PREFERRED_MD_COVARIATES: tuple[str, ...] = (
    "UNRATE",
    "PAYEMS",
    "INDPRO",
    "IPFINAL",
    "RETAILx",
    "CMRMTSPLx",
    "RPI",
    "W875RX1",
    "M2REAL",
    "FEDFUNDS",
    "TB3MS",
    "GS10",
    "CP3Mx",
    "CPIAUCSL",
    "PCEPI",
    "OILPRICEx",
    "S&P 500",
    "UMCSENTx",
    "HOUST",
    "PERMIT",
)

_MD_VINTAGE_CACHE: dict[tuple[str, str, int], pd.DataFrame] = {}
_MD_PANEL_CACHE: dict[str, pd.DataFrame] = {}


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


def saar_growth_series_from_levels(y: pd.Series) -> pd.Series:
    values = pd.to_numeric(pd.Series(y), errors="coerce")
    prev = values.shift(1)
    out = pd.Series(index=values.index, dtype=float)
    for idx, (current, previous) in enumerate(zip(values.to_numpy(dtype=float), prev.to_numpy(dtype=float), strict=False)):
        out.iloc[idx] = to_saar_growth(current_level=current, previous_level=previous)
    return out


def levels_from_saar_growth(last_level: float, g_hat: np.ndarray) -> np.ndarray:
    if not np.isfinite(last_level) or last_level <= 0:
        raise ValueError("levels_from_saar_growth requires a positive finite last_level.")

    growth = np.asarray(g_hat, dtype=float).reshape(-1)
    out = np.full(growth.size, np.nan, dtype=float)
    level = float(last_level)
    for i, g in enumerate(growth):
        if not np.isfinite(g):
            continue
        quarter_factor = 1.0 + float(g) / 100.0
        if quarter_factor <= 0:
            continue
        level = float(level * (quarter_factor ** 0.25))
        out[i] = level
    return out


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

    realtime_first_col = RELEASE_STAGE_TO_REALTIME_SAAR_COL["first"]
    if realtime_first_col in df.columns:
        df["g_true_saar_first"] = pd.to_numeric(df[realtime_first_col], errors="coerce")
    else:
        print(
            "WARNING: release table is missing "
            f"'{realtime_first_col}'. Falling back to first_release.shift(1) for g_true_saar_first, "
            "which may be inconsistent around reindex/rebenchmark events."
        )
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


def _target_array(train_df: pd.DataFrame, target_col: str) -> np.ndarray:
    y = pd.to_numeric(train_df[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    if y.size == 0:
        raise ValueError("No target history available.")
    return y


def _target_series_and_last_level(train_df: pd.DataFrame, target_col: str) -> tuple[pd.Series, float]:
    y_all = pd.to_numeric(train_df[target_col], errors="coerce")
    observed = y_all.dropna()
    if observed.empty:
        raise ValueError("No target history available.")
    return y_all, float(observed.iloc[-1])


def _build_growth_target_frame(train_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, str, float]:
    df = train_df.copy().sort_values("quarter").reset_index(drop=True)
    y_all, last_level = _target_series_and_last_level(df, target_col)
    growth_target_col = "__target_saar_growth"
    df[growth_target_col] = saar_growth_series_from_levels(y_all)
    return df, growth_target_col, last_level


def _growth_path_from_last_growth(last_level: float, g_history: np.ndarray, horizon: int) -> np.ndarray:
    if g_history.size == 0:
        return np.repeat(last_level, horizon)
    g_last = float(g_history[-1]) if np.isfinite(g_history[-1]) else 0.0
    g_hat = np.repeat(g_last, horizon).astype(float)
    return levels_from_saar_growth(last_level=last_level, g_hat=g_hat)


def _recursive_log_level_regression(
    y: np.ndarray,
    horizon: int,
    regressor_builder,
    lags: int = 4,
) -> np.ndarray:
    if y.size == 0:
        raise ValueError("No target history available.")
    if np.any(y <= 0):
        raise ValueError("Model requires strictly positive levels.")
    if y.size <= lags + 1:
        return np.repeat(y[-1], horizon)

    ly = np.log(y)
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for t in range(lags, ly.size):
        x_rows.append(ly[t - lags : t].copy())
        y_rows.append(float(ly[t]))

    x = np.asarray(x_rows, dtype=float)
    y_target = np.asarray(y_rows, dtype=float)

    model = regressor_builder()
    model.fit(x, y_target)

    hist = list(ly.astype(float))
    preds_ly: list[float] = []
    for _ in range(horizon):
        feats = np.asarray(hist[-lags:], dtype=float).reshape(1, -1)
        pred = float(model.predict(feats)[0])
        preds_ly.append(pred)
        hist.append(pred)

    return np.exp(np.asarray(preds_ly, dtype=float))


def _load_md_vintage_quarterly_features(
    vintage_id: str,
    md_panel_path: Path = DEFAULT_MD_PANEL_PATH,
    max_covariates: int = 24,
) -> pd.DataFrame:
    key = (str(md_panel_path), str(vintage_id), int(max_covariates))
    if key in _MD_VINTAGE_CACHE:
        return _MD_VINTAGE_CACHE[key]

    panel_path = Path(md_panel_path)
    if not panel_path.exists():
        out = pd.DataFrame()
        _MD_VINTAGE_CACHE[key] = out
        return out

    cache_key = str(panel_path)
    if cache_key in _MD_PANEL_CACHE:
        panel = _MD_PANEL_CACHE[cache_key]
    else:
        panel = pd.read_parquet(panel_path)
        panel["vintage"] = panel["vintage"].astype(str)
        panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
        panel = panel.dropna(subset=["timestamp"]).sort_values(["vintage", "timestamp"]).reset_index(drop=True)
        _MD_PANEL_CACHE[cache_key] = panel

    d = panel.loc[panel["vintage"].astype(str) == str(vintage_id)].copy()
    if d.empty:
        out = pd.DataFrame()
        _MD_VINTAGE_CACHE[key] = out
        return out

    numeric_cols = [
        c
        for c in d.columns
        if c not in {"vintage", "timestamp", "vintage_timestamp"} and pd.api.types.is_numeric_dtype(d[c])
    ]
    selected: list[str] = []
    for col in PREFERRED_MD_COVARIATES:
        if col in numeric_cols and col not in selected:
            selected.append(col)
        if len(selected) >= max_covariates:
            break
    if len(selected) < max_covariates:
        for col in numeric_cols:
            if col in selected:
                continue
            s = pd.to_numeric(d[col], errors="coerce")
            if s.notna().sum() < 24:
                continue
            selected.append(col)
            if len(selected) >= max_covariates:
                break

    if not selected:
        out = pd.DataFrame()
        _MD_VINTAGE_CACHE[key] = out
        return out

    for col in selected:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["quarter"] = d["timestamp"].dt.to_period("Q-DEC")
    q = d.groupby("quarter", sort=True)[selected].last().ffill().bfill()
    q.columns = [f"md_{c}" for c in q.columns]
    out = q
    _MD_VINTAGE_CACHE[key] = out
    return out


def _augment_with_md_features(
    train_df: pd.DataFrame,
    md_panel_path: Path = DEFAULT_MD_PANEL_PATH,
    max_md_covariates: int = 24,
) -> pd.DataFrame:
    out = train_df.copy()
    vintage_id = str(out.get("__origin_vintage", pd.Series([np.nan])).iloc[-1])
    if vintage_id in {"nan", "None"}:
        return out

    q_md = _load_md_vintage_quarterly_features(
        vintage_id=vintage_id,
        md_panel_path=Path(md_panel_path),
        max_covariates=max_md_covariates,
    )
    if q_md.empty:
        return out

    q = pd.PeriodIndex(out["quarter"], freq="Q-DEC")
    q_md_aligned = q_md.reindex(q).ffill().bfill()
    for col in q_md_aligned.columns:
        out[col] = pd.to_numeric(q_md_aligned[col], errors="coerce").to_numpy(dtype=float)
    return out


def _recursive_log_level_regression_with_covariates(
    train_df: pd.DataFrame,
    target_col: str,
    horizon: int,
    regressor_builder,
    lags: int = 4,
    max_covariates: int = 80,
    include_md_features: bool = True,
    max_md_covariates: int = 24,
    md_panel_path: Path = DEFAULT_MD_PANEL_PATH,
) -> np.ndarray:
    df = train_df.copy().sort_values("quarter").reset_index(drop=True)
    if include_md_features:
        df = _augment_with_md_features(
            train_df=df,
            md_panel_path=md_panel_path,
            max_md_covariates=max_md_covariates,
        )

    y_all = pd.to_numeric(df[target_col], errors="coerce")
    y_mask = y_all.notna()
    if not y_mask.any():
        raise ValueError("No target history available.")

    y = y_all[y_mask].to_numpy(dtype=float)
    if y.size <= lags + 1 or np.any(y <= 0):
        return np.repeat(y[-1], horizon)

    q_all = pd.PeriodIndex(df["quarter"], freq="Q-DEC")
    q_hist = pd.PeriodIndex(df.loc[y_mask, "quarter"], freq="Q-DEC")
    q_future = [q_hist[-1] + i for i in range(1, horizon + 1)]

    covariates = _select_covariates_for_wrapped_models(
        train_df=df,
        target_col=target_col,
        max_covariates=max_covariates,
    )
    if not covariates:
        return _recursive_log_level_regression(y=y, horizon=horizon, regressor_builder=regressor_builder, lags=lags)

    cov_panel = (
        df.assign(__q=q_all)
        .groupby("__q", sort=True)[covariates]
        .last()
    )
    cov_panel = cov_panel.reindex(pd.PeriodIndex(list(q_hist) + list(q_future), freq="Q-DEC")).ffill().bfill().fillna(0.0)
    hist_cov = cov_panel.loc[q_hist, covariates].to_numpy(dtype=float)
    fut_cov = cov_panel.loc[q_future, covariates].to_numpy(dtype=float)

    ly = np.log(y)
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for t in range(lags, ly.size):
        row = np.concatenate([ly[t - lags : t], hist_cov[t]], axis=0)
        if not np.isfinite(row).all():
            continue
        x_rows.append(row)
        y_rows.append(float(ly[t]))

    if len(y_rows) < max(12, lags + 4):
        return _recursive_log_level_regression(y=y, horizon=horizon, regressor_builder=regressor_builder, lags=lags)

    x = np.asarray(x_rows, dtype=float)
    y_target = np.asarray(y_rows, dtype=float)

    model = regressor_builder()
    model.fit(x, y_target)

    hist_ly = list(ly.astype(float))
    preds_ly: list[float] = []
    for step in range(horizon):
        lag_vec = np.asarray(hist_ly[-lags:], dtype=float)
        feat = np.concatenate([lag_vec, fut_cov[step]], axis=0).reshape(1, -1)
        pred = float(model.predict(feat)[0])
        preds_ly.append(pred)
        hist_ly.append(pred)

    return np.exp(np.asarray(preds_ly, dtype=float))


def _recursive_growth_regression_with_covariates(
    train_df: pd.DataFrame,
    target_col: str,
    horizon: int,
    regressor_builder,
    lags: int = 4,
    max_covariates: int = 80,
    include_md_features: bool = True,
    max_md_covariates: int = 24,
    md_panel_path: Path = DEFAULT_MD_PANEL_PATH,
) -> np.ndarray:
    growth_df, growth_target_col, last_level = _build_growth_target_frame(train_df=train_df, target_col=target_col)
    if include_md_features:
        growth_df = _augment_with_md_features(
            train_df=growth_df,
            md_panel_path=md_panel_path,
            max_md_covariates=max_md_covariates,
        )

    g_all = pd.to_numeric(growth_df[growth_target_col], errors="coerce")
    g_mask = g_all.notna()
    if not g_mask.any():
        return np.repeat(last_level, horizon)

    g = g_all[g_mask].to_numpy(dtype=float)
    if g.size <= lags + 1:
        return _growth_path_from_last_growth(last_level=last_level, g_history=g, horizon=horizon)

    q_all = pd.PeriodIndex(growth_df["quarter"], freq="Q-DEC")
    q_hist = pd.PeriodIndex(growth_df.loc[g_mask, "quarter"], freq="Q-DEC")
    q_future = [q_hist[-1] + i for i in range(1, horizon + 1)]

    covariates = _select_covariates_for_wrapped_models(
        train_df=growth_df,
        target_col=growth_target_col,
        max_covariates=max_covariates,
    )
    if not covariates:
        return _growth_path_from_last_growth(last_level=last_level, g_history=g, horizon=horizon)

    cov_panel = growth_df.assign(__q=q_all).groupby("__q", sort=True)[covariates].last()
    cov_panel = cov_panel.reindex(pd.PeriodIndex(list(q_hist) + list(q_future), freq="Q-DEC")).ffill().bfill().fillna(0.0)
    hist_cov = cov_panel.loc[q_hist, covariates].to_numpy(dtype=float)
    fut_cov = cov_panel.loc[q_future, covariates].to_numpy(dtype=float)

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for t in range(lags, g.size):
        row = np.concatenate([g[t - lags : t], hist_cov[t]], axis=0)
        if not np.isfinite(row).all():
            continue
        x_rows.append(row)
        y_rows.append(float(g[t]))

    if len(y_rows) < max(12, lags + 4):
        return _growth_path_from_last_growth(last_level=last_level, g_history=g, horizon=horizon)

    x = np.asarray(x_rows, dtype=float)
    y_target = np.asarray(y_rows, dtype=float)

    model = regressor_builder()
    model.fit(x, y_target)

    hist_g = list(g.astype(float))
    preds_g: list[float] = []
    for step in range(horizon):
        lag_vec = np.asarray(hist_g[-lags:], dtype=float)
        feat = np.concatenate([lag_vec, fut_cov[step]], axis=0).reshape(1, -1)
        pred = float(model.predict(feat)[0])
        preds_g.append(pred)
        hist_g.append(pred)

    return levels_from_saar_growth(last_level=last_level, g_hat=np.asarray(preds_g, dtype=float))


@dataclass
class _TaskShim:
    horizon: int
    target_col: str
    past_dynamic_columns: list[str]
    known_dynamic_columns: list[str]
    id_column: str = "id"
    timestamp_column: str = "timestamp"
    freq: str = "Q"

    @property
    def target(self) -> str:
        return self.target_col

    @property
    def prediction_length(self) -> int:
        return self.horizon


def _select_covariates_for_wrapped_models(train_df: pd.DataFrame, target_col: str, max_covariates: int = 120) -> list[str]:
    exclude = {
        target_col,
        "quarter",
        "timestamp",
        "vintage",
        "vintage_timestamp",
        "asof_date",
        "__origin_vintage",
        "__origin_schedule",
    }
    candidates: list[str] = []
    for col in train_df.columns:
        if col in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        s = pd.to_numeric(train_df[col], errors="coerce")
        if s.notna().sum() < 8:
            continue
        candidates.append(col)
    return candidates[:max_covariates]


def _build_single_series_datasets(
    train_df: pd.DataFrame,
    target_col: str,
    horizon: int,
    covariate_cols: list[str],
) -> tuple[Any, Any, _TaskShim]:
    try:
        from datasets import Dataset
    except Exception as exc:
        raise ImportError("datasets package is required for wrapped fev_macro model adapters.") from exc
    df = train_df.copy().sort_values("quarter").reset_index(drop=True)
    q_all = pd.PeriodIndex(df["quarter"], freq="Q-DEC")
    y_all = pd.to_numeric(df[target_col], errors="coerce")
    mask = y_all.notna()
    if not mask.any():
        raise ValueError("No non-missing target in train_df for wrapped model.")

    q_hist = q_all[mask]
    y_hist = y_all[mask].to_numpy(dtype=float)
    past_ts = [period.start_time for period in q_hist]

    past_row: dict[str, object] = {
        "id": "series_1",
        "timestamp": past_ts,
        target_col: y_hist.tolist(),
    }
    for col in covariate_cols:
        vals = pd.to_numeric(df.loc[mask, col], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        past_row[col] = vals.tolist()

    last_q = q_hist[-1]
    fut_q = [last_q + i for i in range(1, horizon + 1)]
    future_ts = [period.start_time for period in fut_q]
    future_row: dict[str, object] = {"id": "series_1", "timestamp": future_ts}
    for col in covariate_cols:
        hist_cov = pd.to_numeric(df.loc[mask, col], errors="coerce").ffill().bfill().fillna(0.0)
        last_val = float(hist_cov.iloc[-1]) if len(hist_cov) else 0.0
        all_cov = pd.to_numeric(df[col], errors="coerce")
        cov_map = pd.Series(all_cov.to_numpy(dtype=float), index=q_all).groupby(level=0).last()

        future_vals: list[float] = []
        for fq in fut_q:
            v = cov_map.get(fq, np.nan)
            if pd.notna(v):
                last_val = float(v)
            future_vals.append(last_val)
        future_row[col] = future_vals

    past_data = Dataset.from_list([past_row])
    future_data = Dataset.from_list([future_row])
    task = _TaskShim(
        horizon=horizon,
        target_col=target_col,
        past_dynamic_columns=list(covariate_cols),
        known_dynamic_columns=list(covariate_cols),
    )
    return past_data, future_data, task


class _WrappedFevModel(RealtimeModel):
    def __init__(self, name: str, builder, max_covariates: int = 120) -> None:
        super().__init__(name=name)
        self._builder = builder
        self.max_covariates = int(max_covariates)
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = self._builder()
        return self._model

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        covs = _select_covariates_for_wrapped_models(train_df=train_df, target_col=target_col, max_covariates=self.max_covariates)
        try:
            past_data, future_data, task = _build_single_series_datasets(
                train_df=train_df,
                target_col=target_col,
                horizon=horizon,
                covariate_cols=covs,
            )
            pred_ds = self._get_model().predict(past_data=past_data, future_data=future_data, task=task)
            arr = np.asarray(pred_ds["predictions"][0], dtype=float).reshape(-1)
            if arr.size != horizon or not np.isfinite(arr).all():
                raise ValueError("Wrapped model returned invalid forecasts.")
            return arr
        except Exception:
            return np.repeat(y[-1], horizon)


class _WrappedFevGrowthModel(RealtimeModel):
    def __init__(self, name: str, builder, max_covariates: int = 120) -> None:
        super().__init__(name=name)
        self._builder = builder
        self.max_covariates = int(max_covariates)
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = self._builder()
        return self._model

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        try:
            growth_df, growth_target_col, last_level = _build_growth_target_frame(train_df=train_df, target_col=target_col)
            g_hist = pd.to_numeric(growth_df[growth_target_col], errors="coerce").dropna().to_numpy(dtype=float)
            if g_hist.size < 8:
                return _growth_path_from_last_growth(last_level=last_level, g_history=g_hist, horizon=horizon)

            covs = _select_covariates_for_wrapped_models(
                train_df=growth_df,
                target_col=growth_target_col,
                max_covariates=self.max_covariates,
            )
            past_data, future_data, task = _build_single_series_datasets(
                train_df=growth_df,
                target_col=growth_target_col,
                horizon=horizon,
                covariate_cols=covs,
            )
            pred_ds = self._get_model().predict(past_data=past_data, future_data=future_data, task=task)
            g_hat = np.asarray(pred_ds["predictions"][0], dtype=float).reshape(-1)
            if g_hat.size != horizon or not np.isfinite(g_hat).all():
                raise ValueError("Wrapped model returned invalid growth forecasts.")
            return levels_from_saar_growth(last_level=last_level, g_hat=g_hat)
        except Exception:
            y = _target_array(train_df=train_df, target_col=target_col)
            return np.repeat(y[-1], horizon)


class NaiveLastModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="naive_last")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        return np.repeat(y[-1], horizon)


class NaiveLastGrowthModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="naive_last_growth")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y_all, last_level = _target_series_and_last_level(train_df=train_df, target_col=target_col)
        g_hist = saar_growth_series_from_levels(y_all).dropna().to_numpy(dtype=float)
        return _growth_path_from_last_growth(last_level=last_level, g_history=g_hist, horizon=horizon)


class RandomWalkDriftLogModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="rw_drift_log")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
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
        y = _target_array(train_df=train_df, target_col=target_col)
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


class MeanLevelModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="mean")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        return np.repeat(float(np.mean(y)), horizon)


class DriftLevelModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="drift")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        if y.size == 1:
            return np.repeat(y[-1], horizon)
        slope = (y[-1] - y[0]) / float(y.size - 1)
        steps = np.arange(1, horizon + 1, dtype=float)
        return y[-1] + slope * steps


class SeasonalNaiveLevelModel(RealtimeModel):
    def __init__(self, season_length: int = 4) -> None:
        super().__init__(name="seasonal_naive")
        self.season_length = int(season_length)

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        template = y[-self.season_length :] if y.size >= self.season_length else y
        return np.resize(template, horizon)


class RandomNormalModel(RealtimeModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_normal")
        self.seed = int(seed)

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        rng = np.random.default_rng(self.seed)
        mu = float(np.mean(y))
        sigma = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
        sigma = max(sigma, 1e-9)
        return rng.normal(loc=mu, scale=sigma, size=horizon).astype(float)


class RandomUniformModel(RealtimeModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_uniform")
        self.seed = int(seed)

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        rng = np.random.default_rng(self.seed)
        lo = float(np.min(y))
        hi = float(np.max(y))
        if np.isclose(lo, hi):
            return np.repeat(lo, horizon)
        return rng.uniform(low=lo, high=hi, size=horizon).astype(float)


class RandomPermutationModel(RealtimeModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_permutation")
        self.seed = int(seed)

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(y)
        if perm.size >= horizon:
            return perm[:horizon].astype(float)
        reps = int(np.ceil(horizon / perm.size))
        return np.tile(perm, reps)[:horizon].astype(float)


def _growth_training_history(
    train_df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, pd.PeriodIndex, float]:
    growth_df, growth_target_col, last_level = _build_growth_target_frame(train_df=train_df, target_col=target_col)
    g_all = pd.to_numeric(growth_df[growth_target_col], errors="coerce")
    mask = g_all.notna()
    g_hist = g_all[mask].to_numpy(dtype=float)
    q_hist = pd.PeriodIndex(growth_df.loc[mask, "quarter"], freq="Q-DEC")
    return g_hist, q_hist, last_level


def _forecast_growth_with_statsforecast(
    train_df: pd.DataFrame,
    target_col: str,
    horizon: int,
    model_builder,
) -> np.ndarray:
    g_hist, q_hist, last_level = _growth_training_history(train_df=train_df, target_col=target_col)
    if g_hist.size < 8:
        return _growth_path_from_last_growth(last_level=last_level, g_history=g_hist, horizon=horizon)

    ds = q_hist.to_timestamp(how="end")
    sf_df = pd.DataFrame({"unique_id": "__series__", "ds": ds, "y": g_hist})

    try:
        from statsforecast import StatsForecast

        sf = StatsForecast(models=[model_builder()], freq="Q")
        fc = sf.forecast(df=sf_df, h=horizon)
        point_col = [c for c in fc.columns if c not in {"unique_id", "ds"}][0]
        g_hat = fc[point_col].to_numpy(dtype=float)
        if g_hat.size != horizon or not np.isfinite(g_hat).all():
            raise ValueError("statsforecast returned invalid growth forecasts.")
        return levels_from_saar_growth(last_level=last_level, g_hat=g_hat)
    except Exception:
        return _growth_path_from_last_growth(last_level=last_level, g_history=g_hist, horizon=horizon)


class AutoARIMALevelModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="auto_arima")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        ds_col = "quarter" if "quarter" in train_df.columns else None
        if ds_col is None:
            ds = pd.period_range("2000Q1", periods=y.size, freq="Q-DEC").to_timestamp(how="end")
        else:
            ds = pd.PeriodIndex(train_df.loc[train_df[target_col].notna(), ds_col], freq="Q-DEC").to_timestamp(how="end")
        sf_df = pd.DataFrame({"unique_id": "__series__", "ds": ds, "y": y})

        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA

            sf = StatsForecast(models=[AutoARIMA(season_length=4)], freq="Q")
            fc = sf.forecast(df=sf_df, h=horizon)
            point_col = [c for c in fc.columns if c not in {"unique_id", "ds"}][0]
            return fc[point_col].to_numpy(dtype=float)
        except Exception:
            return np.repeat(y[-1], horizon)


class AutoARIMAGrowthModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="auto_arima_growth")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        def _builder():
            from statsforecast.models import AutoARIMA

            return AutoARIMA(season_length=4)

        return _forecast_growth_with_statsforecast(
            train_df=train_df,
            target_col=target_col,
            horizon=horizon,
            model_builder=_builder,
        )


class AutoETSLevelModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="auto_ets")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        ds = pd.period_range("2000Q1", periods=y.size, freq="Q-DEC").to_timestamp(how="end")
        sf_df = pd.DataFrame({"unique_id": "__series__", "ds": ds, "y": y})

        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoETS

            try:
                model = AutoETS(season_length=4)
            except TypeError:
                model = AutoETS()
            sf = StatsForecast(models=[model], freq="Q")
            fc = sf.forecast(df=sf_df, h=horizon)
            point_col = [c for c in fc.columns if c not in {"unique_id", "ds"}][0]
            return fc[point_col].to_numpy(dtype=float)
        except Exception:
            return np.repeat(y[-1], horizon)


class AutoETSGrowthModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="auto_ets_growth")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        def _builder():
            from statsforecast.models import AutoETS

            try:
                return AutoETS(season_length=4)
            except TypeError:
                return AutoETS()

        return _forecast_growth_with_statsforecast(
            train_df=train_df,
            target_col=target_col,
            horizon=horizon,
            model_builder=_builder,
        )


class ThetaLevelModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="theta")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        ds = pd.period_range("2000Q1", periods=y.size, freq="Q-DEC").to_timestamp(how="end")
        sf_df = pd.DataFrame({"unique_id": "__series__", "ds": ds, "y": y})

        try:
            from statsforecast import StatsForecast
            from statsforecast.models import Theta

            try:
                model = Theta(season_length=4)
            except TypeError:
                model = Theta()
            sf = StatsForecast(models=[model], freq="Q")
            fc = sf.forecast(df=sf_df, h=horizon)
            point_col = [c for c in fc.columns if c not in {"unique_id", "ds"}][0]
            return fc[point_col].to_numpy(dtype=float)
        except Exception:
            return np.repeat(y[-1], horizon)


class ThetaGrowthModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="theta_growth")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        def _builder():
            from statsforecast.models import Theta

            try:
                return Theta(season_length=4)
            except TypeError:
                return Theta()

        return _forecast_growth_with_statsforecast(
            train_df=train_df,
            target_col=target_col,
            horizon=horizon,
            model_builder=_builder,
        )


class LocalTrendSSMModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="local_trend_ssm")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y = _target_array(train_df=train_df, target_col=target_col)
        if y.size < 6:
            return np.repeat(y[-1], horizon)

        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents

            model = UnobservedComponents(endog=np.log(y), level="local linear trend")
            res = model.fit(disp=False)
            pred_log = np.asarray(res.forecast(steps=horizon), dtype=float)
            return np.exp(pred_log)
        except Exception:
            return np.repeat(y[-1], horizon)


class LocalTrendSSMGrowthModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="local_trend_ssm_growth")

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        g_hist, _, last_level = _growth_training_history(train_df=train_df, target_col=target_col)
        if g_hist.size < 6:
            return _growth_path_from_last_growth(last_level=last_level, g_history=g_hist, horizon=horizon)

        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents

            model = UnobservedComponents(endog=g_hist, level="local linear trend")
            res = model.fit(disp=False)
            g_hat = np.asarray(res.forecast(steps=horizon), dtype=float)
            if g_hat.size != horizon or not np.isfinite(g_hat).all():
                raise ValueError("local_trend_ssm_growth produced invalid growth forecasts.")
            return levels_from_saar_growth(last_level=last_level, g_hat=g_hat)
        except Exception:
            return _growth_path_from_last_growth(last_level=last_level, g_history=g_hist, horizon=horizon)


class RandomForestLevelModel(RealtimeModel):
    def __init__(
        self,
        seed: int = 0,
        lags: int = 4,
        max_covariates: int = 80,
        max_md_covariates: int = 24,
        md_panel_path: Path = DEFAULT_MD_PANEL_PATH,
    ) -> None:
        super().__init__(name="random_forest")
        self.seed = int(seed)
        self.lags = int(lags)
        self.max_covariates = int(max_covariates)
        self.max_md_covariates = int(max_md_covariates)
        self.md_panel_path = Path(md_panel_path)

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        def _builder():
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                random_state=self.seed,
                n_jobs=1,
            )

        return _recursive_log_level_regression_with_covariates(
            train_df=train_df,
            target_col=target_col,
            horizon=horizon,
            regressor_builder=_builder,
            lags=self.lags,
            max_covariates=self.max_covariates,
            include_md_features=True,
            max_md_covariates=self.max_md_covariates,
            md_panel_path=self.md_panel_path,
        )


class RandomForestGrowthModel(RandomForestLevelModel):
    def __init__(
        self,
        seed: int = 0,
        lags: int = 4,
        max_covariates: int = 80,
        max_md_covariates: int = 24,
        md_panel_path: Path = DEFAULT_MD_PANEL_PATH,
    ) -> None:
        super().__init__(
            seed=seed,
            lags=lags,
            max_covariates=max_covariates,
            max_md_covariates=max_md_covariates,
            md_panel_path=md_panel_path,
        )
        self.name = "random_forest_growth"

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        def _builder():
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                random_state=self.seed,
                n_jobs=1,
            )

        return _recursive_growth_regression_with_covariates(
            train_df=train_df,
            target_col=target_col,
            horizon=horizon,
            regressor_builder=_builder,
            lags=self.lags,
            max_covariates=self.max_covariates,
            include_md_features=True,
            max_md_covariates=self.max_md_covariates,
            md_panel_path=self.md_panel_path,
        )


class XGBoostLevelModel(RealtimeModel):
    def __init__(
        self,
        seed: int = 0,
        lags: int = 4,
        max_covariates: int = 80,
        max_md_covariates: int = 24,
        md_panel_path: Path = DEFAULT_MD_PANEL_PATH,
    ) -> None:
        super().__init__(name="xgboost")
        self.seed = int(seed)
        self.lags = int(lags)
        self.max_covariates = int(max_covariates)
        self.max_md_covariates = int(max_md_covariates)
        self.md_panel_path = Path(md_panel_path)

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        def _builder():
            import xgboost as xgb

            return xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.seed,
                n_jobs=1,
            )

        return _recursive_log_level_regression_with_covariates(
            train_df=train_df,
            target_col=target_col,
            horizon=horizon,
            regressor_builder=_builder,
            lags=self.lags,
            max_covariates=self.max_covariates,
            include_md_features=True,
            max_md_covariates=self.max_md_covariates,
            md_panel_path=self.md_panel_path,
        )


class XGBoostGrowthModel(XGBoostLevelModel):
    def __init__(
        self,
        seed: int = 0,
        lags: int = 4,
        max_covariates: int = 80,
        max_md_covariates: int = 24,
        md_panel_path: Path = DEFAULT_MD_PANEL_PATH,
    ) -> None:
        super().__init__(
            seed=seed,
            lags=lags,
            max_covariates=max_covariates,
            max_md_covariates=max_md_covariates,
            md_panel_path=md_panel_path,
        )
        self.name = "xgboost_growth"

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        def _builder():
            import xgboost as xgb

            return xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.seed,
                n_jobs=1,
            )

        return _recursive_growth_regression_with_covariates(
            train_df=train_df,
            target_col=target_col,
            horizon=horizon,
            regressor_builder=_builder,
            lags=self.lags,
            max_covariates=self.max_covariates,
            include_md_features=True,
            max_md_covariates=self.max_md_covariates,
            md_panel_path=self.md_panel_path,
        )


class FactorPCAQDRealtimeModel(_WrappedFevModel):
    def __init__(self) -> None:
        super().__init__(
            name="factor_pca_qd",
            builder=lambda: __import__("fev_macro.models.factor_models", fromlist=["QuarterlyFactorPCAModel"]).QuarterlyFactorPCAModel(),
            max_covariates=120,
        )


class FactorPCAQDGrowthRealtimeModel(_WrappedFevGrowthModel):
    def __init__(self) -> None:
        super().__init__(
            name="factor_pca_qd_growth",
            builder=lambda: __import__("fev_macro.models.factor_models", fromlist=["QuarterlyFactorPCAModel"]).QuarterlyFactorPCAModel(),
            max_covariates=120,
        )


class BVARMinnesota8RealtimeModel(_WrappedFevModel):
    def __init__(self) -> None:
        super().__init__(
            name="bvar_minnesota_8",
            builder=lambda: __import__("fev_macro.models.bvar_minnesota", fromlist=["BVARMinnesota8Model"]).BVARMinnesota8Model(),
            max_covariates=120,
        )


class BVARMinnesota20RealtimeModel(_WrappedFevModel):
    def __init__(self) -> None:
        super().__init__(
            name="bvar_minnesota_20",
            builder=lambda: __import__("fev_macro.models.bvar_minnesota", fromlist=["BVARMinnesota20Model"]).BVARMinnesota20Model(),
            max_covariates=120,
        )


class BVARMinnesotaGrowth8RealtimeModel(_WrappedFevGrowthModel):
    def __init__(self) -> None:
        super().__init__(
            name="bvar_minnesota_growth_8",
            builder=lambda: __import__(
                "fev_macro.models.bvar_minnesota",
                fromlist=["BVARMinnesotaGrowth8Model"],
            ).BVARMinnesotaGrowth8Model(),
            max_covariates=120,
        )


class BVARMinnesotaGrowth20RealtimeModel(_WrappedFevGrowthModel):
    def __init__(self) -> None:
        super().__init__(
            name="bvar_minnesota_growth_20",
            builder=lambda: __import__(
                "fev_macro.models.bvar_minnesota",
                fromlist=["BVARMinnesotaGrowth20Model"],
            ).BVARMinnesotaGrowth20Model(),
            max_covariates=120,
        )


class Chronos2RealtimeModel(_WrappedFevModel):
    def __init__(self) -> None:
        super().__init__(
            name="chronos2",
            builder=lambda: __import__("fev_macro.models.chronos2", fromlist=["Chronos2Model"]).Chronos2Model(),
            max_covariates=40,
        )


class Chronos2GrowthRealtimeModel(_WrappedFevGrowthModel):
    def __init__(self) -> None:
        super().__init__(
            name="chronos2_growth",
            builder=lambda: __import__("fev_macro.models.chronos2", fromlist=["Chronos2Model"]).Chronos2Model(),
            max_covariates=40,
        )


class MixedFreqDFMMDVintageModel(RealtimeModel):
    def __init__(
        self,
        md_panel_path: str = "data/panels/fred_md_vintage_panel.parquet",
        max_monthly_covariates: int = 100,
        n_factors: int = 6,
        max_lag: int = 2,
        alpha: float = 1.0,
        target_mode: str = "level",
        excluded_years: Sequence[int] = (),
        seed: int = 0,
    ) -> None:
        normalized_target_mode = str(target_mode).strip().lower()
        if normalized_target_mode not in {"level", "growth"}:
            raise ValueError("target_mode must be one of {'level', 'growth'}")

        model_name = "mixed_freq_dfm_md_growth" if normalized_target_mode == "growth" else "mixed_freq_dfm_md"
        super().__init__(name=model_name)
        self.md_panel_path = Path(md_panel_path)
        self.max_monthly_covariates = int(max_monthly_covariates)
        self.n_factors = int(n_factors)
        self.max_lag = int(max_lag)
        self.alpha = float(alpha)
        self.target_mode = normalized_target_mode
        self.excluded_years = tuple(sorted({int(y) for y in excluded_years}))
        self.seed = int(seed)
        self._panel: pd.DataFrame | None = None
        self._cache_quarterly: dict[str, pd.DataFrame] = {}

    def _load_panel(self) -> pd.DataFrame:
        if self._panel is not None:
            return self._panel
        if not self.md_panel_path.exists():
            raise FileNotFoundError(f"FRED-MD vintage panel not found: {self.md_panel_path}")
        df = pd.read_parquet(self.md_panel_path)
        df["vintage"] = df["vintage"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        keep = [c for c in df.columns if c not in {"vintage", "vintage_timestamp", "timestamp"}]
        keep = keep[: self.max_monthly_covariates]
        out = df[["vintage", "timestamp", *keep]].copy()
        for c in keep:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.sort_values(["vintage", "timestamp"]).reset_index(drop=True)
        self._panel = out
        return out

    def _quarterly_factors(self, vintage_id: str) -> pd.DataFrame:
        if vintage_id in self._cache_quarterly:
            return self._cache_quarterly[vintage_id]

        panel = self._load_panel()
        sub = panel.loc[panel["vintage"] == str(vintage_id)].copy()
        if sub.empty:
            raise ValueError(f"No FRED-MD vintage rows for {vintage_id}")
        covars = [c for c in sub.columns if c not in {"vintage", "timestamp"}]
        sub = sub.sort_values("timestamp")
        sub[covars] = sub[covars].ffill().bfill().fillna(0.0)
        sub["quarter"] = sub["timestamp"].dt.to_period("Q-DEC")
        q = sub.groupby("quarter", sort=True)[covars].mean(numeric_only=True).ffill().bfill().fillna(0.0)
        if self.excluded_years:
            q = q.loc[~q.index.year.isin(list(self.excluded_years))]
        self._cache_quarterly[vintage_id] = q
        return q

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        y_series = pd.to_numeric(train_df[target_col], errors="coerce")
        observed_levels = y_series.dropna()
        if observed_levels.empty:
            raise ValueError("No target history available.")

        last_level = float(observed_levels.iloc[-1])
        if self.target_mode == "growth":
            growth_series = saar_growth_series_from_levels(y_series)
            target_mask = growth_series.notna()
            y = growth_series[target_mask].to_numpy(dtype=float)
            y_quarters = pd.PeriodIndex(train_df.loc[target_mask, "quarter"], freq="Q-DEC")
            fallback = _growth_path_from_last_growth(last_level=last_level, g_history=y, horizon=horizon)
        else:
            target_mask = y_series.notna()
            y = y_series[target_mask].to_numpy(dtype=float)
            y_quarters = pd.PeriodIndex(train_df.loc[target_mask, "quarter"], freq="Q-DEC")
            fallback = np.repeat(last_level, horizon)

        if y.size < 20:
            return fallback

        vintage_id = str(train_df.get("__origin_vintage", pd.Series([np.nan])).iloc[-1])
        if vintage_id in {"nan", "None"}:
            return fallback

        try:
            qf = self._quarterly_factors(vintage_id=vintage_id)
        except Exception:
            return fallback

        cutoff_q = pd.Period(train_df["quarter"].max(), freq="Q-DEC")
        qf = qf.loc[qf.index <= cutoff_q]
        if qf.empty:
            return fallback

        qf_train = qf.reindex(y_quarters).ffill().bfill().fillna(0.0)
        if qf_train.empty:
            return fallback
        F_past = qf_train.to_numpy(dtype=float)

        future_quarters = [y_quarters[-1] + i for i in range(1, horizon + 1)]
        last_factor = F_past[-1].copy()
        future_rows: list[np.ndarray] = []
        for fq in future_quarters:
            if fq in qf.index:
                vec = qf.loc[fq].to_numpy(dtype=float)
                vec = np.where(np.isfinite(vec), vec, last_factor)
                last_factor = vec
            else:
                vec = last_factor
            future_rows.append(vec.copy())
        F_future = np.vstack(future_rows)

        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return fallback

        lag = min(self.max_lag, max(1, y.size // 12))
        if y.size <= lag + 4:
            return fallback

        scaler = StandardScaler(with_mean=True, with_std=True)
        F_scaled = scaler.fit_transform(F_past)
        n_f = max(1, min(self.n_factors, F_scaled.shape[1], F_scaled.shape[0] - 1))
        pca = PCA(n_components=n_f, random_state=self.seed)
        Z_past = pca.fit_transform(F_scaled)
        Z_future = pca.transform(scaler.transform(F_future))

        rows: list[np.ndarray] = []
        targets: list[float] = []
        for t in range(lag, y.size):
            rows.append(np.concatenate([y[t - lag : t], Z_past[t]], axis=0))
            targets.append(float(y[t]))

        X_train = np.vstack(rows)
        y_train = np.asarray(targets, dtype=float)
        if y_train.size < max(10, lag + 3):
            return fallback

        reg = Ridge(alpha=self.alpha, random_state=self.seed)
        reg.fit(X_train, y_train)

        y_hist = list(y.astype(float))
        forecasts: list[float] = []
        for step in range(horizon):
            lag_vals = np.asarray(y_hist[-lag:], dtype=float)
            if lag_vals.size < lag:
                lag_vals = np.pad(lag_vals, (lag - lag_vals.size, 0), mode="edge")
            feat = np.concatenate([lag_vals, Z_future[step]], axis=0)
            pred = float(reg.predict(feat.reshape(1, -1))[0])
            if not np.isfinite(pred):
                pred = float(y_hist[-1])
            forecasts.append(pred)
            y_hist.append(pred)
        fc = np.asarray(forecasts, dtype=float)
        if self.target_mode == "growth":
            return levels_from_saar_growth(last_level=last_level, g_hat=fc)
        return fc


class MixedFreqDFMMDVintageGrowthModel(MixedFreqDFMMDVintageModel):
    def __init__(
        self,
        md_panel_path: str = "data/panels/fred_md_vintage_panel.parquet",
        max_monthly_covariates: int = 100,
        n_factors: int = 6,
        max_lag: int = 2,
        alpha: float = 1.0,
        excluded_years: Sequence[int] = (),
        seed: int = 0,
    ) -> None:
        super().__init__(
            md_panel_path=md_panel_path,
            max_monthly_covariates=max_monthly_covariates,
            n_factors=n_factors,
            max_lag=max_lag,
            alpha=alpha,
            target_mode="growth",
            excluded_years=excluded_years,
            seed=seed,
        )


class EnsembleAvgTop3RealtimeModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="ensemble_avg_top3")
        self.members = [DriftLevelModel(), AutoARIMALevelModel(), LocalTrendSSMModel()]

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        preds = [m.forecast_levels(train_df=train_df, horizon=horizon, target_col=target_col) for m in self.members]
        arr = np.stack(preds, axis=0)
        return arr.mean(axis=0)


class EnsembleWeightedTop5RealtimeModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="ensemble_weighted_top5")
        self.members = [
            DriftLevelModel(),
            AutoARIMALevelModel(),
            LocalTrendSSMModel(),
            FactorPCAQDRealtimeModel(),
            SeasonalNaiveLevelModel(season_length=4),
        ]
        self.weights = np.asarray([0.28, 0.25, 0.20, 0.17, 0.10], dtype=float)
        self.weights = self.weights / self.weights.sum()

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        preds = [m.forecast_levels(train_df=train_df, horizon=horizon, target_col=target_col) for m in self.members]
        arr = np.stack(preds, axis=0)
        return np.tensordot(self.weights, arr, axes=(0, 0))


class EnsembleAvgTop3UnprocessedLLRealtimeModel(EnsembleAvgTop3RealtimeModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = "ensemble_avg_top3_unprocessed_ll"


class EnsembleWeightedTop5UnprocessedLLRealtimeModel(EnsembleWeightedTop5RealtimeModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = "ensemble_weighted_top5_unprocessed_ll"


class EnsembleAvgTop3ProcessedGRealtimeModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="ensemble_avg_top3_processed_g")
        self.members = [AutoARIMAGrowthModel(), LocalTrendSSMGrowthModel(), BVARMinnesotaGrowth8RealtimeModel()]

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        preds = [m.forecast_levels(train_df=train_df, horizon=horizon, target_col=target_col) for m in self.members]
        arr = np.stack(preds, axis=0)
        return arr.mean(axis=0)


class EnsembleWeightedTop5ProcessedGRealtimeModel(RealtimeModel):
    def __init__(self) -> None:
        super().__init__(name="ensemble_weighted_top5_processed_g")
        self.members = [
            AutoARIMAGrowthModel(),
            ThetaGrowthModel(),
            LocalTrendSSMGrowthModel(),
            FactorPCAQDGrowthRealtimeModel(),
            BVARMinnesotaGrowth8RealtimeModel(),
        ]
        self.weights = np.asarray([0.26, 0.22, 0.20, 0.17, 0.15], dtype=float)
        self.weights = self.weights / self.weights.sum()

    def forecast_levels(self, train_df: pd.DataFrame, horizon: int, target_col: str) -> np.ndarray:
        preds = [m.forecast_levels(train_df=train_df, horizon=horizon, target_col=target_col) for m in self.members]
        arr = np.stack(preds, axis=0)
        return np.tensordot(self.weights, arr, axes=(0, 0))


BUILTIN_MODELS: dict[str, type[RealtimeModel]] = {
    "mean": MeanLevelModel,
    "drift": DriftLevelModel,
    "seasonal_naive": SeasonalNaiveLevelModel,
    "naive_last": NaiveLastModel,
    "naive_last_growth": NaiveLastGrowthModel,
    "random_normal": RandomNormalModel,
    "random_uniform": RandomUniformModel,
    "random_permutation": RandomPermutationModel,
    "rw_drift_log": RandomWalkDriftLogModel,
    "ar4_growth": AR4GrowthModel,
    "auto_arima": AutoARIMALevelModel,
    "auto_arima_growth": AutoARIMAGrowthModel,
    "auto_ets": AutoETSLevelModel,
    "auto_ets_growth": AutoETSGrowthModel,
    "theta": ThetaLevelModel,
    "theta_growth": ThetaGrowthModel,
    "local_trend_ssm": LocalTrendSSMModel,
    "local_trend_ssm_growth": LocalTrendSSMGrowthModel,
    "random_forest": RandomForestLevelModel,
    "random_forest_growth": RandomForestGrowthModel,
    "xgboost": XGBoostLevelModel,
    "xgboost_growth": XGBoostGrowthModel,
    "factor_pca_qd": FactorPCAQDRealtimeModel,
    "factor_pca_qd_growth": FactorPCAQDGrowthRealtimeModel,
    "bvar_minnesota_8": BVARMinnesota8RealtimeModel,
    "bvar_minnesota_20": BVARMinnesota20RealtimeModel,
    "bvar_minnesota_growth_8": BVARMinnesotaGrowth8RealtimeModel,
    "bvar_minnesota_growth_20": BVARMinnesotaGrowth20RealtimeModel,
    "mixed_freq_dfm_md": MixedFreqDFMMDVintageModel,
    "mixed_freq_dfm_md_growth": MixedFreqDFMMDVintageGrowthModel,
    "chronos2": Chronos2RealtimeModel,
    "chronos2_growth": Chronos2GrowthRealtimeModel,
    "ensemble_avg_top3": EnsembleAvgTop3RealtimeModel,
    "ensemble_avg_top3_unprocessed_ll": EnsembleAvgTop3UnprocessedLLRealtimeModel,
    "ensemble_avg_top3_processed_g": EnsembleAvgTop3ProcessedGRealtimeModel,
    "ensemble_weighted_top5": EnsembleWeightedTop5RealtimeModel,
    "ensemble_weighted_top5_unprocessed_ll": EnsembleWeightedTop5UnprocessedLLRealtimeModel,
    "ensemble_weighted_top5_processed_g": EnsembleWeightedTop5ProcessedGRealtimeModel,
}


class _RealtimeModelAdapter(BaseModel):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(name=realtime_model.name)
        self.realtime_model = realtime_model

    def predict(self, past_data, future_data, task):
        from datasets import Dataset

        train_df = getattr(task, "train_df", None)
        target_col = getattr(task, "target_col", getattr(task, "target", DEFAULT_TARGET_COL))
        if train_df is None:
            raise ValueError("Realtime adapter requires task.train_df.")
        horizon = int(getattr(task, "horizon", getattr(task, "prediction_length")))
        path = np.asarray(
            self.realtime_model.forecast_levels(train_df=train_df, horizon=horizon, target_col=target_col),
            dtype=float,
        )
        return Dataset.from_dict({"predictions": [path.tolist()]})


def resolve_models(models: Sequence[str | RealtimeModel | BaseModel]) -> list[BaseModel]:
    resolved: list[BaseModel] = []
    for m in models:
        if isinstance(m, BaseModel):
            resolved.append(m)
            continue
        if isinstance(m, RealtimeModel):
            resolved.append(_RealtimeModelAdapter(m))
            continue

        key = str(m).strip().lower()
        if key in BUILTIN_MODELS:
            resolved.append(_RealtimeModelAdapter(BUILTIN_MODELS[key]()))
            continue
        if key in MODEL_REGISTRY:
            resolved.append(build_models([key], seed=0)[key])
            continue
        raise ValueError(f"Unknown model '{m}'. Available: {sorted(set(BUILTIN_MODELS) | set(MODEL_REGISTRY))}")

    if not resolved:
        raise ValueError("No models were provided.")

    return resolved


def apply_model_runtime_options(
    model_list: Sequence[BaseModel],
    *,
    md_panel_path: str | Path | None = None,
    mixed_freq_excluded_years: Sequence[int] | None = None,
) -> None:
    md_path = Path(md_panel_path).expanduser().resolve() if md_panel_path else None
    excluded_years = tuple(sorted({int(y) for y in (mixed_freq_excluded_years or [])}))

    for model in model_list:
        realtime_model = getattr(model, "realtime_model", None)
        if realtime_model is None:
            continue

        if md_path is not None and hasattr(realtime_model, "md_panel_path"):
            setattr(realtime_model, "md_panel_path", Path(md_path))
            if hasattr(realtime_model, "_panel"):
                setattr(realtime_model, "_panel", None)
            if hasattr(realtime_model, "_cache_quarterly"):
                setattr(realtime_model, "_cache_quarterly", {})

        if hasattr(realtime_model, "excluded_years"):
            setattr(realtime_model, "excluded_years", excluded_years)
            if hasattr(realtime_model, "_cache_quarterly"):
                setattr(realtime_model, "_cache_quarterly", {})


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
    models: Sequence[str | RealtimeModel | BaseModel],
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
    ragged_edge_covariates: bool = True,
    md_panel_path: str | Path | None = None,
    mixed_freq_excluded_years: Sequence[int] | None = None,
) -> pd.DataFrame:
    if origin_schedule not in {"quarterly", "monthly"}:
        raise ValueError("origin_schedule must be one of {'quarterly', 'monthly'}")

    horizon_list = sorted({int(h) for h in horizons if int(h) >= 1})
    if not horizon_list:
        raise ValueError("At least one horizon >= 1 is required.")

    stage_list = _normalize_release_stages(release_stages=release_stages, origin_schedule=origin_schedule)
    min_target_period = pd.Period(min_target_quarter, freq="Q-DEC") if min_target_quarter is not None else None
    model_list = resolve_models(models)
    apply_model_runtime_options(
        model_list=model_list,
        md_panel_path=md_panel_path,
        mixed_freq_excluded_years=mixed_freq_excluded_years,
    )

    if "valid_origin" not in release_table.columns:
        raise ValueError("Release table must include 'valid_origin'. Use load_release_table().")
    if "quarter" not in release_table.columns:
        raise ValueError("Release table must include 'quarter'.")

    releases = release_table.sort_values("quarter").reset_index(drop=True)
    first_release_date_map = releases.set_index("quarter")["first_release_date"].to_dict()
    truth_maps: dict[str, dict[pd.Period, float]] = {}
    truth_saar_maps: dict[str, dict[pd.Period, float]] = {}
    for stage in stage_list:
        col = RELEASE_STAGE_TO_COL[stage]
        if col not in releases.columns:
            raise ValueError(f"Release table missing required column for stage '{stage}': {col}")
        truth_df = releases.loc[releases[col].notna(), ["quarter", col]].copy()
        truth_maps[stage] = {pd.Period(q, freq="Q-DEC"): float(v) for q, v in zip(truth_df["quarter"], truth_df[col])}

        realtime_col = RELEASE_STAGE_TO_REALTIME_SAAR_COL.get(stage)
        if realtime_col and realtime_col in releases.columns:
            saar_df = releases.loc[releases[realtime_col].notna(), ["quarter", realtime_col]].copy()
            truth_saar_maps[stage] = {
                pd.Period(q, freq="Q-DEC"): float(v) for q, v in zip(saar_df["quarter"], saar_df[realtime_col])
            }
        else:
            if realtime_col:
                print(
                    "WARNING: release table is missing "
                    f"'{realtime_col}'. Falling back to level-based g_true_saar for stage='{stage}', "
                    "which may be inconsistent around reindex/rebenchmark events."
                )
            stage_map = truth_maps[stage]
            truth_saar_maps[stage] = {
                q: to_saar_growth(float(stage_map.get(q, np.nan)), float(stage_map.get(q - 1, np.nan)))
                for q in stage_map
            }

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

        covariate_cutoff = observed_quarter + max_h if ragged_edge_covariates else observed_quarter

        try:
            ds = build_origin_datasets(
                origin_quarter=observed_quarter + 1,
                vintage_id=vintage_id,
                release_table=releases,
                vintage_panel=vintage_panel,
                target_col=target_col,
                train_window=train_window,
                rolling_size=rolling_size,
                cutoff_quarter=covariate_cutoff,
            )
        except ValueError:
            # Skip origins where selected vintage/cutoff does not provide a usable training sample.
            continue

        train_df = ds["train_df"].copy()
        if ragged_edge_covariates:
            future_target_mask = pd.PeriodIndex(train_df["quarter"], freq="Q-DEC") > observed_quarter
            if future_target_mask.any():
                train_df.loc[future_target_mask, target_col] = np.nan
        train_df["__origin_vintage"] = vintage_id
        train_df["__origin_schedule"] = str(origin.origin_schedule)
        train_y = pd.to_numeric(train_df[target_col], errors="coerce")
        obs_mask = train_y.notna()
        if not obs_mask.any():
            continue
        obs_quarters = pd.PeriodIndex(train_df.loc[obs_mask, "quarter"], freq="Q-DEC")
        training_min_quarter = pd.Period(obs_quarters.min(), freq="Q-DEC")
        training_max_quarter = pd.Period(obs_quarters.max(), freq="Q-DEC")
        n_train = int(obs_mask.sum())
        if n_train < int(min_train_observations):
            continue
        last_observed = float(train_y.loc[obs_mask].iloc[-1])

        covariate_cols = select_covariate_columns(train_df=train_df, target_col=target_col)
        past_data, future_data, task_shim = train_df_to_datasets(
            train_df=train_df,
            target_col=target_col,
            horizon=max_h,
            covariate_cols=covariate_cols,
        )
        task_shim.train_df = train_df

        for model in model_list:
            pred_ds = model.predict(past_data=past_data, future_data=future_data, task=task_shim)
            path = np.asarray(pred_ds["predictions"][0], dtype=float)
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
                g_true_first = truth_saar_maps.get("first", {}).get(target_quarter, np.nan)

                for stage in stage_list:
                    stage_map = truth_maps[stage]
                    y_true = stage_map.get(target_quarter, np.nan)
                    y_true_prev = stage_map.get(target_quarter - 1, np.nan)
                    g_true = truth_saar_maps.get(stage, {}).get(target_quarter, np.nan)
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
                            "covariate_cutoff_quarter": str(covariate_cutoff),
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
    "apply_model_runtime_options",
    "build_origin_datasets",
    "build_vintage_calendar",
    "compute_metrics",
    "compute_vintage_asof_date",
    "levels_from_saar_growth",
    "load_release_table",
    "load_vintage_panel",
    "months_to_first_release",
    "months_to_first_release_bucket",
    "saar_growth_series_from_levels",
    "resolve_models",
    "run_backtest",
    "select_training_vintage",
    "to_saar_growth",
    "to_quarter_period",
]
