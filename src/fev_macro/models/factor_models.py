from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from .base import BaseModel, get_history_by_item, get_item_order, get_task_horizon, get_task_id_column, to_prediction_dataset
from .random_forest import _drift_fallback, _rows_by_item, _to_numeric_array


PREFERRED_FACTOR_COVARIATES: tuple[str, ...] = (
    "UNRATE",
    "FEDFUNDS",
    "CPIAUCSL",
    "INDPRO",
    "PAYEMS",
    "HOUST",
    "GS10",
    "M2REAL",
    "PCECC96",
    "IPFINAL",
    "TB3MS",
    "MORTGAGE30US",
    "SPCS20RSA",
    "WPSFD49207",
    "USSTHPI",
    "EXUSEU",
    "OILPRICEx",
)


class QuarterlyFactorPCAModel(BaseModel):
    """Quarterly factor model: PCA factors from FRED-QD covariates + autoregressive head."""

    def __init__(
        self,
        max_covariates: int = 80,
        n_factors: int = 6,
        max_lag: int = 4,
        alpha: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(name="factor_pca_qd")
        self.max_covariates = int(max_covariates)
        self.n_factors = int(n_factors)
        self.max_lag = int(max_lag)
        self.alpha = float(alpha)
        self.seed = int(seed)

        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError("scikit-learn is required for QuarterlyFactorPCAModel.") from exc

        self._pca_cls = PCA
        self._ridge_cls = Ridge
        self._scaler_cls = StandardScaler

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        id_col = get_task_id_column(task)
        known_covars = [
            c for c in list(getattr(task, "known_dynamic_columns", [])) if c in past_data.column_names or c in future_data.column_names
        ]
        past_covars = [c for c in list(getattr(task, "past_dynamic_columns", [])) if c in past_data.column_names]
        lag_covars = list(dict.fromkeys(past_covars + known_covars))

        selected_covars = _select_covariates(
            available=lag_covars,
            preferred=PREFERRED_FACTOR_COVARIATES,
            max_covariates=self.max_covariates,
        )

        past_rows = _rows_by_item(dataset=past_data, id_col=id_col)
        future_rows = _rows_by_item(dataset=future_data, id_col=id_col)

        preds: dict[Any, np.ndarray] = {}
        for item_id in item_order:
            y = np.asarray(history[item_id], dtype=float)
            if y.size == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            past_row = past_rows.get(item_id)
            future_row = future_rows.get(item_id)
            if past_row is None:
                raise ValueError(f"Missing past row for item_id={item_id}")

            past_cov = {col: _to_numeric_array(past_row.get(col), expected_len=len(y)) for col in selected_covars}
            future_known_cov = {
                col: _to_numeric_array((future_row or {}).get(col), expected_len=horizon)
                for col in selected_covars
                if col in known_covars
            }

            preds[item_id] = self._forecast_one(
                y=y,
                selected_covars=selected_covars,
                past_cov=past_cov,
                future_known_cov=future_known_cov,
                horizon=horizon,
            )

        return to_prediction_dataset(preds, item_order)

    def _forecast_one(
        self,
        y: np.ndarray,
        selected_covars: list[str],
        past_cov: dict[str, np.ndarray],
        future_known_cov: dict[str, np.ndarray],
        horizon: int,
    ) -> np.ndarray:
        if y.size < 16 or not selected_covars:
            return _drift_fallback(y, horizon=horizon)

        lag = min(self.max_lag, max(2, y.size // 8))
        if y.size <= lag + 4:
            return _drift_fallback(y, horizon=horizon)

        cov_ext = _extend_covariates(selected_covars, past_cov, future_known_cov, horizon=horizon)
        X_cov_past = np.column_stack([cov_ext[c][: y.size] for c in selected_covars]).astype(float)
        X_cov_future = np.column_stack([cov_ext[c][y.size : y.size + horizon] for c in selected_covars]).astype(float)

        scaler = self._scaler_cls(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_cov_past)

        n_factors = max(1, min(self.n_factors, X_scaled.shape[1], X_scaled.shape[0] - 1))
        pca = self._pca_cls(n_components=n_factors, random_state=self.seed)
        F_past = pca.fit_transform(X_scaled)
        F_future = pca.transform(scaler.transform(X_cov_future))

        rows: list[np.ndarray] = []
        targets: list[float] = []
        for t in range(lag, y.size):
            feat = np.concatenate([y[t - lag : t], F_past[t]], axis=0)
            rows.append(feat)
            targets.append(float(y[t]))

        X_train = np.vstack(rows)
        y_train = np.asarray(targets, dtype=float)
        if y_train.size < max(10, lag + 3):
            return _drift_fallback(y, horizon=horizon)

        reg = self._ridge_cls(alpha=self.alpha, random_state=self.seed)
        reg.fit(X_train, y_train)

        y_hist = list(y.astype(float))
        forecasts: list[float] = []
        for step in range(horizon):
            lag_vals = np.asarray(y_hist[-lag:], dtype=float)
            if lag_vals.size < lag:
                lag_vals = np.pad(lag_vals, (lag - lag_vals.size, 0), mode="edge")

            feat = np.concatenate([lag_vals, F_future[step]], axis=0)
            pred = float(reg.predict(feat.reshape(1, -1))[0])
            if not np.isfinite(pred):
                pred = float(y_hist[-1])
            forecasts.append(pred)
            y_hist.append(pred)

        return np.asarray(forecasts, dtype=float)


class MixedFrequencyDFMModel(BaseModel):
    """Mixed-frequency factor model using local vintage-panel covariates + GDP target."""

    def __init__(
        self,
        md_dataset_path: str = "data/panels/fred_md_vintage_panel_process.parquet",
        md_config: str = "local_qd_panel",
        max_monthly_covariates: int = 100,
        n_factors: int = 6,
        max_lag: int = 2,
        alpha: float = 1.0,
        excluded_years: tuple[int, ...] = (),
        seed: int = 0,
    ) -> None:
        super().__init__(name="mixed_freq_dfm_md")
        self.md_dataset_path = md_dataset_path
        self.md_config = md_config
        self.max_monthly_covariates = int(max_monthly_covariates)
        self.n_factors = int(n_factors)
        self.max_lag = int(max_lag)
        self.alpha = float(alpha)
        self.excluded_years = tuple(sorted({int(y) for y in excluded_years}))
        self.seed = int(seed)

        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError("scikit-learn is required for MixedFrequencyDFMModel.") from exc

        self._pca_cls = PCA
        self._ridge_cls = Ridge
        self._scaler_cls = StandardScaler

        self._quarter_panel = _load_fred_md_quarter_panel(
            dataset_path=self.md_dataset_path,
            config=self.md_config,
            max_covariates=self.max_monthly_covariates,
            excluded_years=self.excluded_years,
        )

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[Any, np.ndarray] = {}
        for item_id in item_order:
            y = np.asarray(history[item_id], dtype=float)
            if y.size == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            preds[item_id] = self._forecast_one(y=y, horizon=horizon)

        return to_prediction_dataset(preds, item_order)

    def _forecast_one(self, y: np.ndarray, horizon: int) -> np.ndarray:
        if y.size < 20:
            return _drift_fallback(y, horizon=horizon)

        F = _slice_quarter_factors(self._quarter_panel, required_length=y.size + horizon)
        F_past = F[: y.size]
        F_future = F[y.size : y.size + horizon]

        lag = min(self.max_lag, max(1, y.size // 12))
        if y.size <= lag + 4:
            return _drift_fallback(y, horizon=horizon)

        scaler = self._scaler_cls(with_mean=True, with_std=True)
        F_scaled = scaler.fit_transform(F_past)

        n_factors = max(1, min(self.n_factors, F_scaled.shape[1], F_scaled.shape[0] - 1))
        pca = self._pca_cls(n_components=n_factors, random_state=self.seed)
        Z_past = pca.fit_transform(F_scaled)
        Z_future = pca.transform(scaler.transform(F_future))

        rows: list[np.ndarray] = []
        targets: list[float] = []
        for t in range(lag, y.size):
            feat = np.concatenate([y[t - lag : t], Z_past[t]], axis=0)
            rows.append(feat)
            targets.append(float(y[t]))

        X_train = np.vstack(rows)
        y_train = np.asarray(targets, dtype=float)
        if y_train.size < max(10, lag + 3):
            return _drift_fallback(y, horizon=horizon)

        reg = self._ridge_cls(alpha=self.alpha, random_state=self.seed)
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

        return np.asarray(forecasts, dtype=float)


def _select_covariates(available: list[str], preferred: tuple[str, ...], max_covariates: int) -> list[str]:
    ordered: list[str] = []
    avail_set = set(available)

    for c in preferred:
        if c in avail_set and c not in ordered:
            ordered.append(c)
        if len(ordered) >= max_covariates:
            return ordered

    for c in available:
        if c not in ordered:
            ordered.append(c)
        if len(ordered) >= max_covariates:
            break

    return ordered


def _extend_covariates(
    selected_covars: list[str],
    past_cov: dict[str, np.ndarray],
    future_known_cov: dict[str, np.ndarray],
    horizon: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cov in selected_covars:
        past_vals = np.asarray(past_cov[cov], dtype=float)
        fut = future_known_cov.get(cov)
        if fut is None:
            fut_vals = np.repeat(past_vals[-1], horizon)
        else:
            fut_vals = np.asarray(fut, dtype=float)
            if fut_vals.size < horizon:
                fut_vals = np.pad(fut_vals, (0, horizon - fut_vals.size), mode="edge")
        out[cov] = np.concatenate([past_vals, fut_vals[:horizon]])
    return out


@lru_cache(maxsize=4)
def _load_fred_md_quarter_panel(
    dataset_path: str,
    config: str,
    max_covariates: int,
    excluded_years: tuple[int, ...],
) -> np.ndarray:
    _ = config
    panel_path = Path(dataset_path).expanduser().resolve()
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Local vintage panel not found for mixed_freq_dfm_md: {panel_path}. "
            "Set md_dataset_path to a local parquet panel."
        )

    panel = pd.read_parquet(panel_path)
    required_cols = {"timestamp", "vintage"}
    missing = sorted(required_cols.difference(panel.columns))
    if missing:
        raise ValueError(f"Vintage panel missing required columns {missing}: {panel_path}")

    work = panel.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["vintage"] = work["vintage"].astype(str)
    work = work.dropna(subset=["timestamp", "vintage"]).sort_values(["vintage", "timestamp"])
    if work.empty:
        return np.zeros((1, 1), dtype=float)

    latest_vintage = sorted(work["vintage"].unique())[-1]
    latest = work.loc[work["vintage"] == latest_vintage].copy()

    drop_cols = {"timestamp", "vintage", "vintage_timestamp", "GDPC1"}
    candidate_cols = [c for c in latest.columns if c not in drop_cols]

    numeric_cols: list[str] = []
    for col in candidate_cols:
        values = pd.to_numeric(latest[col], errors="coerce")
        if values.notna().any():
            latest[col] = values.astype(float)
            numeric_cols.append(col)

    if not numeric_cols:
        return np.zeros((1, 1), dtype=float)

    numeric_cols = numeric_cols[:max_covariates]

    quarter = latest[["timestamp", *numeric_cols]].copy()
    quarter["quarter"] = quarter["timestamp"].dt.to_period("Q-DEC").dt.to_timestamp(how="S")
    q = quarter.groupby("quarter", sort=True)[numeric_cols].mean(numeric_only=True)

    if excluded_years:
        q = q.loc[~q.index.year.isin(list(excluded_years))]

    if q.empty:
        return np.zeros((1, max(1, len(numeric_cols))), dtype=float)

    q = q.ffill().bfill().fillna(0.0)
    return q.to_numpy(dtype=float)


def _slice_quarter_factors(panel: np.ndarray, required_length: int) -> np.ndarray:
    if panel.shape[0] >= required_length:
        return panel[:required_length]

    if panel.shape[0] == 0:
        return np.zeros((required_length, 1), dtype=float)

    missing = required_length - panel.shape[0]
    tail = np.repeat(panel[-1:, :], missing, axis=0)
    return np.vstack([panel, tail])
