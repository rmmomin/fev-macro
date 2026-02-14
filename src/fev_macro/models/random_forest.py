from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from .base import (
    BaseModel,
    get_history_by_item,
    get_item_order,
    get_task_horizon,
    get_task_id_column,
    to_prediction_dataset,
)


class RandomForestModel(BaseModel):
    """Random-forest autoregressive forecaster with optional exogenous covariates."""

    def __init__(
        self,
        seed: int = 0,
        n_estimators: int = 120,
        max_lag: int = 8,
        max_depth: int | None = 12,
        max_features: str | float | int = "sqrt",
        min_samples_leaf: int = 1,
    ) -> None:
        super().__init__(name="random_forest")
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.max_lag = int(max_lag)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)

        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for RandomForestModel. Install requirements.txt first."
            ) from exc

        self._rf_cls = RandomForestRegressor

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        id_col = get_task_id_column(task)
        known_covars = [
            c for c in list(getattr(task, "known_dynamic_columns", [])) if c in past_data.column_names or c in future_data.column_names
        ]
        past_covars = [c for c in list(getattr(task, "past_dynamic_columns", [])) if c in past_data.column_names]
        lag_covars = list(dict.fromkeys(past_covars + known_covars))

        past_rows = _rows_by_item(dataset=past_data, id_col=id_col)
        future_rows = _rows_by_item(dataset=future_data, id_col=id_col)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            y = np.asarray(history[item_id], dtype=float)
            if len(y) == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            past_row = past_rows.get(item_id)
            future_row = future_rows.get(item_id)
            if past_row is None:
                raise ValueError(f"Missing past row for item_id={item_id}")

            past_cov = {col: _to_numeric_array(past_row.get(col), expected_len=len(y)) for col in lag_covars}
            future_known_cov = {
                col: _to_numeric_array((future_row or {}).get(col), expected_len=horizon) for col in known_covars
            }

            preds[item_id] = self._forecast_one(
                y=y,
                horizon=horizon,
                lag_covars=lag_covars,
                known_covars=known_covars,
                past_cov=past_cov,
                future_known_cov=future_known_cov,
            )

        return to_prediction_dataset(preds, item_order)

    def _forecast_one(
        self,
        y: np.ndarray,
        horizon: int,
        lag_covars: list[str],
        known_covars: list[str],
        past_cov: dict[str, np.ndarray],
        future_known_cov: dict[str, np.ndarray],
    ) -> np.ndarray:
        if y.size < 4:
            return np.repeat(float(y[-1]), horizon)

        lag = min(self.max_lag, max(2, y.size - 2))

        X, y_train = _build_feature_matrix(
            y=y,
            lag=lag,
            lag_covars=lag_covars,
            known_covars=known_covars,
            past_cov=past_cov,
        )

        if y_train.size < max(8, lag + 2):
            return _drift_fallback(y, horizon=horizon)

        model = self._rf_cls(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
        )
        model.fit(X, y_train)

        y_hist = list(y.astype(float))
        cov_extended = _extend_covariates(
            lag_covars=lag_covars,
            known_covars=known_covars,
            past_cov=past_cov,
            future_known_cov=future_known_cov,
            horizon=horizon,
        )

        forecasts: list[float] = []
        for step in range(horizon):
            t_idx = len(y) + step
            feat = _build_single_feature(
                y_hist=y_hist,
                t_idx=t_idx,
                lag=lag,
                lag_covars=lag_covars,
                known_covars=known_covars,
                cov_series=cov_extended,
            )
            pred = float(model.predict(feat.reshape(1, -1))[0])
            if not np.isfinite(pred):
                pred = float(y_hist[-1])

            forecasts.append(pred)
            y_hist.append(pred)

        return np.asarray(forecasts, dtype=float)


def _rows_by_item(dataset: Dataset, id_col: str) -> dict[Any, dict[str, Any]]:
    rows: dict[Any, dict[str, Any]] = {}
    for rec in dataset:
        key = rec[id_col] if id_col in rec else "__single_series__"
        rows[key] = rec
    return rows


def _to_numeric_array(value: Any, expected_len: int) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        s = pd.to_numeric(pd.Series(value), errors="coerce")
    elif value is None:
        s = pd.Series([np.nan] * expected_len, dtype=float)
    else:
        s = pd.to_numeric(pd.Series([value] * expected_len), errors="coerce")

    if len(s) < expected_len:
        if len(s) == 0:
            s = pd.Series([np.nan] * expected_len, dtype=float)
        else:
            s = pd.concat([s, pd.Series([s.iloc[-1]] * (expected_len - len(s)), dtype=float)], ignore_index=True)
    elif len(s) > expected_len:
        s = s.iloc[:expected_len]

    s = s.astype(float).ffill().bfill().fillna(0.0)
    return s.to_numpy(dtype=float)


def _build_feature_matrix(
    y: np.ndarray,
    lag: int,
    lag_covars: list[str],
    known_covars: list[str],
    past_cov: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if y.size <= lag:
        return np.empty((0, 0), dtype=float), np.empty(0, dtype=float)

    rows: list[list[float]] = []
    targets: list[float] = []

    for t in range(lag, y.size):
        feat: list[float] = []
        feat.extend(y[t - lag : t].tolist())

        for cov in lag_covars:
            feat.extend(past_cov[cov][t - lag : t].tolist())

        for cov in known_covars:
            feat.append(float(past_cov[cov][t]))

        rows.append(feat)
        targets.append(float(y[t]))

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _extend_covariates(
    lag_covars: list[str],
    known_covars: list[str],
    past_cov: dict[str, np.ndarray],
    future_known_cov: dict[str, np.ndarray],
    horizon: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cov in lag_covars:
        past_vals = np.asarray(past_cov[cov], dtype=float)
        if cov in known_covars:
            future_vals = np.asarray(future_known_cov.get(cov, np.repeat(past_vals[-1], horizon)), dtype=float)
            if future_vals.size < horizon:
                future_vals = np.pad(future_vals, (0, horizon - future_vals.size), mode="edge")
            out[cov] = np.concatenate([past_vals, future_vals[:horizon]])
        else:
            out[cov] = past_vals
    return out


def _build_single_feature(
    y_hist: list[float],
    t_idx: int,
    lag: int,
    lag_covars: list[str],
    known_covars: list[str],
    cov_series: dict[str, np.ndarray],
) -> np.ndarray:
    feat: list[float] = []

    y_arr = np.asarray(y_hist, dtype=float)
    y_window = y_arr[-lag:]
    if y_window.size < lag:
        y_window = np.pad(y_window, (lag - y_window.size, 0), mode="edge")
    feat.extend(y_window.tolist())

    for cov in lag_covars:
        c_arr = np.asarray(cov_series[cov], dtype=float)
        start = max(0, t_idx - lag)
        end = max(0, t_idx)
        window = c_arr[start:end]
        if window.size < lag:
            pad_value = c_arr[0] if c_arr.size else 0.0
            window = np.pad(window, (lag - window.size, 0), mode="constant", constant_values=pad_value)
        feat.extend(window.tolist())

    for cov in known_covars:
        c_arr = np.asarray(cov_series[cov], dtype=float)
        if c_arr.size == 0:
            feat.append(0.0)
        elif t_idx < c_arr.size:
            feat.append(float(c_arr[t_idx]))
        else:
            feat.append(float(c_arr[-1]))

    return np.asarray(feat, dtype=float)


def _drift_fallback(y: np.ndarray, horizon: int) -> np.ndarray:
    if y.size <= 1:
        return np.repeat(float(y[-1]), horizon)

    slope = (float(y[-1]) - float(y[0])) / float(y.size - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    return float(y[-1]) + slope * steps
