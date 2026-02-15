from __future__ import annotations

from datasets import Dataset
import numpy as np

from .base import BaseModel, get_history_by_item, get_item_order, get_task_horizon, get_task_id_column, to_prediction_dataset
from .random_forest import (
    _build_feature_matrix,
    _build_single_feature,
    _drift_fallback,
    _extend_covariates,
    _rows_by_item,
    _to_numeric_array,
)


class XGBoostModel(BaseModel):
    """Gradient-boosted autoregressive forecaster with optional exogenous covariates."""

    def __init__(
        self,
        seed: int = 0,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        max_lag: int = 8,
    ) -> None:
        super().__init__(name="xgboost")
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.max_lag = int(max_lag)

        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostModel. Install requirements.txt first.") from exc

        self._xgb_cls = XGBRegressor

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
            if y.size == 0:
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
        if y.size < 8:
            return _drift_fallback(y, horizon=horizon)

        lag = min(self.max_lag, max(2, y.size - 2))
        X, y_train = _build_feature_matrix(
            y=y,
            lag=lag,
            lag_covars=lag_covars,
            known_covars=known_covars,
            past_cov=past_cov,
        )

        if y_train.size < max(12, lag + 4):
            return _drift_fallback(y, horizon=horizon)

        model = self._xgb_cls(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            n_jobs=-1,
            tree_method="hist",
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
