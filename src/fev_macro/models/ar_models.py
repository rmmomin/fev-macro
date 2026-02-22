from __future__ import annotations

from typing import Any

import numpy as np
from datasets import Dataset

from .base import (
    BaseModel,
    apply_default_covid_intervention,
    get_history_by_item,
    get_item_order,
    get_task_horizon,
    get_timestamps_by_item,
    to_prediction_dataset,
)


class AR4Model(BaseModel):
    """Univariate AR(4) baseline with naive fallback."""

    def __init__(self) -> None:
        super().__init__(name="ar4")

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)
        past_timestamps = get_timestamps_by_item(past_data, task)
        future_timestamps = get_timestamps_by_item(future_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            y = np.asarray(history.get(item_id, np.array([], dtype=float)), dtype=float)
            y = y[np.isfinite(y)]
            adjusted, future_effect = apply_default_covid_intervention(
                y,
                past_timestamps=past_timestamps.get(item_id),
                future_timestamps=future_timestamps.get(item_id),
                horizon=horizon,
            )
            preds[item_id] = self._forecast_or_fallback(y=adjusted, horizon=horizon) + future_effect

        return to_prediction_dataset(preds, item_order)

    @staticmethod
    def _forecast_or_fallback(y: np.ndarray, horizon: int) -> np.ndarray:
        if y.size == 0:
            return np.zeros(horizon, dtype=float)
        if y.size < 8:
            return np.repeat(float(y[-1]), horizon).astype(float)

        try:
            from statsmodels.tsa.ar_model import AutoReg

            fitted = AutoReg(endog=y, lags=4, old_names=False).fit()
            fcst = np.asarray(fitted.forecast(steps=horizon), dtype=float).reshape(-1)
            if fcst.size != horizon or not np.isfinite(fcst).all():
                raise ValueError("AR4 forecast returned invalid shape/values.")
            return fcst
        except Exception:
            return np.repeat(float(y[-1]), horizon).astype(float)
