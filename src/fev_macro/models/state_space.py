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
from .random_forest import _drift_fallback


class LocalTrendStateSpaceModel(BaseModel):
    """Local-trend state-space baseline (optionally with cycle) on log GDP."""

    def __init__(
        self,
        include_cycle: bool = True,
        maxiter: int = 200,
    ) -> None:
        super().__init__(name="local_trend_ssm")
        self.include_cycle = bool(include_cycle)
        self.maxiter = int(maxiter)

        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError as exc:
            raise ImportError("statsmodels is required for LocalTrendStateSpaceModel.") from exc

        self._uc_cls = UnobservedComponents

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)
        past_timestamps = get_timestamps_by_item(past_data, task)
        future_timestamps = get_timestamps_by_item(future_data, task)

        preds: dict[Any, np.ndarray] = {}
        for item_id in item_order:
            y = np.asarray(history[item_id], dtype=float)
            if y.size == 0:
                raise ValueError(f"No history available for item_id={item_id}")
            adjusted, future_effect = apply_default_covid_intervention(
                y,
                past_timestamps=past_timestamps.get(item_id),
                future_timestamps=future_timestamps.get(item_id),
                horizon=horizon,
            )
            preds[item_id] = self._forecast_one(y=adjusted, horizon=horizon) + future_effect

        return to_prediction_dataset(preds, item_order)

    def _forecast_one(self, y: np.ndarray, horizon: int) -> np.ndarray:
        if y.size < 12:
            return _drift_fallback(y, horizon=horizon)

        try:
            kwargs: dict[str, Any] = {"level": "local linear trend"}
            if self.include_cycle and y.size >= 24:
                kwargs.update(
                    {
                        "cycle": True,
                        "stochastic_cycle": True,
                        "damped_cycle": True,
                    }
                )

            model = self._uc_cls(y, **kwargs)
            result = model.fit(disp=False, maxiter=self.maxiter)
            fc = np.asarray(result.forecast(horizon), dtype=float).reshape(-1)
            if fc.size != horizon or not np.isfinite(fc).all():
                return _drift_fallback(y, horizon=horizon)
            return fc
        except Exception:
            return _drift_fallback(y, horizon=horizon)
