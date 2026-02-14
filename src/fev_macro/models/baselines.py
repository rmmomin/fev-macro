from __future__ import annotations

import numpy as np
from datasets import Dataset

from .base import BaseModel, get_history_by_item, get_item_order, get_task_horizon, to_prediction_dataset


class NaiveLast(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="naive_last")

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")
            preds[item_id] = np.repeat(values[-1], horizon)

        return to_prediction_dataset(preds, item_order)


class Mean(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="mean")

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")
            preds[item_id] = np.repeat(float(values.mean()), horizon)

        return to_prediction_dataset(preds, item_order)


class Drift(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="drift")

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[object, np.ndarray] = {}
        steps = np.arange(1, horizon + 1, dtype=float)

        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")
            if len(values) == 1:
                preds[item_id] = np.repeat(values[-1], horizon)
                continue

            slope = (values[-1] - values[0]) / float(len(values) - 1)
            preds[item_id] = values[-1] + slope * steps

        return to_prediction_dataset(preds, item_order)


class SeasonalNaive(BaseModel):
    def __init__(self, season_length: int = 4) -> None:
        super().__init__(name="seasonal_naive")
        self.season_length = int(season_length)

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            if len(values) >= self.season_length:
                template = values[-self.season_length :]
            else:
                template = values

            preds[item_id] = np.resize(template, horizon)

        return to_prediction_dataset(preds, item_order)
