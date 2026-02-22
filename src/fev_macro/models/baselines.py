from __future__ import annotations

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


class NaiveLast(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="naive_last")

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)
        past_timestamps = get_timestamps_by_item(past_data, task)
        future_timestamps = get_timestamps_by_item(future_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")
            adjusted, future_effect = apply_default_covid_intervention(
                values,
                past_timestamps=past_timestamps.get(item_id),
                future_timestamps=future_timestamps.get(item_id),
                horizon=horizon,
            )
            preds[item_id] = np.repeat(adjusted[-1], horizon) + future_effect

        return to_prediction_dataset(preds, item_order)


class Mean(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="mean")

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)
        past_timestamps = get_timestamps_by_item(past_data, task)
        future_timestamps = get_timestamps_by_item(future_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")
            adjusted, future_effect = apply_default_covid_intervention(
                values,
                past_timestamps=past_timestamps.get(item_id),
                future_timestamps=future_timestamps.get(item_id),
                horizon=horizon,
            )
            preds[item_id] = np.repeat(float(adjusted.mean()), horizon) + future_effect

        return to_prediction_dataset(preds, item_order)


class Drift(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="drift")

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)
        past_timestamps = get_timestamps_by_item(past_data, task)
        future_timestamps = get_timestamps_by_item(future_data, task)

        preds: dict[object, np.ndarray] = {}
        steps = np.arange(1, horizon + 1, dtype=float)

        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")
            adjusted, future_effect = apply_default_covid_intervention(
                values,
                past_timestamps=past_timestamps.get(item_id),
                future_timestamps=future_timestamps.get(item_id),
                horizon=horizon,
            )
            if len(adjusted) == 1:
                preds[item_id] = np.repeat(adjusted[-1], horizon) + future_effect
                continue

            slope = (adjusted[-1] - adjusted[0]) / float(len(adjusted) - 1)
            preds[item_id] = adjusted[-1] + slope * steps + future_effect

        return to_prediction_dataset(preds, item_order)


class SeasonalNaive(BaseModel):
    def __init__(self, season_length: int = 4) -> None:
        super().__init__(name="seasonal_naive")
        self.season_length = int(season_length)

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)
        past_timestamps = get_timestamps_by_item(past_data, task)
        future_timestamps = get_timestamps_by_item(future_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")
            adjusted, future_effect = apply_default_covid_intervention(
                values,
                past_timestamps=past_timestamps.get(item_id),
                future_timestamps=future_timestamps.get(item_id),
                horizon=horizon,
            )

            if len(adjusted) >= self.season_length:
                template = adjusted[-self.season_length :]
            else:
                template = adjusted

            preds[item_id] = np.resize(template, horizon) + future_effect

        return to_prediction_dataset(preds, item_order)
