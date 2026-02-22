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


class RandomNormal(BaseModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_normal")
        self.rng = np.random.default_rng(seed)

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

            mu = float(np.mean(adjusted))
            sigma = float(np.std(adjusted, ddof=1)) if len(adjusted) > 1 else 0.0
            if not np.isfinite(sigma) or sigma <= 0:
                preds[item_id] = np.repeat(mu, horizon) + future_effect
            else:
                preds[item_id] = self.rng.normal(loc=mu, scale=sigma, size=horizon) + future_effect

        return to_prediction_dataset(preds, item_order)


class RandomUniform(BaseModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_uniform")
        self.rng = np.random.default_rng(seed)

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

            lo = float(np.min(adjusted))
            hi = float(np.max(adjusted))
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"History contains non-finite values for item_id={item_id}")

            if hi == lo:
                preds[item_id] = np.repeat(lo, horizon) + future_effect
            else:
                preds[item_id] = self.rng.uniform(low=lo, high=hi, size=horizon) + future_effect

        return to_prediction_dataset(preds, item_order)


class RandomPermutation(BaseModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_permutation")
        self.rng = np.random.default_rng(seed)

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

            perm = self.rng.permutation(adjusted)
            base = perm[:horizon] if len(perm) >= horizon else np.resize(perm, horizon)
            preds[item_id] = base + future_effect

        return to_prediction_dataset(preds, item_order)
