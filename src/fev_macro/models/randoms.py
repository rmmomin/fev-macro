from __future__ import annotations

import numpy as np
from datasets import Dataset

from .base import BaseModel, get_history_by_item, get_item_order, get_task_horizon, to_prediction_dataset


class RandomNormal(BaseModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_normal")
        self.rng = np.random.default_rng(seed)

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            mu = float(np.mean(values))
            sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if not np.isfinite(sigma) or sigma <= 0:
                preds[item_id] = np.repeat(mu, horizon)
            else:
                preds[item_id] = self.rng.normal(loc=mu, scale=sigma, size=horizon)

        return to_prediction_dataset(preds, item_order)


class RandomUniform(BaseModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_uniform")
        self.rng = np.random.default_rng(seed)

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            lo = float(np.min(values))
            hi = float(np.max(values))
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"History contains non-finite values for item_id={item_id}")

            if hi == lo:
                preds[item_id] = np.repeat(lo, horizon)
            else:
                preds[item_id] = self.rng.uniform(low=lo, high=hi, size=horizon)

        return to_prediction_dataset(preds, item_order)


class RandomPermutation(BaseModel):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random_permutation")
        self.rng = np.random.default_rng(seed)

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[object, np.ndarray] = {}
        for item_id in item_order:
            values = history[item_id]
            if len(values) == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            perm = self.rng.permutation(values)
            preds[item_id] = perm[:horizon] if len(perm) >= horizon else np.resize(perm, horizon)

        return to_prediction_dataset(preds, item_order)
