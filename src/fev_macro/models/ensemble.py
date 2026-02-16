from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from datasets import Dataset

from .base import BaseModel, get_item_order, to_prediction_dataset
from .baselines import Mean, NaiveLast
from .random_forest import RandomForestModel
from .statsforecast_models import AutoARIMAModel


class SimpleEnsembleModel(BaseModel):
    """Simple averaging ensemble over a fixed set of constituent models."""

    def __init__(
        self,
        name: str,
        members: Sequence[BaseModel],
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__(name=name)
        self.members = list(members)
        if not self.members:
            raise ValueError("SimpleEnsembleModel requires at least one member model")

        if weights is None:
            self.weights = np.repeat(1.0 / len(self.members), len(self.members)).astype(float)
        else:
            w = np.asarray(list(weights), dtype=float)
            if w.size != len(self.members):
                raise ValueError("weights length must match number of members")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            if float(w.sum()) == 0.0:
                raise ValueError("weights sum must be > 0")
            self.weights = w / w.sum()

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        item_order = get_item_order(future_data, task)
        member_preds: list[np.ndarray] = []
        valid_weights: list[float] = []

        for model, w in zip(self.members, self.weights, strict=False):
            try:
                pred_ds = model.predict(past_data=past_data, future_data=future_data, task=task)
                arr = np.asarray(pred_ds["predictions"], dtype=float)
                if arr.ndim != 2 or arr.shape[0] != len(item_order):
                    continue
                if not np.isfinite(arr).all():
                    continue
                member_preds.append(arr)
                valid_weights.append(float(w))
            except Exception:
                continue

        if not member_preds:
            fallback = NaiveLast().predict(past_data=past_data, future_data=future_data, task=task)
            return fallback

        W = np.asarray(valid_weights, dtype=float)
        W = W / W.sum()
        stacked = np.stack(member_preds, axis=0)
        weighted = np.tensordot(W, stacked, axes=(0, 0))

        pred_map = {item_id: weighted[i, :] for i, item_id in enumerate(item_order)}
        return to_prediction_dataset(pred_map, item_order)


class EnsembleAvgTop3Model(SimpleEnsembleModel):
    def __init__(self) -> None:
        members = [
            Mean(),
            RandomForestModel(seed=0),
            AutoARIMAModel(season_length=4),
        ]
        super().__init__(name="ensemble_avg_top3", members=members)
