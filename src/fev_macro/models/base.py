from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset


class BaseModel(ABC):
    """Common interface for all forecasting models in this harness."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__.lower()

    @abstractmethod
    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        """Return forecasts as Dataset with a `predictions` column."""


def dataset_to_pandas(dataset: Dataset) -> pd.DataFrame:
    return dataset.to_pandas().copy()


def get_task_id_column(task: Any) -> str:
    return getattr(task, "id_column", getattr(task, "id_col", "id"))


def get_task_timestamp_column(task: Any) -> str:
    return getattr(task, "timestamp_column", getattr(task, "timestamp_col", "timestamp"))


def get_task_target_column(task: Any) -> str:
    target = getattr(task, "target", getattr(task, "target_col", "target"))
    if isinstance(target, list):
        if not target:
            raise ValueError("Task target list is empty")
        return str(target[0])
    return str(target)


def get_task_horizon(task: Any) -> int:
    if hasattr(task, "horizon"):
        return int(getattr(task, "horizon"))
    return int(getattr(task, "prediction_length"))


def get_item_order(data: Dataset, task: Any) -> list[Any]:
    id_col = get_task_id_column(task)
    if id_col in data.column_names:
        return [row[id_col] for row in data]

    return ["__single_series__"]


def _is_sequence_value(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def get_history_by_item(past_data: Dataset, task: Any) -> dict[Any, np.ndarray]:
    id_col = get_task_id_column(task)
    ts_col = get_task_timestamp_column(task)
    target_col = get_task_target_column(task)

    if len(past_data) == 0:
        return {}

    sample = past_data[0]
    is_sequence = target_col in sample and _is_sequence_value(sample[target_col])

    if is_sequence:
        history: dict[Any, np.ndarray] = {}
        for row in past_data:
            item_id = row[id_col] if id_col in row else "__single_series__"
            values = pd.to_numeric(pd.Series(row[target_col]), errors="coerce").dropna().to_numpy(dtype=float)
            history[item_id] = values
        return history

    pdf = dataset_to_pandas(past_data)
    if id_col not in pdf.columns:
        values = pd.to_numeric(pdf[target_col], errors="coerce").dropna().to_numpy(dtype=float)
        return {"__single_series__": values}

    if ts_col in pdf.columns:
        pdf[ts_col] = pd.to_datetime(pdf[ts_col], errors="coerce")
        pdf = pdf.sort_values([id_col, ts_col])

    history: dict[Any, np.ndarray] = {}
    for item_id, grp in pdf.groupby(id_col, sort=False):
        values = pd.to_numeric(grp[target_col], errors="coerce").dropna().to_numpy(dtype=float)
        history[item_id] = values

    return history


def to_prediction_dataset(predictions_by_item: dict[Any, np.ndarray], item_order: list[Any]) -> Dataset:
    rows: list[list[float]] = []
    for item_id in item_order:
        if item_id not in predictions_by_item:
            raise KeyError(f"Missing predictions for item_id={item_id}")
        arr = np.asarray(predictions_by_item[item_id], dtype=float).reshape(-1)
        rows.append(arr.tolist())

    return Dataset.from_dict({"predictions": rows})
