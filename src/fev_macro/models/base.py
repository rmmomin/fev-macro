from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
import pandas as pd
from datasets import Dataset

from ..data import COVID_DUMMY_COLUMNS, COVID_DUMMY_PERIOD_BY_COLUMN


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


def get_timestamps_by_item(data: Dataset, task: Any) -> dict[Any, np.ndarray]:
    id_col = get_task_id_column(task)
    ts_col = get_task_timestamp_column(task)

    if len(data) == 0:
        return {}

    sample = data[0]
    is_sequence = ts_col in sample and _is_sequence_value(sample[ts_col])

    if is_sequence:
        timestamps: dict[Any, np.ndarray] = {}
        for row in data:
            item_id = row[id_col] if id_col in row else "__single_series__"
            ts = pd.to_datetime(pd.Series(row.get(ts_col, [])), errors="coerce").dropna()
            timestamps[item_id] = ts.to_numpy(dtype="datetime64[ns]")
        return timestamps

    pdf = dataset_to_pandas(data)
    if ts_col not in pdf.columns:
        return {}

    pdf[ts_col] = pd.to_datetime(pdf[ts_col], errors="coerce")
    pdf = pdf.dropna(subset=[ts_col])
    if pdf.empty:
        return {}

    if id_col not in pdf.columns:
        values = pdf.sort_values(ts_col)[ts_col].to_numpy(dtype="datetime64[ns]")
        return {"__single_series__": values}

    pdf = pdf.sort_values([id_col, ts_col])
    timestamps: dict[Any, np.ndarray] = {}
    for item_id, grp in pdf.groupby(id_col, sort=False):
        timestamps[item_id] = grp[ts_col].to_numpy(dtype="datetime64[ns]")
    return timestamps


def apply_default_covid_intervention(
    y: np.ndarray,
    *,
    past_timestamps: Sequence[Any] | np.ndarray | None,
    future_timestamps: Sequence[Any] | np.ndarray | None,
    horizon: int,
    covid_columns: Sequence[str] = COVID_DUMMY_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    h = int(horizon)
    if y_arr.size == 0:
        return y_arr, np.zeros(h, dtype=float)

    cols = [c for c in covid_columns if c in COVID_DUMMY_PERIOD_BY_COLUMN]
    if not cols:
        return y_arr, np.zeros(h, dtype=float)

    past_design = _covid_design_from_timestamps(past_timestamps, cols=cols)
    future_design = _covid_design_from_timestamps(future_timestamps, cols=cols)

    past_design = _align_design_rows(past_design, expected_rows=y_arr.size, align="tail")
    future_design = _align_design_rows(future_design, expected_rows=h, align="head")

    return _estimate_intervention_effect(y=y_arr, past_design=past_design, future_design=future_design)


def _coerce_timestamps(values: Sequence[Any] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype="datetime64[ns]")

    ts = pd.to_datetime(pd.Series(values), errors="coerce").dropna()
    if ts.empty:
        return np.empty(0, dtype="datetime64[ns]")
    return ts.to_numpy(dtype="datetime64[ns]")


def _covid_design_from_timestamps(
    timestamps: Sequence[Any] | np.ndarray | None,
    *,
    cols: Sequence[str],
) -> np.ndarray:
    ts = _coerce_timestamps(timestamps)
    if len(ts) == 0:
        return np.zeros((0, len(cols)), dtype=float)

    q = pd.PeriodIndex(ts, freq="Q-DEC")
    design = np.zeros((len(q), len(cols)), dtype=float)
    for j, col in enumerate(cols):
        target_period = COVID_DUMMY_PERIOD_BY_COLUMN[col]
        design[:, j] = (q == target_period).astype(float)
    return design


def _align_design_rows(design: np.ndarray, *, expected_rows: int, align: str) -> np.ndarray:
    n = int(expected_rows)
    k = int(design.shape[1]) if design.ndim == 2 else 0
    if n <= 0:
        return np.zeros((0, k), dtype=float)

    if k == 0:
        return np.zeros((n, 0), dtype=float)

    if design.shape[0] == n:
        return design

    if design.shape[0] > n:
        if align == "tail":
            return design[-n:, :]
        return design[:n, :]

    pad = np.zeros((n - design.shape[0], k), dtype=float)
    if align == "tail":
        return np.vstack([pad, design])
    return np.vstack([design, pad])


def _estimate_intervention_effect(
    *,
    y: np.ndarray,
    past_design: np.ndarray,
    future_design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if y.size == 0:
        return y, np.zeros(future_design.shape[0], dtype=float)

    if past_design.shape[1] == 0:
        return y, np.zeros(future_design.shape[0], dtype=float)

    X = np.nan_to_num(past_design.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    X_future = np.nan_to_num(future_design.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    active = np.where(np.any(np.abs(X) > 0.0, axis=0))[0]
    if active.size == 0:
        return y, np.zeros(future_design.shape[0], dtype=float)

    X = X[:, active]
    X_future = X_future[:, active] if X_future.shape[1] else np.zeros((future_design.shape[0], active.size), dtype=float)

    design = np.column_stack([np.ones(y.size, dtype=float), X])
    mask = np.isfinite(y) & np.isfinite(design).all(axis=1)
    if int(mask.sum()) < max(8, active.size + 2):
        return y, np.zeros(future_design.shape[0], dtype=float)

    try:
        coefs = np.linalg.lstsq(design[mask], y[mask], rcond=None)[0]
    except np.linalg.LinAlgError:
        return y, np.zeros(future_design.shape[0], dtype=float)

    beta = np.asarray(coefs[1:], dtype=float).reshape(-1)
    if beta.size == 0:
        return y, np.zeros(future_design.shape[0], dtype=float)

    past_effect = X @ beta
    future_effect = X_future @ beta if X_future.size else np.zeros(future_design.shape[0], dtype=float)
    adjusted = y - past_effect

    if not np.isfinite(adjusted).all():
        return y, np.zeros(future_design.shape[0], dtype=float)

    return adjusted.astype(float), np.asarray(future_effect, dtype=float).reshape(-1)


def to_prediction_dataset(predictions_by_item: dict[Any, np.ndarray], item_order: list[Any]) -> Dataset:
    rows: list[list[float]] = []
    for item_id in item_order:
        if item_id not in predictions_by_item:
            raise KeyError(f"Missing predictions for item_id={item_id}")
        arr = np.asarray(predictions_by_item[item_id], dtype=float).reshape(-1)
        rows.append(arr.tolist())

    return Dataset.from_dict({"predictions": rows})
