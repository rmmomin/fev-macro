from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from .models.base import BaseModel, get_task_horizon, get_task_id_column, get_task_timestamp_column


def run_models_on_task(task: Any, models: dict[str, BaseModel], num_windows: int = 80) -> list[dict[str, Any]]:
    """Run each model on one task and return fev evaluation summaries."""
    _ = num_windows
    summaries: list[dict[str, Any]] = []

    for model_name, model in models.items():
        predictions_per_window: list[Dataset] = []
        inference_time_s = 0.0

        for window_idx, window in enumerate(task.iter_windows()):
            input_data = window.get_input_data()
            if isinstance(input_data, tuple) and len(input_data) == 2:
                past_data, future_data = input_data
            else:
                past_data = input_data
                future_data = window.get_future_data()

            start = time.perf_counter()
            raw_predictions = model.predict(past_data=past_data, future_data=future_data, task=task)
            inference_time_s += time.perf_counter() - start

            predictions = _coerce_prediction_dataset(raw_predictions)
            expected_items = _expected_item_count(
                future_data=future_data,
                id_col=get_task_id_column(task),
                timestamp_col=get_task_timestamp_column(task),
            )
            _validate_predictions(
                predictions=predictions,
                horizon=get_task_horizon(task),
                expected_items=expected_items,
                model_name=model_name,
                task_name=task.task_name,
                window_idx=window_idx,
            )
            predictions_per_window.append(predictions)

        summary = task.evaluation_summary(
            predictions_per_window=predictions_per_window,
            model_name=model_name,
            training_time_s=0.0,
            inference_time_s=float(inference_time_s),
        )
        summaries.append(summary)

    return summaries


def run_models_on_tasks(
    tasks: list[Any],
    models: dict[str, BaseModel],
    num_windows: int = 80,
) -> list[dict[str, Any]]:
    all_summaries: list[dict[str, Any]] = []

    for task in tasks:
        task_summaries = run_models_on_task(task=task, models=models, num_windows=num_windows)
        all_summaries.extend(task_summaries)

    return all_summaries


def save_summaries(
    summaries: list[dict[str, Any]],
    jsonl_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    jsonl_path = Path(jsonl_path)
    csv_path = Path(csv_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, default=_json_default) + "\n")

    pd.DataFrame(summaries).to_csv(csv_path, index=False)
    return jsonl_path, csv_path


def _coerce_prediction_dataset(raw_predictions: Any) -> Dataset:
    if isinstance(raw_predictions, Dataset):
        return raw_predictions

    if isinstance(raw_predictions, dict):
        if "predictions" not in raw_predictions:
            raise ValueError("Prediction dict must include a 'predictions' key")

        preds = raw_predictions["predictions"]
        rows = _normalize_prediction_rows(preds)
        return Dataset.from_dict({"predictions": rows})

    raise TypeError(
        "Model.predict must return datasets.Dataset or a dict with {'predictions': ...}; "
        f"got {type(raw_predictions)}"
    )


def _normalize_prediction_rows(predictions: Any) -> list[list[float]]:
    if isinstance(predictions, np.ndarray):
        if predictions.ndim == 1:
            return [predictions.astype(float).tolist()]
        if predictions.ndim == 2:
            return [row.astype(float).tolist() for row in predictions]
        raise ValueError(f"Unexpected prediction ndarray ndim={predictions.ndim}")

    if isinstance(predictions, list):
        if len(predictions) == 0:
            return []

        first = predictions[0]
        if np.isscalar(first):
            return [np.asarray(predictions, dtype=float).reshape(-1).tolist()]

        rows: list[list[float]] = []
        for row in predictions:
            rows.append(np.asarray(row, dtype=float).reshape(-1).tolist())
        return rows

    raise TypeError(f"Unsupported prediction type: {type(predictions)}")


def _expected_item_count(future_data: Dataset, id_col: str, timestamp_col: str) -> int:
    if len(future_data) == 0:
        return 0

    sample = future_data[0]
    if id_col in future_data.column_names and isinstance(sample.get(timestamp_col), (list, tuple)):
        return len(future_data)

    pdf = future_data.to_pandas()
    if id_col not in pdf.columns:
        return 1
    return int(pd.Series(pdf[id_col]).nunique())


def _validate_predictions(
    predictions: Dataset,
    horizon: int,
    expected_items: int,
    model_name: str,
    task_name: str,
    window_idx: int,
) -> None:
    if "predictions" not in predictions.column_names:
        raise ValueError(
            f"Model={model_name} task={task_name} window={window_idx}: missing 'predictions' column"
        )

    rows = predictions["predictions"]

    if len(rows) != expected_items:
        raise ValueError(
            f"Model={model_name} task={task_name} window={window_idx}: "
            f"expected {expected_items} item forecasts, got {len(rows)}"
        )

    for item_idx, row in enumerate(rows):
        arr = np.asarray(row, dtype=float).reshape(-1)
        if arr.size != horizon:
            raise ValueError(
                f"Model={model_name} task={task_name} window={window_idx} item={item_idx}: "
                f"expected horizon={horizon}, got {arr.size}"
            )

        if np.isnan(arr).any() or not np.isfinite(arr).all():
            raise ValueError(
                f"Model={model_name} task={task_name} window={window_idx} item={item_idx}: "
                "predictions contain NaN/inf"
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
