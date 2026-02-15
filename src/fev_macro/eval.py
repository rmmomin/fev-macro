from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from datasets import Dataset

from .models.base import (
    BaseModel,
    get_task_horizon,
    get_task_id_column,
    get_task_target_column,
    get_task_timestamp_column,
)


PastDataAdapter = Callable[[Dataset, Dataset, Any, int], Dataset]


def run_models_on_task(
    task: Any,
    models: dict[str, BaseModel],
    num_windows: int = 80,
    past_data_adapter: PastDataAdapter | None = None,
) -> list[dict[str, Any]]:
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

            if past_data_adapter is not None:
                past_data = past_data_adapter(past_data, future_data, task, window_idx)

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


def run_models_on_task_with_records(
    task: Any,
    models: dict[str, BaseModel],
    num_windows: int = 80,
    past_data_adapter: PastDataAdapter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run each model on one task and return summaries plus per-window predictions."""
    _ = num_windows
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    horizon = get_task_horizon(task)
    id_col = get_task_id_column(task)
    ts_col = get_task_timestamp_column(task)
    target_col = get_task_target_column(task)
    task_name = str(getattr(task, "task_name", "task"))

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

            if past_data_adapter is not None:
                past_data = past_data_adapter(past_data, future_data, task, window_idx)

            start = time.perf_counter()
            raw_predictions = model.predict(past_data=past_data, future_data=future_data, task=task)
            inference_time_s += time.perf_counter() - start

            predictions = _coerce_prediction_dataset(raw_predictions)
            expected_items = _expected_item_count(
                future_data=future_data,
                id_col=id_col,
                timestamp_col=ts_col,
            )
            _validate_predictions(
                predictions=predictions,
                horizon=horizon,
                expected_items=expected_items,
                model_name=model_name,
                task_name=task_name,
                window_idx=window_idx,
            )
            predictions_per_window.append(predictions)

            item_order = _item_order_from_future_data(future_data=future_data, id_col=id_col)
            truth_data = window.get_ground_truth()
            future_payload = _future_payload_by_item(
                data=truth_data,
                id_col=id_col,
                timestamp_col=ts_col,
                target_col=target_col,
                item_order=item_order,
            )
            past_last_target = _past_last_target_by_item(
                past_data=past_data,
                id_col=id_col,
                target_col=target_col,
            )

            pred_rows = predictions["predictions"]
            for item_idx, item_id in enumerate(item_order):
                pred_arr = np.asarray(pred_rows[item_idx], dtype=float).reshape(-1)
                payload = future_payload.get(item_id, {"timestamps": [], "targets": np.array([], dtype=float)})
                ts_vals = pd.to_datetime(pd.Series(payload["timestamps"]), errors="coerce")
                true_vals = pd.to_numeric(pd.Series(payload["targets"]), errors="coerce").to_numpy(dtype=float)
                usable = min(horizon, pred_arr.size, true_vals.size, len(ts_vals))
                if usable <= 0:
                    continue

                last_obs = past_last_target.get(item_id, np.nan)
                for step in range(usable):
                    ts = ts_vals.iloc[step]
                    y_true = float(true_vals[step])
                    y_pred = float(pred_arr[step])
                    if pd.isna(ts):
                        continue
                    records.append(
                        {
                            "task_name": task_name,
                            "model_name": model_name,
                            "window_idx": int(window_idx),
                            "item_id": str(item_id),
                            "horizon_step": int(step + 1),
                            "timestamp": pd.Timestamp(ts),
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "last_observed_target": float(last_obs) if np.isfinite(last_obs) else np.nan,
                        }
                    )

        summary = task.evaluation_summary(
            predictions_per_window=predictions_per_window,
            model_name=model_name,
            training_time_s=0.0,
            inference_time_s=float(inference_time_s),
        )
        summaries.append(summary)

    return summaries, records


def run_models_on_tasks(
    tasks: list[Any],
    models: dict[str, BaseModel],
    num_windows: int = 80,
    past_data_adapter: PastDataAdapter | None = None,
) -> list[dict[str, Any]]:
    all_summaries: list[dict[str, Any]] = []

    for task in tasks:
        task_summaries = run_models_on_task(
            task=task,
            models=models,
            num_windows=num_windows,
            past_data_adapter=past_data_adapter,
        )
        all_summaries.extend(task_summaries)

    return all_summaries


def run_models_on_tasks_with_records(
    tasks: list[Any],
    models: dict[str, BaseModel],
    num_windows: int = 80,
    past_data_adapter: PastDataAdapter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []

    for task in tasks:
        task_summaries, task_records = run_models_on_task_with_records(
            task=task,
            models=models,
            num_windows=num_windows,
            past_data_adapter=past_data_adapter,
        )
        all_summaries.extend(task_summaries)
        all_records.extend(task_records)

    return all_summaries, all_records


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


def _item_order_from_future_data(future_data: Dataset, id_col: str) -> list[Any]:
    if len(future_data) == 0:
        return []
    if id_col in future_data.column_names:
        return [row.get(id_col, "__single_series__") for row in future_data]
    return ["__single_series__"]


def _future_payload_by_item(
    data: Dataset,
    id_col: str,
    timestamp_col: str,
    target_col: str,
    item_order: list[Any],
) -> dict[Any, dict[str, Any]]:
    if len(data) == 0:
        return {}

    payload: dict[Any, dict[str, Any]] = {}
    for idx, rec in enumerate(data):
        if id_col in rec:
            item_id = rec.get(id_col, "__single_series__")
        elif item_order:
            item_id = item_order[min(idx, len(item_order) - 1)]
        else:
            item_id = "__single_series__"
        ts_val = rec.get(timestamp_col, [])
        y_val = rec.get(target_col, [])

        if isinstance(ts_val, (list, tuple, np.ndarray, pd.Series)):
            ts_seq = list(pd.to_datetime(pd.Series(ts_val), errors="coerce"))
        else:
            ts_seq = [pd.to_datetime(ts_val, errors="coerce")]

        if isinstance(y_val, (list, tuple, np.ndarray, pd.Series)):
            y_seq = pd.to_numeric(pd.Series(y_val), errors="coerce").to_numpy(dtype=float)
        else:
            y_seq = pd.to_numeric(pd.Series([y_val]), errors="coerce").to_numpy(dtype=float)

        payload[item_id] = {"timestamps": ts_seq, "targets": y_seq}

    return payload


def _past_last_target_by_item(
    past_data: Dataset,
    id_col: str,
    target_col: str,
) -> dict[Any, float]:
    out: dict[Any, float] = {}
    for rec in past_data:
        item_id = rec.get(id_col, "__single_series__") if id_col in rec else "__single_series__"
        y_val = rec.get(target_col, [])
        if isinstance(y_val, (list, tuple, np.ndarray, pd.Series)):
            arr = pd.to_numeric(pd.Series(y_val), errors="coerce").to_numpy(dtype=float)
        else:
            arr = pd.to_numeric(pd.Series([y_val]), errors="coerce").to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        out[item_id] = float(finite[-1]) if finite.size else np.nan
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
