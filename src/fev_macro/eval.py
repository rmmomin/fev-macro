from __future__ import annotations

import itertools
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
    summaries, _ = _run_models_on_task_core(
        task=task,
        models=models,
        num_windows=num_windows,
        past_data_adapter=past_data_adapter,
        with_records=False,
    )
    return summaries


def run_models_on_task_with_records(
    task: Any,
    models: dict[str, BaseModel],
    num_windows: int = 80,
    past_data_adapter: PastDataAdapter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run each model on one task and return summaries plus per-window predictions."""
    summaries, records = _run_models_on_task_core(
        task=task,
        models=models,
        num_windows=num_windows,
        past_data_adapter=past_data_adapter,
        with_records=True,
    )
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


def save_timing_report(summaries: list[dict[str, Any]], output_csv: str | Path) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "task_name",
        "model_name",
        "num_windows",
        "prep_time_s",
        "inference_time_s",
        "avg_time_per_window",
    ]

    if not summaries:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(output_csv, index=False)
        return empty

    df = pd.DataFrame(summaries).copy()
    if df.empty:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(output_csv, index=False)
        return empty

    if "task_name" not in df.columns:
        df["task_name"] = "task"
    if "model_name" not in df.columns:
        df["model_name"] = "model"

    for col, default in (("num_windows", 0), ("prep_time_s", 0.0), ("inference_time_s", 0.0)):
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(["task_name", "model_name"], as_index=False, dropna=False)
        .agg(
            num_windows=("num_windows", "max"),
            prep_time_s=("prep_time_s", "mean"),
            inference_time_s=("inference_time_s", "sum"),
        )
        .copy()
    )
    grouped["num_windows"] = grouped["num_windows"].fillna(0).astype(int)
    grouped["prep_time_s"] = grouped["prep_time_s"].fillna(0.0)
    grouped["inference_time_s"] = grouped["inference_time_s"].fillna(0.0)

    denom = grouped["num_windows"].replace(0, np.nan)
    grouped["avg_time_per_window"] = grouped["inference_time_s"] / denom
    grouped = grouped.sort_values(["inference_time_s", "task_name", "model_name"], ascending=[False, True, True])
    grouped = grouped.reset_index(drop=True)

    grouped[columns].to_csv(output_csv, index=False)
    return grouped[columns]


def _run_models_on_task_core(
    task: Any,
    models: dict[str, BaseModel],
    num_windows: int,
    past_data_adapter: PastDataAdapter | None,
    with_records: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    horizon = get_task_horizon(task)
    id_col = get_task_id_column(task)
    ts_col = get_task_timestamp_column(task)
    target_col = get_task_target_column(task)
    task_name = str(getattr(task, "task_name", "task"))

    window_cache, prep_time_s_total = _prepare_window_cache(
        task=task,
        num_windows=num_windows,
        past_data_adapter=past_data_adapter,
        id_col=id_col,
        timestamp_col=ts_col,
        target_col=target_col,
        with_records=with_records,
    )

    base_models, ensemble_models = _split_models(models=models)
    model_outputs: dict[str, dict[str, Any]] = {}
    predictions_cache: dict[str, list[Dataset]] = {}

    for model_name, model in base_models:
        predictions_per_window, inference_time_s = _predict_model_on_window_cache(
            model=model,
            model_name=model_name,
            task=task,
            task_name=task_name,
            horizon=horizon,
            window_cache=window_cache,
        )
        predictions_cache[model_name] = predictions_per_window
        model_outputs[model_name] = _build_model_output(
            task=task,
            model_name=model_name,
            predictions_per_window=predictions_per_window,
            inference_time_s=inference_time_s,
            prep_time_s_total=prep_time_s_total,
            window_cache=window_cache,
            horizon=horizon,
            task_name=task_name,
            with_records=with_records,
            ensemble_mode=None,
        )

    for model_name, model in ensemble_models:
        predictions_per_window, inference_time_s, fallback_reason = _posthoc_ensemble_predictions(
            model=model,
            model_name=model_name,
            task_name=task_name,
            horizon=horizon,
            window_cache=window_cache,
            predictions_cache=predictions_cache,
        )

        ensemble_mode = "posthoc"
        if predictions_per_window is None:
            if fallback_reason:
                print(
                    f"[eval] Ensemble '{model_name}' on task '{task_name}' "
                    f"falling back to native predict: {fallback_reason}"
                )
            predictions_per_window, inference_time_s = _predict_model_on_window_cache(
                model=model,
                model_name=model_name,
                task=task,
                task_name=task_name,
                horizon=horizon,
                window_cache=window_cache,
            )
            ensemble_mode = "native"

        predictions_cache[model_name] = predictions_per_window
        model_outputs[model_name] = _build_model_output(
            task=task,
            model_name=model_name,
            predictions_per_window=predictions_per_window,
            inference_time_s=inference_time_s,
            prep_time_s_total=prep_time_s_total,
            window_cache=window_cache,
            horizon=horizon,
            task_name=task_name,
            with_records=with_records,
            ensemble_mode=ensemble_mode,
        )

    ordered_summaries: list[dict[str, Any]] = []
    ordered_records: list[dict[str, Any]] = []
    for model_name in models.keys():
        payload = model_outputs.get(model_name)
        if payload is None:
            continue
        ordered_summaries.append(payload["summary"])
        if with_records:
            ordered_records.extend(payload["records"])

    return ordered_summaries, ordered_records


def _prepare_window_cache(
    task: Any,
    num_windows: int,
    past_data_adapter: PastDataAdapter | None,
    id_col: str,
    timestamp_col: str,
    target_col: str,
    with_records: bool,
) -> tuple[list[dict[str, Any]], float]:
    prep_time_s_total = 0.0
    window_cache: list[dict[str, Any]] = []

    max_windows = max(0, int(num_windows))
    windows = itertools.islice(task.iter_windows(), max_windows)
    for window_idx, window in enumerate(windows):
        input_data = window.get_input_data()
        if isinstance(input_data, tuple) and len(input_data) == 2:
            past_data, future_data = input_data
        else:
            past_data = input_data
            future_data = window.get_future_data()

        if past_data_adapter is not None:
            prep_start = time.perf_counter()
            past_data = past_data_adapter(past_data, future_data, task, window_idx)
            prep_time_s_total += time.perf_counter() - prep_start

        cache_row: dict[str, Any] = {
            "window_idx": int(window_idx),
            "past_data": past_data,
            "future_data": future_data,
            "expected_items": _expected_item_count(
                future_data=future_data,
                id_col=id_col,
                timestamp_col=timestamp_col,
            ),
        }

        if with_records:
            item_order = _item_order_from_future_data(future_data=future_data, id_col=id_col)
            truth_data = window.get_ground_truth()
            cache_row["item_order"] = item_order
            cache_row["future_payload"] = _future_payload_by_item(
                data=truth_data,
                id_col=id_col,
                timestamp_col=timestamp_col,
                target_col=target_col,
                item_order=item_order,
            )
            cache_row["past_last_target"] = _past_last_target_by_item(
                past_data=past_data,
                id_col=id_col,
                target_col=target_col,
            )

        window_cache.append(cache_row)

    return window_cache, float(prep_time_s_total)


def _split_models(models: dict[str, BaseModel]) -> tuple[list[tuple[str, BaseModel]], list[tuple[str, BaseModel]]]:
    base_models: list[tuple[str, BaseModel]] = []
    ensemble_models: list[tuple[str, BaseModel]] = []
    for model_name, model in models.items():
        members = getattr(model, "members", None)
        if isinstance(members, (list, tuple)) and len(members) > 0:
            ensemble_models.append((model_name, model))
        else:
            base_models.append((model_name, model))
    return base_models, ensemble_models


def _predict_model_on_window_cache(
    model: BaseModel,
    model_name: str,
    task: Any,
    task_name: str,
    horizon: int,
    window_cache: list[dict[str, Any]],
) -> tuple[list[Dataset], float]:
    predictions_per_window: list[Dataset] = []
    inference_time_s = 0.0

    for row in window_cache:
        window_idx = int(row["window_idx"])
        past_data = row["past_data"]
        future_data = row["future_data"]
        expected_items = int(row["expected_items"])

        start = time.perf_counter()
        raw_predictions = model.predict(past_data=past_data, future_data=future_data, task=task)
        inference_time_s += time.perf_counter() - start

        predictions = _coerce_prediction_dataset(raw_predictions)
        _validate_predictions(
            predictions=predictions,
            horizon=horizon,
            expected_items=expected_items,
            model_name=model_name,
            task_name=task_name,
            window_idx=window_idx,
        )
        predictions_per_window.append(predictions)

    return predictions_per_window, float(inference_time_s)


def _posthoc_ensemble_predictions(
    model: BaseModel,
    model_name: str,
    task_name: str,
    horizon: int,
    window_cache: list[dict[str, Any]],
    predictions_cache: dict[str, list[Dataset]],
) -> tuple[list[Dataset] | None, float, str | None]:
    members = getattr(model, "members", None)
    if not isinstance(members, (list, tuple)) or len(members) == 0:
        return None, 0.0, "model has no members attribute"

    member_names: list[str] = []
    for member in members:
        member_name = str(getattr(member, "name", "")).strip()
        if not member_name:
            return None, 0.0, "a member model has no name"
        member_names.append(member_name)

    missing = [member_name for member_name in member_names if member_name not in predictions_cache]
    if missing:
        return None, 0.0, f"missing cached member predictions: {missing}"

    raw_weights = np.asarray(getattr(model, "weights", []), dtype=float)
    if raw_weights.size != len(member_names):
        return None, 0.0, "weights length does not match member count"
    if np.any(raw_weights < 0) or float(raw_weights.sum()) <= 0.0:
        return None, 0.0, "invalid non-positive member weights"
    normalized_weights = raw_weights / raw_weights.sum()

    predictions_per_window: list[Dataset] = []
    combine_time_s = 0.0

    for window_pos, row in enumerate(window_cache):
        window_idx = int(row["window_idx"])
        expected_items = int(row["expected_items"])
        start = time.perf_counter()

        member_arrays: list[np.ndarray] = []
        valid_weights: list[float] = []
        shape_2d: tuple[int, int] | None = None

        for member_name, w in zip(member_names, normalized_weights, strict=False):
            member_pred_windows = predictions_cache.get(member_name, [])
            if window_pos >= len(member_pred_windows):
                continue

            arr = np.asarray(member_pred_windows[window_pos]["predictions"], dtype=float)
            if arr.ndim != 2 or not np.isfinite(arr).all():
                continue
            if shape_2d is None:
                shape_2d = arr.shape
            elif arr.shape != shape_2d:
                continue

            member_arrays.append(arr)
            valid_weights.append(float(w))

        if not member_arrays:
            return None, 0.0, f"no valid member predictions at window={window_idx}"

        weights = np.asarray(valid_weights, dtype=float)
        if float(weights.sum()) <= 0.0:
            return None, 0.0, f"all member weights were invalid at window={window_idx}"
        weights = weights / weights.sum()

        stacked = np.stack(member_arrays, axis=0)
        weighted = np.tensordot(weights, stacked, axes=(0, 0))
        predictions = Dataset.from_dict({"predictions": [row.tolist() for row in weighted]})
        combine_time_s += time.perf_counter() - start

        _validate_predictions(
            predictions=predictions,
            horizon=horizon,
            expected_items=expected_items,
            model_name=model_name,
            task_name=task_name,
            window_idx=window_idx,
        )
        predictions_per_window.append(predictions)

    return predictions_per_window, float(combine_time_s), None


def _build_model_output(
    task: Any,
    model_name: str,
    predictions_per_window: list[Dataset],
    inference_time_s: float,
    prep_time_s_total: float,
    window_cache: list[dict[str, Any]],
    horizon: int,
    task_name: str,
    with_records: bool,
    ensemble_mode: str | None,
) -> dict[str, Any]:
    summary = task.evaluation_summary(
        predictions_per_window=predictions_per_window,
        model_name=model_name,
        training_time_s=0.0,
        inference_time_s=float(inference_time_s),
    )
    summary = dict(summary)
    summary["inference_time_s"] = float(inference_time_s)
    summary["num_windows"] = int(len(predictions_per_window))
    summary["prep_time_s"] = float(prep_time_s_total)
    if ensemble_mode is not None:
        summary["ensemble_eval_mode"] = str(ensemble_mode)

    records: list[dict[str, Any]] = []
    if with_records:
        records = _build_records_for_model(
            predictions_per_window=predictions_per_window,
            model_name=model_name,
            window_cache=window_cache,
            horizon=horizon,
            task_name=task_name,
        )

    return {"summary": summary, "records": records}


def _build_records_for_model(
    predictions_per_window: list[Dataset],
    model_name: str,
    window_cache: list[dict[str, Any]],
    horizon: int,
    task_name: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for window_pos, predictions in enumerate(predictions_per_window):
        if window_pos >= len(window_cache):
            break

        row = window_cache[window_pos]
        window_idx = int(row["window_idx"])
        item_order = list(row.get("item_order", []))
        future_payload = dict(row.get("future_payload", {}))
        past_last_target = dict(row.get("past_last_target", {}))

        pred_rows = predictions["predictions"]
        for item_idx, item_id in enumerate(item_order):
            if item_idx >= len(pred_rows):
                continue

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
                if pd.isna(ts):
                    continue

                y_true = float(true_vals[step])
                y_pred = float(pred_arr[step])
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

    return records


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
