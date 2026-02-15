from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from .base import (
    BaseModel,
    dataset_to_pandas,
    get_item_order,
    get_task_horizon,
    get_task_id_column,
    get_task_target_column,
    get_task_timestamp_column,
    to_prediction_dataset,
)


class Chronos2Model(BaseModel):
    """Optional zero-shot Chronos-2 adapter.

    Requires optional dependencies (torch + chronos-forecasting).
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device_map: str = "auto",
        point_column_preference: tuple[str, ...] = ("predictions", "mean", "median", "0.5", "p50"),
    ) -> None:
        super().__init__(name="chronos2")
        try:
            from chronos.chronos2.pipeline import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting is required for Chronos-2. "
                "Install optional dependencies listed in requirements.txt comments."
            ) from exc

        self.pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)
        self.point_column_preference = point_column_preference

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        id_col = get_task_id_column(task)
        ts_col = get_task_timestamp_column(task)
        target_col = get_task_target_column(task)
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        known_covars = [
            c for c in list(getattr(task, "known_dynamic_columns", [])) if c in past_data.column_names or c in future_data.column_names
        ]
        past_covars = [c for c in list(getattr(task, "past_dynamic_columns", [])) if c in past_data.column_names]
        context_covars = list(dict.fromkeys(past_covars + known_covars))
        future_covars = [c for c in known_covars if c in future_data.column_names]

        context_df = _dataset_to_long_context_df(
            dataset=past_data,
            id_col=id_col,
            ts_col=ts_col,
            target_col=target_col,
            covariate_columns=context_covars,
        )
        future_df = _dataset_to_long_future_df(
            dataset=future_data,
            id_col=id_col,
            ts_col=ts_col,
            covariate_columns=future_covars,
        )

        pred_map: dict[Any, np.ndarray] = {}
        short_context_ids: list[Any] = []
        for item_id in item_order:
            item_ctx = context_df.loc[context_df["item_id"] == item_id, "target"]
            if len(item_ctx) < 3:
                short_context_ids.append(item_id)
                last_val = float(item_ctx.iloc[-1]) if len(item_ctx) > 0 else 0.0
                pred_map[item_id] = np.repeat(last_val, horizon).astype(float)

        eligible_ids = [item_id for item_id in item_order if item_id not in short_context_ids]
        if eligible_ids:
            pred_df = self.pipeline.predict_df(
                df=context_df[context_df["item_id"].isin(eligible_ids)].copy(),
                future_df=future_df[future_df["item_id"].isin(eligible_ids)].copy(),
                id_column="item_id",
                timestamp_column="timestamp",
                target="target",
                prediction_length=horizon,
            )

            point_col = self._select_point_column(pred_df)
            for item_id, grp in pred_df.sort_values(["item_id", "timestamp"]).groupby("item_id", sort=False):
                arr = grp[point_col].to_numpy(dtype=float)
                if arr.size != horizon:
                    raise ValueError(
                        f"Chronos-2 produced horizon={arr.size} for item_id={item_id}; expected {horizon}"
                    )
                pred_map[item_id] = arr

        for item_id in item_order:
            if item_id not in pred_map:
                raise ValueError(f"Chronos-2 missing forecast for item_id={item_id}")

        return to_prediction_dataset(pred_map, item_order)

    def _select_point_column(self, pred_df: pd.DataFrame) -> str:
        for col in self.point_column_preference:
            if col in pred_df.columns:
                return col

        forbidden = {"item_id", "timestamp", "target"}
        numeric = [c for c in pred_df.columns if c not in forbidden and pd.api.types.is_numeric_dtype(pred_df[c])]
        if not numeric:
            raise ValueError(f"Unable to infer Chronos-2 point forecast column from {list(pred_df.columns)}")
        return numeric[0]


def _dataset_to_long_context_df(
    dataset: Dataset,
    id_col: str,
    ts_col: str,
    target_col: str,
    covariate_columns: list[str] | None = None,
) -> pd.DataFrame:
    covariate_columns = list(covariate_columns or [])
    out_cols = ["item_id", "timestamp", "target", *covariate_columns]
    if len(dataset) == 0:
        return pd.DataFrame(columns=out_cols)

    sample = dataset[0]
    if isinstance(sample[target_col], (list, tuple, np.ndarray, pd.Series)):
        rows: list[dict[str, Any]] = []
        for rec in dataset:
            item_id = rec[id_col] if id_col in rec else "__single_series__"
            ts = pd.to_datetime(pd.Series(rec[ts_col]), errors="coerce")
            y = pd.to_numeric(pd.Series(rec[target_col]), errors="coerce")
            mask = ts.notna() & y.notna()
            if not mask.any():
                continue

            cov_by_name = {
                cov: _to_numeric_array(rec.get(cov), expected_len=len(ts)) for cov in covariate_columns
            }

            idxs = np.flatnonzero(mask.to_numpy())
            for idx in idxs:
                row: dict[str, Any] = {
                    "item_id": item_id,
                    "timestamp": ts.iloc[idx],
                    "target": float(y.iloc[idx]),
                }
                for cov in covariate_columns:
                    row[cov] = float(cov_by_name[cov].iloc[idx])
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=out_cols)

        df = pd.DataFrame(rows).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
        for cov in covariate_columns:
            df[cov] = pd.to_numeric(df[cov], errors="coerce").astype(float)
            df[cov] = df.groupby("item_id", sort=False)[cov].ffill().bfill().fillna(0.0)
        return df

    keep_cols = [id_col, ts_col, target_col, *[c for c in covariate_columns if c in dataset.column_names]]
    df = dataset_to_pandas(dataset)[keep_cols].copy()
    rename = {id_col: "item_id", ts_col: "timestamp", target_col: "target"}
    df = df.rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    for cov in covariate_columns:
        if cov in df.columns:
            df[cov] = pd.to_numeric(df[cov], errors="coerce").astype(float)

    df = df.dropna(subset=["timestamp", "target"]).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    for cov in covariate_columns:
        if cov in df.columns:
            df[cov] = df.groupby("item_id", sort=False)[cov].ffill().bfill().fillna(0.0)
    return df


def _dataset_to_long_future_df(
    dataset: Dataset,
    id_col: str,
    ts_col: str,
    covariate_columns: list[str] | None = None,
) -> pd.DataFrame:
    covariate_columns = list(covariate_columns or [])
    out_cols = ["item_id", "timestamp", *covariate_columns]
    if len(dataset) == 0:
        return pd.DataFrame(columns=out_cols)

    sample = dataset[0]
    if isinstance(sample[ts_col], (list, tuple, np.ndarray, pd.Series)):
        rows: list[dict[str, Any]] = []
        for rec in dataset:
            item_id = rec[id_col] if id_col in rec else "__single_series__"
            ts = pd.to_datetime(pd.Series(rec[ts_col]), errors="coerce")
            mask = ts.notna()
            if not mask.any():
                continue

            cov_by_name = {
                cov: _to_numeric_array(rec.get(cov), expected_len=len(ts)) for cov in covariate_columns
            }

            idxs = np.flatnonzero(mask.to_numpy())
            for idx in idxs:
                row: dict[str, Any] = {"item_id": item_id, "timestamp": ts.iloc[idx]}
                for cov in covariate_columns:
                    row[cov] = float(cov_by_name[cov].iloc[idx])
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=out_cols)

        df = pd.DataFrame(rows).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
        for cov in covariate_columns:
            df[cov] = pd.to_numeric(df[cov], errors="coerce").astype(float)
            df[cov] = df.groupby("item_id", sort=False)[cov].ffill().bfill().fillna(0.0)
        return df

    keep_cols = [id_col, ts_col, *[c for c in covariate_columns if c in dataset.column_names]]
    df = dataset_to_pandas(dataset)[keep_cols].copy()
    rename = {id_col: "item_id", ts_col: "timestamp"}
    df = df.rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for cov in covariate_columns:
        if cov in df.columns:
            df[cov] = pd.to_numeric(df[cov], errors="coerce").astype(float)

    df = df.dropna(subset=["timestamp"]).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    for cov in covariate_columns:
        if cov in df.columns:
            df[cov] = df.groupby("item_id", sort=False)[cov].ffill().bfill().fillna(0.0)
    return df


def _to_numeric_array(value: Any, expected_len: int) -> pd.Series:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        s = pd.to_numeric(pd.Series(value), errors="coerce")
    elif value is None:
        s = pd.Series([np.nan] * expected_len, dtype=float)
    else:
        s = pd.to_numeric(pd.Series([value] * expected_len), errors="coerce")

    if len(s) < expected_len:
        if len(s) == 0:
            s = pd.Series([np.nan] * expected_len, dtype=float)
        else:
            s = pd.concat([s, pd.Series([s.iloc[-1]] * (expected_len - len(s)), dtype=float)], ignore_index=True)
    elif len(s) > expected_len:
        s = s.iloc[:expected_len]

    return s.astype(float).ffill().bfill().fillna(0.0).reset_index(drop=True)
