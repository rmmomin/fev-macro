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


def _normalize_freq(freq: str | None) -> str:
    if not freq:
        return "Q"

    upper = str(freq).upper()
    if upper.startswith("Q"):
        return "Q"
    if upper.startswith("M"):
        return "M"
    if upper.startswith("W"):
        return "W"
    if upper.startswith("D"):
        return "D"
    if upper.startswith("H"):
        return "H"
    if upper.startswith("A") or upper.startswith("Y"):
        return "Y"
    return str(freq)


def _to_statsforecast_dataframe(past_data: Dataset, task: Any) -> pd.DataFrame:
    id_col = get_task_id_column(task)
    ts_col = get_task_timestamp_column(task)
    target_col = get_task_target_column(task)

    if len(past_data) == 0:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    sample = past_data[0]
    if target_col in sample and isinstance(sample[target_col], (list, tuple, np.ndarray, pd.Series)):
        rows: list[dict[str, Any]] = []
        for row in past_data:
            item_id = row[id_col] if id_col in row else "__single_series__"
            ds = pd.to_datetime(pd.Series(row[ts_col]), errors="coerce")
            y = pd.to_numeric(pd.Series(row[target_col]), errors="coerce")
            mask = ds.notna() & y.notna()
            for dt, val in zip(ds[mask], y[mask], strict=False):
                rows.append({"unique_id": item_id, "ds": dt, "y": float(val)})
        return pd.DataFrame(rows).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    pdf = dataset_to_pandas(past_data)
    cols = [id_col, ts_col, target_col]
    missing = [c for c in cols if c not in pdf.columns]
    if missing:
        raise ValueError(f"Past data missing required columns for statsforecast: {missing}")

    sf_df = pdf[cols].copy()
    sf_df = sf_df.rename(columns={id_col: "unique_id", ts_col: "ds", target_col: "y"})
    sf_df["y"] = pd.to_numeric(sf_df["y"], errors="coerce")
    sf_df["ds"] = pd.to_datetime(sf_df["ds"], errors="coerce")
    return sf_df.dropna(subset=["ds", "y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _prediction_map_from_statsforecast_output(
    fcst_df: pd.DataFrame,
    item_order: list[Any],
    horizon: int,
) -> dict[Any, np.ndarray]:
    reserved = {"unique_id", "ds"}
    point_cols = [c for c in fcst_df.columns if c not in reserved]
    if not point_cols:
        raise ValueError("statsforecast output does not contain a point forecast column")

    point_col = point_cols[0]
    mapping: dict[Any, np.ndarray] = {}

    for item_id, grp in fcst_df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
        arr = grp[point_col].to_numpy(dtype=float)
        if arr.size != horizon:
            raise ValueError(
                f"statsforecast produced horizon={arr.size} for item_id={item_id}; expected {horizon}"
            )
        mapping[item_id] = arr

    for item_id in item_order:
        if item_id not in mapping:
            raise ValueError(f"statsforecast missing forecast for item_id={item_id}")

    return mapping


class _StatsForecastModel(BaseModel):
    def __init__(self, name: str, season_length: int = 4) -> None:
        super().__init__(name=name)
        self.season_length = int(season_length)

    def _build_stats_model(self):
        raise NotImplementedError

    def predict(self, past_data: Dataset, future_data: Dataset, task) -> Dataset:
        try:
            from statsforecast import StatsForecast
        except ImportError as exc:
            raise ImportError(
                "statsforecast is required for classical models. Install requirements.txt first."
            ) from exc

        horizon = get_task_horizon(task)
        sf_df = _to_statsforecast_dataframe(past_data, task)
        item_order = get_item_order(future_data, task)

        sf = StatsForecast(models=[self._build_stats_model()], freq=_normalize_freq(getattr(task, "freq", None)))
        fcst_df = sf.forecast(df=sf_df, h=horizon)

        pred_map = _prediction_map_from_statsforecast_output(fcst_df=fcst_df, item_order=item_order, horizon=horizon)
        return to_prediction_dataset(pred_map, item_order)


class AutoARIMAModel(_StatsForecastModel):
    def __init__(self, season_length: int = 4) -> None:
        super().__init__(name="auto_arima", season_length=season_length)

    def _build_stats_model(self):
        from statsforecast.models import AutoARIMA

        return AutoARIMA(season_length=self.season_length)


class AutoETSModel(_StatsForecastModel):
    def __init__(self, season_length: int = 4) -> None:
        super().__init__(name="auto_ets", season_length=season_length)

    def _build_stats_model(self):
        from statsforecast.models import AutoETS

        try:
            return AutoETS(season_length=self.season_length)
        except TypeError:
            return AutoETS()


class ThetaModel(_StatsForecastModel):
    def __init__(self, season_length: int = 4) -> None:
        super().__init__(name="theta", season_length=season_length)

    def _build_stats_model(self):
        from statsforecast.models import Theta

        try:
            return Theta(season_length=self.season_length)
        except TypeError:
            return Theta()
