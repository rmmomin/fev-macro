from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class RealtimeTaskShim:
    horizon: int
    target_col: str
    past_dynamic_columns: list[str]
    known_dynamic_columns: list[str]
    task_name: str = "realtime_oos"
    id_column: str = "id"
    timestamp_column: str = "timestamp"
    freq: str = "Q"

    @property
    def target(self) -> str:
        return self.target_col

    @property
    def prediction_length(self) -> int:
        return self.horizon


def select_covariate_columns(
    train_df: pd.DataFrame,
    target_col: str,
    max_covariates: int = 120,
) -> list[str]:
    exclude = {
        target_col,
        "quarter",
        "timestamp",
        "vintage",
        "vintage_timestamp",
        "asof_date",
        "__origin_vintage",
        "__origin_schedule",
    }
    candidates: list[str] = []
    for col in train_df.columns:
        if col in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        s = pd.to_numeric(train_df[col], errors="coerce")
        if s.notna().sum() < 8:
            continue
        candidates.append(col)
    return candidates[:max_covariates]


def train_df_to_datasets(
    train_df: pd.DataFrame,
    target_col: str,
    horizon: int,
    covariate_cols: Sequence[str],
    id_value: str = "gdp",
) -> tuple[Any, Any, RealtimeTaskShim]:
    try:
        from datasets import Dataset
    except Exception as exc:
        raise ImportError("datasets package is required for realtime fev adapters.") from exc

    df = train_df.copy().sort_values("quarter").reset_index(drop=True)
    q_all = pd.PeriodIndex(df["quarter"], freq="Q-DEC")
    y_all = pd.to_numeric(df[target_col], errors="coerce")
    mask = y_all.notna()
    if not mask.any():
        raise ValueError("No non-missing target in train_df.")

    q_hist = q_all[mask]
    y_hist = y_all[mask].to_numpy(dtype=float)
    past_ts = [period.start_time for period in q_hist]

    past_row: dict[str, object] = {
        "id": str(id_value),
        "timestamp": past_ts,
        target_col: y_hist.tolist(),
    }
    for col in covariate_cols:
        vals = pd.to_numeric(df.loc[mask, col], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        past_row[col] = vals.tolist()

    last_q = q_hist[-1]
    fut_q = [last_q + i for i in range(1, horizon + 1)]
    future_ts = [period.start_time for period in fut_q]
    future_row: dict[str, object] = {"id": str(id_value), "timestamp": future_ts}
    for col in covariate_cols:
        hist_cov = pd.to_numeric(df.loc[mask, col], errors="coerce").ffill().bfill().fillna(0.0)
        last_val = float(hist_cov.iloc[-1]) if len(hist_cov) else 0.0
        all_cov = pd.to_numeric(df[col], errors="coerce")
        cov_map = pd.Series(all_cov.to_numpy(dtype=float), index=q_all).groupby(level=0).last()

        future_vals: list[float] = []
        for fq in fut_q:
            v = cov_map.get(fq, np.nan)
            if pd.notna(v):
                last_val = float(v)
            future_vals.append(last_val)
        future_row[col] = future_vals

    past_data = Dataset.from_list([past_row])
    future_data = Dataset.from_list([future_row])
    task = RealtimeTaskShim(
        horizon=horizon,
        target_col=target_col,
        past_dynamic_columns=list(covariate_cols),
        known_dynamic_columns=list(covariate_cols),
    )
    return past_data, future_data, task
