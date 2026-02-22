from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from ..data import resolve_gdpc1_release_csv_path
from .base import (
    BaseModel,
    get_history_by_item,
    get_item_order,
    get_task_horizon,
    get_timestamps_by_item,
    to_prediction_dataset,
)

DEFAULT_GDPNOW_CSV_CANDIDATES: tuple[Path, ...] = (
    Path("data/panels/atlantafedgdpnow.csv"),
    Path("data/atlantafedgdpnow.csv"),
)


class AtlantaFedGDPNowModel(BaseModel):
    """Atlanta Fed GDPNow benchmark using latest nowcast before BEA first release."""

    def __init__(
        self,
        gdpnow_csv_path: str | Path | None = None,
        release_csv_path: str | Path | None = None,
    ) -> None:
        super().__init__(name="atlantafed_gdpnow")
        self.gdpnow_csv_path = _resolve_gdpnow_csv_path(gdpnow_csv_path)
        self.release_csv_path = resolve_gdpc1_release_csv_path(path=release_csv_path)
        self._nowcast_by_quarter = _build_pre_release_nowcast_map(
            gdpnow_csv_path=self.gdpnow_csv_path,
            release_csv_path=self.release_csv_path,
        )

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)
        future_timestamps = get_timestamps_by_item(future_data, task)

        preds: dict[Any, np.ndarray] = {}
        for item_id in item_order:
            hist = np.asarray(history.get(item_id, np.array([], dtype=float)), dtype=float).reshape(-1)
            hist = hist[np.isfinite(hist)]
            fallback = float(hist[-1]) if hist.size else 0.0

            ts = pd.to_datetime(pd.Series(future_timestamps.get(item_id, [])), errors="coerce").dropna()
            quarters = pd.PeriodIndex(ts, freq="Q-DEC")

            out = np.empty(horizon, dtype=float)
            for step in range(horizon):
                value = np.nan
                if step < len(quarters):
                    value = self._nowcast_by_quarter.get(quarters[step], np.nan)
                if not np.isfinite(value):
                    value = fallback
                out[step] = float(value)

            preds[item_id] = out

        return to_prediction_dataset(preds, item_order)


def _resolve_gdpnow_csv_path(path: str | Path | None) -> Path:
    if path is not None and str(path).strip():
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"GDPNow CSV does not exist: {resolved}")
        return resolved

    for candidate in DEFAULT_GDPNOW_CSV_CANDIDATES:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved

    checked = ", ".join(str(p.expanduser().resolve()) for p in DEFAULT_GDPNOW_CSV_CANDIDATES)
    raise FileNotFoundError(f"Could not locate Atlanta Fed GDPNow CSV. Checked: {checked}")


def _find_column(raw_columns: list[str], required_tokens: tuple[str, ...]) -> str:
    normalized: dict[str, str] = {
        str(col).strip().lstrip("\ufeff").lower(): str(col) for col in raw_columns
    }
    for norm, original in normalized.items():
        if all(token in norm for token in required_tokens):
            return original
    raise ValueError(f"Could not find required column with tokens={required_tokens} in {raw_columns}")


def _build_pre_release_nowcast_map(
    *,
    gdpnow_csv_path: Path,
    release_csv_path: Path,
) -> dict[pd.Period, float]:
    raw = pd.read_csv(gdpnow_csv_path)
    if raw.empty:
        return {}

    forecast_col = _find_column(list(raw.columns), ("forecast", "date"))
    quarter_col = _find_column(list(raw.columns), ("quarter", "forecast"))
    value_col = _find_column(list(raw.columns), ("gdp", "nowcast"))

    now = pd.DataFrame(
        {
            "forecast_date": _parse_atlantafed_date(raw[forecast_col]),
            "quarter_end": _parse_atlantafed_date(raw[quarter_col]),
            "nowcast": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    now = now.dropna(subset=["forecast_date", "quarter_end", "nowcast"]).copy()
    if now.empty:
        return {}
    now["quarter"] = pd.PeriodIndex(now["quarter_end"], freq="Q-DEC")
    now = now.sort_values(["quarter", "forecast_date"], kind="stable").reset_index(drop=True)

    releases = pd.read_csv(release_csv_path)
    if releases.empty or "observation_date" not in releases.columns or "first_release_date" not in releases.columns:
        return {}

    rel = releases[["observation_date", "first_release_date"]].copy()
    rel["observation_date"] = pd.to_datetime(rel["observation_date"], errors="coerce")
    rel["first_release_date"] = pd.to_datetime(rel["first_release_date"], errors="coerce")
    rel = rel.dropna(subset=["observation_date", "first_release_date"]).copy()
    if rel.empty:
        return {}

    rel["quarter"] = pd.PeriodIndex(rel["observation_date"], freq="Q-DEC")
    rel = rel.sort_values("first_release_date", kind="stable").drop_duplicates(subset=["quarter"], keep="last")
    rel = rel[["quarter", "first_release_date"]].reset_index(drop=True)

    merged = now.merge(rel, on="quarter", how="inner")
    merged = merged.loc[merged["forecast_date"] <= merged["first_release_date"]].copy()
    if merged.empty:
        return {}

    merged = merged.sort_values(["quarter", "forecast_date"], kind="stable").reset_index(drop=True)
    latest = merged.groupby("quarter", as_index=False).tail(1)
    mapping: dict[pd.Period, float] = {}
    for _, row in latest.iterrows():
        quarter = row["quarter"]
        value = float(row["nowcast"])
        if isinstance(quarter, pd.Period) and np.isfinite(value):
            mapping[quarter] = value
    return mapping


def _parse_atlantafed_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%m/%d/%y", errors="coerce")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(values, errors="coerce")
