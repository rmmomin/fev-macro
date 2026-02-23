from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from datasets import Dataset

from ..data import extract_past_cutoff_timestamp, resolve_gdpc1_release_csv_path
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
GDPNowSelectionMode = Literal["final_pre_release", "asof_window_cutoff"]


class AtlantaFedGDPNowModel(BaseModel):
    """Atlanta Fed GDPNow benchmark model with configurable information cutoff policy."""

    def __init__(
        self,
        gdpnow_csv_path: str | Path | None = None,
        release_csv_path: str | Path | None = None,
        *,
        name: str = "atlantafed_gdpnow",
        selection_mode: GDPNowSelectionMode = "final_pre_release",
    ) -> None:
        super().__init__(name=name)
        self.selection_mode = _normalize_selection_mode(selection_mode)
        self.allow_missing_predictions = self.selection_mode == "asof_window_cutoff"

        self.gdpnow_csv_path = _resolve_gdpnow_csv_path(gdpnow_csv_path)
        self.release_csv_path = resolve_gdpc1_release_csv_path(path=release_csv_path)

        updates = _load_gdpnow_updates(gdpnow_csv_path=self.gdpnow_csv_path)
        self._updates_by_quarter: dict[pd.Period, pd.DataFrame] = {
            quarter: grp.sort_values("forecast_date", kind="stable").reset_index(drop=True)
            for quarter, grp in updates.groupby("quarter", sort=False)
        }
        self._first_release_by_quarter = _load_first_release_dates(release_csv_path=self.release_csv_path)

        self._selection_debug_rows: list[dict[str, Any]] = []
        self._window_counter = 0

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)
        future_timestamps = get_timestamps_by_item(future_data, task)

        window_cutoff_ts = pd.Timestamp(extract_past_cutoff_timestamp(past_data=past_data, task=task))
        window_id = int(self._window_counter)
        self._window_counter += 1
        task_name = str(getattr(task, "task_name", "task"))

        preds: dict[Any, np.ndarray] = {}
        for item_id in item_order:
            hist = np.asarray(history.get(item_id, np.array([], dtype=float)), dtype=float).reshape(-1)
            hist = hist[np.isfinite(hist)]
            fallback = float(hist[-1]) if hist.size else 0.0

            ts = pd.to_datetime(pd.Series(future_timestamps.get(item_id, [])), errors="coerce").dropna()
            quarters = pd.PeriodIndex(ts, freq="Q-DEC")

            out = np.empty(horizon, dtype=float)
            for step in range(horizon):
                if step >= len(quarters):
                    out[step] = fallback if self.selection_mode == "final_pre_release" else np.nan
                    continue

                quarter = quarters[step]
                selected_value, selected_forecast_date, first_release_date = self._select_nowcast(
                    quarter=quarter,
                    window_cutoff_ts=window_cutoff_ts,
                )

                value = selected_value
                if not np.isfinite(value) and self.selection_mode == "final_pre_release":
                    value = fallback
                out[step] = float(value)

                self._selection_debug_rows.append(
                    _build_selection_debug_row(
                        model_name=self.name,
                        task_name=task_name,
                        window_id=window_id,
                        window_cutoff_ts=window_cutoff_ts,
                        target_quarter=quarter,
                        selected_forecast_date=selected_forecast_date,
                        first_release_date=first_release_date,
                    )
                )

            preds[item_id] = out

        return to_prediction_dataset(preds, item_order)

    def get_selection_debug_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._selection_debug_rows]

    def _select_nowcast(
        self,
        *,
        quarter: pd.Period,
        window_cutoff_ts: pd.Timestamp,
    ) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
        rows = self._updates_by_quarter.get(quarter)
        first_release = self._first_release_by_quarter.get(quarter)

        if rows is None or rows.empty:
            return np.nan, None, first_release

        if self.selection_mode == "final_pre_release":
            if first_release is None or pd.isna(first_release):
                return np.nan, None, first_release
            eligible = rows.loc[rows["forecast_date"] < first_release].copy()
        else:
            cutoff = pd.Timestamp(window_cutoff_ts)
            eligible = rows.loc[rows["forecast_date"] <= cutoff].copy()

        if eligible.empty:
            return np.nan, None, first_release

        selected = eligible.sort_values("forecast_date", kind="stable").iloc[-1]
        forecast_date = pd.Timestamp(selected["forecast_date"])
        value = float(selected["nowcast"])

        if self.selection_mode == "final_pre_release":
            if first_release is None or not (forecast_date < first_release):
                raise AssertionError(
                    "GDPNow final_pre_release selection violated strict pre-release rule: "
                    f"quarter={quarter}, forecast_date={forecast_date.date()}, "
                    f"first_release_date={first_release.date() if first_release is not None else 'NA'}"
                )
        else:
            cutoff = pd.Timestamp(window_cutoff_ts)
            if forecast_date > cutoff:
                raise AssertionError(
                    "GDPNow asof_window_cutoff selection violated as-of window cutoff rule: "
                    f"quarter={quarter}, window_cutoff_timestamp={cutoff.isoformat()}, "
                    f"forecast_date={forecast_date.isoformat()}"
                )

        return value, forecast_date, first_release


class AtlantaFedGDPNowFinalPreReleaseModel(AtlantaFedGDPNowModel):
    """Latest GDPNow estimate strictly before BEA first release."""

    def __init__(
        self,
        gdpnow_csv_path: str | Path | None = None,
        release_csv_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            gdpnow_csv_path=gdpnow_csv_path,
            release_csv_path=release_csv_path,
            name="atlantafed_gdpnow_final_pre_release",
            selection_mode="final_pre_release",
        )


class AtlantaFedGDPNowAsOfWindowCutoffModel(AtlantaFedGDPNowModel):
    """Latest GDPNow estimate as-of each rolling window cutoff timestamp."""

    def __init__(
        self,
        gdpnow_csv_path: str | Path | None = None,
        release_csv_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            gdpnow_csv_path=gdpnow_csv_path,
            release_csv_path=release_csv_path,
            name="atlantafed_gdpnow_asof_window_cutoff",
            selection_mode="asof_window_cutoff",
        )


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


def _parse_atlantafed_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%m/%d/%y", errors="coerce")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(values, errors="coerce")


def _load_gdpnow_updates(gdpnow_csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(gdpnow_csv_path)
    if raw.empty:
        return pd.DataFrame(columns=["quarter", "forecast_date", "nowcast"])

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
        return pd.DataFrame(columns=["quarter", "forecast_date", "nowcast"])

    now["quarter"] = pd.PeriodIndex(now["quarter_end"], freq="Q-DEC")
    now = now.sort_values(["quarter", "forecast_date"], kind="stable").reset_index(drop=True)
    return now[["quarter", "forecast_date", "nowcast"]].copy()


def _load_first_release_dates(release_csv_path: Path) -> dict[pd.Period, pd.Timestamp]:
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

    out: dict[pd.Period, pd.Timestamp] = {}
    for _, row in rel.iterrows():
        quarter = row["quarter"]
        release_ts = row["first_release_date"]
        if isinstance(quarter, pd.Period) and pd.notna(release_ts):
            out[quarter] = pd.Timestamp(release_ts)
    return out


def _normalize_selection_mode(mode: str) -> GDPNowSelectionMode:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"final_pre_release", "asof_window_cutoff"}:
        raise ValueError(
            "GDPNow selection_mode must be one of {'final_pre_release', 'asof_window_cutoff'}"
        )
    return cast(GDPNowSelectionMode, mode_norm)


def _build_selection_debug_row(
    *,
    model_name: str,
    task_name: str,
    window_id: int,
    window_cutoff_ts: pd.Timestamp,
    target_quarter: pd.Period,
    selected_forecast_date: pd.Timestamp | None,
    first_release_date: pd.Timestamp | None,
) -> dict[str, Any]:
    selected = pd.Timestamp(selected_forecast_date) if selected_forecast_date is not None else pd.NaT
    first_release = pd.Timestamp(first_release_date) if first_release_date is not None else pd.NaT
    cutoff = pd.Timestamp(window_cutoff_ts)

    is_after_window_cutoff = bool(pd.notna(selected) and selected > cutoff)
    is_on_or_after_first_release = bool(pd.notna(selected) and pd.notna(first_release) and selected >= first_release)

    info_advantage_days: float = np.nan
    if pd.notna(selected):
        info_advantage_days = float((selected.normalize() - cutoff.normalize()).days)

    return {
        "model_name": str(model_name),
        "task_name": str(task_name),
        "window_id": int(window_id),
        "window_cutoff_timestamp": cutoff,
        "target_quarter": str(target_quarter),
        "selected_forecast_date": selected,
        "bea_first_release_date": first_release,
        "info_advantage_days": info_advantage_days,
        "is_after_window_cutoff": bool(is_after_window_cutoff),
        "is_on_or_after_first_release": bool(is_on_or_after_first_release),
    }
