from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .asof_store import AsofStore
from .data import (
    DEFAULT_SOURCE_SERIES_CANDIDATES,
    build_real_gdp_target_series_from_time_rows,
    load_fred_qd_transform_codes,
)
from .fred_aliases import candidate_series_ids, dedupe_preserve_order

CovariateMode = Literal["unprocessed", "processed"]


def _normalize_covariate_mode(mode: str) -> CovariateMode:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"unprocessed", "processed"}:
        raise ValueError(
            f"Unsupported covariate_mode={mode!r}. Expected one of {{'unprocessed','processed'}}."
        )
    return mode_norm  # type: ignore[return-value]


def _parse_universe(value: str) -> tuple[str, ...]:
    token = str(value).strip().lower()
    if token == "both":
        return ("qd", "md")
    if token in {"qd", "md"}:
        return (token,)
    raise ValueError("universe must be one of {'qd','md','both'}")


def _to_quarterly_mean(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[timestamp_col])

    out = frame.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=[timestamp_col])

    numeric_cols = [c for c in out.columns if c != timestamp_col]
    if not numeric_cols:
        return pd.DataFrame(columns=[timestamp_col])

    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    out["_quarter"] = out[timestamp_col].dt.to_period("Q-DEC")
    grouped = out.groupby("_quarter", dropna=False, sort=True)[numeric_cols].mean().reset_index()
    grouped[timestamp_col] = grouped["_quarter"].dt.to_timestamp(how="start")
    grouped = grouped.drop(columns=["_quarter"]).sort_values(timestamp_col).reset_index(drop=True)
    return grouped


class AsofVintageProvider:
    """Build quarter-aligned as-of panels from versioned ALFRED/FRED observations."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        covariate_mode: CovariateMode = "unprocessed",
        universe: str = "both",
        historical_qd_dir: str | Path = "data/historical/qd",
        source_series_candidates: Sequence[str] | None = None,
    ) -> None:
        self.covariate_mode = _normalize_covariate_mode(covariate_mode)
        self.universes = _parse_universe(universe)
        self.store = AsofStore(db_path=Path(db_path).expanduser().resolve())
        self.source_series_candidates = tuple(source_series_candidates or DEFAULT_SOURCE_SERIES_CANDIDATES)
        self.available_series = self.store.available_series_ids()
        self.alias_maps: dict[str, dict[str, str]] = {
            uni: self.store.alias_map(universe=uni) for uni in self.universes
        }
        self._resolved_cache: dict[str, str | None] = {}
        self._qd_transform_codes: dict[str, int] = {}
        if self.covariate_mode == "processed":
            try:
                self._qd_transform_codes = load_fred_qd_transform_codes(
                    historical_qd_dir=historical_qd_dir,
                    vintage_period=None,
                )
            except FileNotFoundError:
                self._qd_transform_codes = {}

    def close(self) -> None:
        self.store.close()

    def _resolve_series_id(self, variable_name: str) -> str | None:
        key = str(variable_name).strip()
        if not key:
            return None
        if key in self._resolved_cache:
            return self._resolved_cache[key]

        for uni in self.universes:
            sid = self.alias_maps.get(uni, {}).get(key)
            if sid and sid in self.available_series:
                self._resolved_cache[key] = sid
                return sid

        for cand in candidate_series_ids(key):
            if cand in self.available_series:
                self._resolved_cache[key] = cand
                return cand

        self._resolved_cache[key] = None
        return None

    def resolve_variable_map(self, variable_names: Iterable[str]) -> tuple[dict[str, str], list[str]]:
        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        for variable in dedupe_preserve_order(variable_names):
            sid = self._resolve_series_id(variable)
            if sid is None:
                unresolved.append(variable)
                continue
            resolved[variable] = sid
        return resolved, unresolved

    def build_panel_asof(
        self,
        *,
        asof_ts: object,
        target_col: str,
        covariate_columns: Sequence[str],
        obs_start: object | None = None,
        obs_end: object | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        request_vars = dedupe_preserve_order([target_col, *covariate_columns])
        resolved_map, unresolved = self.resolve_variable_map(request_vars)
        series_ids = dedupe_preserve_order(resolved_map.values())
        if not series_ids:
            return pd.DataFrame(), {
                "resolved_series": {},
                "unresolved_variables": unresolved,
                "requested_variables": request_vars,
                "snapshot_rows": 0,
                "quarterly_rows": 0,
            }

        snap = self.store.snapshot_wide(
            asof_ts=asof_ts,
            series_ids=series_ids,
            obs_start=obs_start,
            obs_end=obs_end,
            timestamp_name="timestamp",
        )
        if snap.empty:
            return pd.DataFrame(), {
                "resolved_series": resolved_map,
                "unresolved_variables": unresolved,
                "requested_variables": request_vars,
                "snapshot_rows": 0,
                "quarterly_rows": 0,
            }

        wide = pd.DataFrame({"timestamp": pd.to_datetime(snap["timestamp"], errors="coerce")})
        for variable, series_id in resolved_map.items():
            if series_id in snap.columns:
                wide[variable] = pd.to_numeric(snap[series_id], errors="coerce")
        if target_col not in wide.columns:
            wide[target_col] = np.nan

        quarterly = _to_quarterly_mean(wide, timestamp_col="timestamp")
        if quarterly.empty:
            return pd.DataFrame(), {
                "resolved_series": resolved_map,
                "unresolved_variables": unresolved,
                "requested_variables": request_vars,
                "snapshot_rows": int(len(snap)),
                "quarterly_rows": 0,
            }

        allow_covs = [c for c in covariate_columns if c in quarterly.columns]
        panel, panel_meta = build_real_gdp_target_series_from_time_rows(
            wide_df=quarterly,
            target_series_name=target_col,
            target_transform="level",
            source_series_candidates=(target_col, *self.source_series_candidates),
            include_covariates=True,
            covariate_allowlist=allow_covs,
            apply_fred_transforms=(self.covariate_mode == "processed"),
            fred_transform_codes=self._qd_transform_codes,
            covariate_mode=self.covariate_mode,
        )
        panel = panel.rename(columns={"target": target_col})
        panel["quarter"] = pd.PeriodIndex(pd.to_datetime(panel["timestamp"], errors="coerce"), freq="Q-DEC")
        panel = panel.dropna(subset=["quarter"]).sort_values("quarter").reset_index(drop=True)
        return panel, {
            "resolved_series": resolved_map,
            "unresolved_variables": unresolved,
            "requested_variables": request_vars,
            "snapshot_rows": int(len(snap)),
            "quarterly_rows": int(len(quarterly)),
            "panel_meta": panel_meta,
        }

    def adapt_train_df(
        self,
        *,
        train_df: pd.DataFrame,
        asof_ts: object,
        cutoff_quarter: pd.Period,
        target_col: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if train_df.empty:
            return train_df.copy(), {"used_snapshot": False, "reason": "empty_train_df"}

        base = train_df.copy()
        if "quarter" not in base.columns:
            if "timestamp" not in base.columns:
                return base, {"used_snapshot": False, "reason": "missing_quarter_and_timestamp"}
            base["quarter"] = pd.PeriodIndex(pd.to_datetime(base["timestamp"], errors="coerce"), freq="Q-DEC")
        else:
            base["quarter"] = pd.PeriodIndex(base["quarter"], freq="Q-DEC")

        exclude = {
            "timestamp",
            "quarter",
            "vintage",
            "vintage_timestamp",
            "asof_date",
            target_col,
            "item_id",
            "__origin_vintage",
            "__origin_schedule",
        }
        covariate_cols = [c for c in base.columns if c not in exclude and not str(c).startswith("__")]
        quarter_min = pd.Period(base["quarter"].min(), freq="Q-DEC")
        quarter_max = pd.Period(cutoff_quarter, freq="Q-DEC")

        asof_panel, meta = self.build_panel_asof(
            asof_ts=asof_ts,
            target_col=target_col,
            covariate_columns=covariate_cols,
            obs_start=quarter_min.start_time,
            obs_end=quarter_max.end_time,
        )
        if asof_panel.empty:
            out_meta = dict(meta)
            out_meta["used_snapshot"] = False
            out_meta["reason"] = "empty_snapshot_panel"
            return base, out_meta

        merge_cols = ["quarter", target_col, *covariate_cols]
        asof_subset_cols = [c for c in merge_cols if c in asof_panel.columns]
        merged = base.merge(
            asof_panel[asof_subset_cols],
            on="quarter",
            how="left",
            suffixes=("", "__asof"),
        )

        replaced_total = 0
        for col in [target_col, *covariate_cols]:
            asof_col = f"{col}__asof"
            if asof_col not in merged.columns:
                continue

            current = pd.to_numeric(merged[col], errors="coerce")
            incoming = pd.to_numeric(merged[asof_col], errors="coerce")
            changed = incoming.notna() & (~current.notna() | (np.abs(incoming - current) > 0.0))
            replaced_total += int(changed.sum())

            merged[col] = np.where(incoming.notna(), incoming, current)
            merged = merged.drop(columns=[asof_col])

        merged["quarter"] = pd.PeriodIndex(merged["quarter"], freq="Q-DEC")
        ordered_cols = [c for c in train_df.columns if c in merged.columns]
        for col in merged.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        merged = merged[ordered_cols]

        out_meta = dict(meta)
        out_meta["used_snapshot"] = True
        out_meta["replaced_values"] = int(replaced_total)
        return merged, out_meta
