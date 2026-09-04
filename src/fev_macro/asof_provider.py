from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .asof_store import AsofStore
from .pit import PITError, information_date
from .fred_transforms import fred_transform
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
        strict_pit: bool = True,
        series_specs: dict[str, dict] | None = None,
    ) -> None:
        self.strict_pit = bool(strict_pit)
        self.series_specs = series_specs or {}
        self.covariate_mode = _normalize_covariate_mode(covariate_mode)
        self.universes = _parse_universe(universe)
        self.store = AsofStore(db_path=Path(db_path).expanduser().resolve(), strict_pit=self.strict_pit)
        self.source_series_candidates = tuple(source_series_candidates or ("GDPC1",))
        self.available_series = self.store.available_series_ids()
        self.alias_maps: dict[str, dict[str, str]] = {
            uni: self.store.alias_map(universe=uni) for uni in self.universes
        }
        self._resolved_cache: dict[str, str | None] = {}
        self._qd_transform_codes: dict[str, int] = {}
        if self.covariate_mode == "processed" and not self.strict_pit:
            try:
                from .data import load_fred_qd_transform_codes
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

        if self.strict_pit:
            # Current alias tables/templates do not prove a historical mapping.
            # A fixed explicit ID is part of the benchmark specification.
            sid = self.series_specs.get(key, {}).get("series_id", key)
            self._resolved_cache[key] = sid if sid in self.available_series else None
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
            if self.strict_pit:
                raise PITError("No requested series has verified historical observations")
            return pd.DataFrame(), {
                "resolved_series": {},
                "unresolved_variables": unresolved,
                "requested_variables": request_vars,
                "snapshot_rows": 0,
                "quarterly_rows": 0,
            }

        if target_col not in resolved_map:
            raise PITError(f"Target {target_col} has no verified series mapping")
        if self.strict_pit and resolved_map[target_col] != "GDPC1":
            raise PITError("Strict real-GDP benchmark target must resolve to GDPC1")
        snap = self.store.snapshot_long(
            asof_ts=asof_ts, series_ids=series_ids, obs_start=obs_start, obs_end=obs_end,
            include_asof_used=True,
        )
        if snap.empty:
            raise PITError("No observations at the requested information cutoff")
        quarterly_series, coverage, inputs = {}, {}, []
        for variable, sid in resolved_map.items():
            sub = snap.loc[snap.series_id == sid].sort_values("obs_ts").copy()
            if sub.empty:
                if variable == target_col:
                    raise PITError("No target available at forecast origin")
                unresolved.append(variable)
                continue
            spec = self.series_specs.get(variable, {})
            frequency = spec.get("frequency")
            if frequency is None:
                row = self.store._con.execute(
                    "SELECT frequency_short FROM asof_series_meta WHERE series_id=?", [sid]).fetchone()
                frequency = row[0] if row else None
            if self.strict_pit and frequency not in {"M", "Q"}:
                raise PITError(f"Explicit monthly/quarterly frequency required for {variable}")
            frequency = frequency or "Q"
            if variable == target_col and frequency != "Q":
                raise PITError("GDP target must be quarterly levels")
            native = pd.Series(sub.value.to_numpy(), index=pd.PeriodIndex(sub.obs_ts, freq=frequency))
            if native.index.has_duplicates:
                raise PITError(f"Multiple observations per native period for {variable}")
            native = native.reindex(pd.period_range(native.index.min(), native.index.max(), freq=frequency))
            code = spec.get("tcode", 1)
            if variable != target_col and self.covariate_mode == "processed":
                if self.strict_pit and "tcode" not in spec:
                    raise PITError(f"Predeclared native-frequency transformation required for {variable}")
                code = spec.get("tcode", self._qd_transform_codes.get(variable, 1))
                native = pd.Series(fred_transform(native, int(code)).to_numpy(), index=native.index)
            native = native.replace([np.inf, -np.inf], np.nan)
            group = native.groupby(native.index.asfreq("Q-DEC"))
            # Partial quarters use only released months; counts remain visible.
            # No zeros/backfill are passed off as observations.
            quarterly_series[variable] = group.mean()
            coverage[variable] = {str(q): int(n) for q, n in group.count().items()}
            for row in sub.itertuples():
                inputs.append(dict(variable=variable, series_id=sid, obs_date=str(row.obs_ts.date()),
                                   vintage_date=str(row.asof_used.date()), realtime_end=row.realtime_end,
                                   value=None if pd.isna(row.value) else float(row.value),
                                   source=row.source, provenance_id=row.provenance_id,
                                   frequency=frequency, tcode=int(code)))
        panel = pd.DataFrame(quarterly_series).sort_index()
        if target_col not in panel or panel[target_col].notna().sum() == 0:
            raise PITError("No usable GDP target at forecast origin")
        panel.index.name = "quarter"
        panel = panel.reset_index()
        panel["timestamp"] = pd.PeriodIndex(panel.quarter, freq="Q-DEC").to_timestamp()
        return panel, {
            "resolved_series": resolved_map, "unresolved_variables": sorted(set(unresolved)),
            "requested_variables": request_vars, "snapshot_rows": len(snap),
            "quarterly_rows": len(panel), "strict_pit": self.strict_pit,
            "information_cutoff": information_date(asof_ts).date().isoformat() if self.strict_pit else str(asof_ts),
            "availability_rule": "vintage_date < New York origin date" if self.strict_pit else "legacy inclusive",
            "inputs": inputs, "quarterly_observation_counts": coverage,
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
                raise PITError("Cannot establish quarters for the requested PIT panel")
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
            raise PITError("Empty snapshot cannot be filled from a historical/latest panel")
        # Base is a schema/time request only. Its values never enter the forecast.
        out = asof_panel.loc[asof_panel.quarter.between(quarter_min, quarter_max)].copy()
        for col in covariate_cols:
            if col not in out:
                out[col] = np.nan
        out_meta = dict(meta)
        out_meta["used_snapshot"] = True
        return out.reset_index(drop=True), out_meta
