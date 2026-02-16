from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import numpy as np
import pandas as pd
from datasets import Dataset
from .fred_transforms import apply_fred_transform_codes, extract_fred_transform_codes

DEFAULT_SOURCE_SERIES_CANDIDATES: tuple[str, ...] = (
    "GDPC1",
    "RGDP",
    "REALGDP",
    "REAL_GDP",
    "GDP",
)
DEFAULT_TARGET_SERIES_NAME = "LOG_REAL_GDP"
DEFAULT_TARGET_TRANSFORM = "log_level"
SUPPORTED_TARGET_TRANSFORMS = {"level", "log_level", "saar_growth"}
SUPPORTED_COVARIATE_MODES = {"unprocessed", "processed"}
FRED_QD_VINTAGE_PATTERN = re.compile(r"fred[-_]?qd_(\d{4})m(\d{1,2})\.csv$", flags=re.IGNORECASE)
DEFAULT_GDPC1_RELEASE_CSV_CANDIDATES: tuple[Path, ...] = (
    Path("data/gdpc1_releases_first_second_third.csv"),
    Path("data/panels/gdpc1_releases_first_second_third.csv"),
)
RELEASE_STAGE_TO_COLUMN: dict[str, str] = {
    "first": "first_release",
    "second": "second_release",
    "third": "third_release",
    "latest": "latest_release",
}
RELEASE_STAGE_TO_REALTIME_SAAR_COLUMN: dict[str, str] = {
    "first": "qoq_saar_growth_realtime_first_pct",
    "second": "qoq_saar_growth_realtime_second_pct",
    "third": "qoq_saar_growth_realtime_third_pct",
}


def _default_qd_panel_candidates(covariate_mode: Literal["unprocessed", "processed"]) -> tuple[Path, ...]:
    if covariate_mode == "processed":
        return (
            Path("data/panels/fred_qd_vintage_panel_processed.parquet"),
            Path("data/panels/fred_qd_vintage_panel_process.parquet"),
        )
    return (Path("data/panels/fred_qd_vintage_panel.parquet"),)


def _resolve_panel_candidate(
    explicit_path: str | Path | None,
    *,
    covariate_mode: Literal["unprocessed", "processed"],
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()

    candidates = _default_qd_panel_candidates(covariate_mode=covariate_mode)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return candidates[0].expanduser().resolve()


def discover_historical_qd_vintage_files(historical_qd_dir: str | Path) -> dict[pd.Period, Path]:
    """Discover historical FRED-QD vintage files keyed by monthly vintage period."""
    preferred = Path(historical_qd_dir).expanduser()
    root = preferred.resolve() if preferred.exists() else None
    if root is None:
        root = _autodiscover_historical_qd_dir(preferred)
    if root is None:
        raise FileNotFoundError(f"Historical FRED-QD directory does not exist: {preferred.resolve()}")

    if not root.exists():
        raise FileNotFoundError(f"Historical FRED-QD directory does not exist: {root}")

    files: dict[pd.Period, Path] = {}
    for path in sorted(root.rglob("*.csv")):
        period = _parse_fred_qd_vintage_period(path.name)
        if period is None:
            continue
        files[period] = path

    if not files:
        raise FileNotFoundError(f"No FRED-QD vintage CSVs found under: {root}")

    return dict(sorted(files.items(), key=lambda kv: kv[0]))


def load_historical_fred_qd_vintage_dataframe(
    csv_path: str | Path,
    return_transform_codes: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Load one historical FRED-QD vintage CSV and return numeric quarterly rows.

    If `return_transform_codes` is True, also return parsed FRED transform codes.
    """
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"Historical vintage CSV has no rows: {csv_path}")

    date_col = str(raw.columns[0])
    transform_codes = extract_fred_transform_codes(raw_df=raw, first_col_name=date_col)
    date_values = pd.to_datetime(raw[date_col], format="%m/%d/%Y", errors="coerce")
    data = raw.loc[date_values.notna()].copy()
    if data.empty:
        raise ValueError(f"No data rows found in historical vintage CSV: {csv_path}")

    data["timestamp"] = pd.to_datetime(data[date_col], format="%m/%d/%Y", errors="coerce")
    data = data.drop(columns=[date_col]).sort_values("timestamp").reset_index(drop=True)

    for col in data.columns:
        if col == "timestamp":
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if return_transform_codes:
        return data, transform_codes
    return data


def load_fred_qd_transform_codes(
    historical_qd_dir: str | Path,
    vintage_period: pd.Period | None = None,
) -> dict[str, int]:
    """Load FRED-QD transform-code map from a historical vintage CSV.

    When `vintage_period` is None, uses the latest available vintage under `historical_qd_dir`.
    """
    vintage_files = discover_historical_qd_vintage_files(historical_qd_dir=historical_qd_dir)
    if not vintage_files:
        return {}

    if vintage_period is None:
        selected_period = max(vintage_files.keys())
    else:
        if vintage_period in vintage_files:
            selected_period = vintage_period
        else:
            eligible = [p for p in vintage_files if p <= vintage_period]
            if not eligible:
                selected_period = min(vintage_files.keys())
            else:
                selected_period = max(eligible)

    _, transform_codes = load_historical_fred_qd_vintage_dataframe(
        csv_path=vintage_files[selected_period],
        return_transform_codes=True,
    )
    return transform_codes


def build_covariate_df(
    historical_qd_dir: str | Path,
    qd_panel_path: str | Path | None = None,
    covariate_mode: Literal["unprocessed", "processed"] = "unprocessed",
    target_series_name: str = DEFAULT_TARGET_SERIES_NAME,
    source_series_candidates: Sequence[str] | None = None,
    vintage_period: pd.Period | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build quarterly covariate frame from a selected FRED-QD vintage.

    Returns dataframe with `timestamp` + covariate columns.
    """
    mode = _normalize_covariate_mode(covariate_mode)
    source_series_candidates = tuple(source_series_candidates or DEFAULT_SOURCE_SERIES_CANDIDATES)

    source_kind = "historical_csv"
    selected_vintage: pd.Period | None = None
    transform_codes: dict[str, int] = {}

    try:
        vintage_files = discover_historical_qd_vintage_files(historical_qd_dir=historical_qd_dir)
        if not vintage_files:
            raise FileNotFoundError("No historical FRED-QD vintage files found.")

        if vintage_period is None:
            selected_vintage = max(vintage_files.keys())
        elif vintage_period in vintage_files:
            selected_vintage = vintage_period
        else:
            eligible = [p for p in vintage_files if p <= vintage_period]
            selected_vintage = max(eligible) if eligible else min(vintage_files.keys())

        wide_df, transform_codes = load_historical_fred_qd_vintage_dataframe(
            csv_path=vintage_files[selected_vintage],
            return_transform_codes=True,
        )
    except FileNotFoundError:
        source_kind = "panel_parquet"
        panel_candidate = _resolve_panel_candidate(
            qd_panel_path,
            covariate_mode=mode,
        )
        if not panel_candidate.exists():
            raise FileNotFoundError(
                f"No historical FRED-QD CSVs and no QD panel parquet at {panel_candidate}"
            )

        panel_df = pd.read_parquet(panel_candidate)
        if "vintage" not in panel_df.columns or "timestamp" not in panel_df.columns:
            raise ValueError(
                "QD panel parquet must contain 'vintage' and 'timestamp' columns."
            )
        panel_df = panel_df.copy()
        panel_df["vintage"] = panel_df["vintage"].astype(str)
        panel_df["timestamp"] = pd.to_datetime(panel_df["timestamp"], errors="coerce")
        panel_df = panel_df.dropna(subset=["vintage", "timestamp"]).sort_values(["vintage", "timestamp"])
        if panel_df.empty:
            raise ValueError(f"QD panel has no usable rows: {panel_candidate}")

        available_periods = sorted({pd.Period(v, freq="M") for v in panel_df["vintage"].unique()})
        if not available_periods:
            raise ValueError(f"QD panel has no vintage labels: {panel_candidate}")

        if vintage_period is None:
            selected_vintage = max(available_periods)
        elif vintage_period in available_periods:
            selected_vintage = vintage_period
        else:
            eligible = [p for p in available_periods if p <= vintage_period]
            selected_vintage = max(eligible) if eligible else min(available_periods)

        vintage_key = f"{selected_vintage.year:04d}-{selected_vintage.month:02d}"
        wide_df = panel_df.loc[panel_df["vintage"] == vintage_key].copy()
        wide_df = wide_df.drop(columns=["vintage", "vintage_timestamp"], errors="ignore")

    target_df, target_meta = build_real_gdp_target_series_from_time_rows(
        wide_df=wide_df,
        target_series_name=target_series_name,
        target_transform="level",
        source_series_candidates=source_series_candidates,
        include_covariates=True,
        apply_fred_transforms=(mode == "processed"),
        fred_transform_codes=transform_codes,
        covariate_mode=mode,
    )
    covariate_columns = list(target_meta.get("covariate_columns", []) or [])

    covariate_df = target_df[["timestamp", *covariate_columns]].copy()
    return covariate_df, {
        "source": source_kind,
        "selected_vintage": str(selected_vintage) if selected_vintage is not None else "",
        "covariate_mode": mode,
        "covariate_count": int(len(covariate_columns)),
        "transform_code_count": int(len(transform_codes)),
    }


def build_real_gdp_target_series_from_time_rows(
    wide_df: pd.DataFrame,
    target_series_name: str = DEFAULT_TARGET_SERIES_NAME,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    source_series_candidates: Sequence[str] | None = None,
    include_covariates: bool = False,
    covariate_allowlist: Sequence[str] | None = None,
    apply_fred_transforms: bool = False,
    fred_transform_codes: dict[str, int] | None = None,
    covariate_mode: Literal["unprocessed", "processed"] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build real-GDP target series from row-wise time dataframe with `timestamp` column."""
    _validate_target_transform(target_transform)
    source_series_candidates = tuple(source_series_candidates or DEFAULT_SOURCE_SERIES_CANDIDATES)

    if "timestamp" not in wide_df.columns:
        raise ValueError("wide_df must include a 'timestamp' column")

    data = wide_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid timestamped rows found in wide_df")

    available_ids = set(map(str, [c for c in data.columns if c != "timestamp"]))
    computed = False
    source_series = target_series_name

    if target_series_name in data.columns:
        target = pd.to_numeric(data[target_series_name], errors="coerce")
    else:
        source_series = _pick_source_series(available_ids=available_ids, preferred=source_series_candidates)
        if source_series is None:
            raise ValueError(
                "Could not find real GDP level series in historical dataframe. "
                f"Checked candidates={list(source_series_candidates)}"
            )
        source = pd.to_numeric(data[source_series], errors="coerce")
        target = _transform_from_level(level_series=source, target_transform=target_transform)
        computed = target_transform != "level"

    out = pd.DataFrame(
        {
            "item_id": target_series_name,
            "timestamp": data["timestamp"],
            "target": pd.to_numeric(target, errors="coerce").to_numpy(dtype=float),
        }
    )

    resolved_covariate_mode = (
        _normalize_covariate_mode(covariate_mode)
        if covariate_mode is not None
        else ("processed" if apply_fred_transforms else "unprocessed")
    )
    use_fred_transforms = resolved_covariate_mode == "processed"

    covariate_columns: list[str] = []
    transformed_covariates: list[str] = []
    if include_covariates:
        excluded = {"timestamp", target_series_name, source_series}
        allowlist = set(covariate_allowlist or [])
        selected_covariates: list[str] = []
        for cov in data.columns:
            if cov in excluded:
                continue
            if covariate_allowlist is not None and cov not in allowlist:
                continue
            selected_covariates.append(cov)

        if selected_covariates:
            cov_frame = data[selected_covariates].apply(pd.to_numeric, errors="coerce")
            if use_fred_transforms and fred_transform_codes:
                cov_frame = apply_fred_transform_codes(
                    data_df=cov_frame,
                    transform_codes=fred_transform_codes,
                    columns=selected_covariates,
                )
                transformed_covariates = [c for c in selected_covariates if c in fred_transform_codes]
            out = pd.concat([out, cov_frame.reset_index(drop=True)], axis=1)
            covariate_columns.extend(selected_covariates)

    out = _drop_invalid_and_fill_covariates(
        out=out,
        covariate_columns=covariate_columns,
        covariate_mode=resolved_covariate_mode,
    )
    return out, {
        "computed": computed,
        "source_series": source_series,
        "target_series": target_series_name,
        "target_transform": target_transform,
        "covariate_columns": covariate_columns,
        "apply_fred_transforms": bool(use_fred_transforms),
        "covariate_mode": resolved_covariate_mode,
        "transformed_covariates": transformed_covariates,
        "transform_code_count": len(fred_transform_codes or {}),
    }


class HistoricalQuarterlyVintageProvider:
    """Window-aware provider that swaps training history to historical FRED-QD vintages."""

    def __init__(
        self,
        historical_qd_dir: str | Path,
        target_series_name: str = DEFAULT_TARGET_SERIES_NAME,
        target_transform: str = DEFAULT_TARGET_TRANSFORM,
        source_series_candidates: Sequence[str] | None = None,
        covariate_columns: Sequence[str] | None = None,
        include_covariates: bool = True,
        apply_fred_transforms: bool = True,
        covariate_mode: Literal["unprocessed", "processed"] | None = None,
        exclude_years_list: Sequence[int] | None = None,
        timestamp_mapping: dict[pd.Timestamp, pd.Timestamp] | None = None,
        strict: bool = True,
        fallback_to_earliest: bool = False,
        qd_panel_path: str | Path | None = None,
    ) -> None:
        self.target_series_name = target_series_name
        self.target_transform = target_transform
        self.source_series_candidates = tuple(source_series_candidates or DEFAULT_SOURCE_SERIES_CANDIDATES)
        self.include_covariates = bool(include_covariates)
        self.covariate_mode = (
            _normalize_covariate_mode(covariate_mode)
            if covariate_mode is not None
            else ("processed" if apply_fred_transforms else "unprocessed")
        )
        self.apply_fred_transforms = self.covariate_mode == "processed"
        self.covariate_columns = list(covariate_columns or [])
        self.exclude_years_list = sorted({int(y) for y in (exclude_years_list or [])})
        self.timestamp_mapping = {
            pd.Timestamp(k): pd.Timestamp(v) for k, v in (timestamp_mapping or {}).items()
        }
        self.strict = bool(strict)
        self.fallback_to_earliest = bool(fallback_to_earliest)

        self.vintage_files: dict[pd.Period, Path] = {}
        self._panel_df: pd.DataFrame | None = None
        self._uses_panel_source = False

        try:
            self.vintage_files = discover_historical_qd_vintage_files(historical_qd_dir=historical_qd_dir)
            self.historical_qd_dir = next(iter(self.vintage_files.values())).parent.resolve()
            self.vintage_periods = sorted(self.vintage_files.keys())
        except FileNotFoundError:
            panel_candidate = _resolve_panel_candidate(
                qd_panel_path,
                covariate_mode=self.covariate_mode,
            )
            if not panel_candidate.exists():
                raise
            panel_df = pd.read_parquet(panel_candidate)
            if "vintage" not in panel_df.columns:
                raise ValueError(f"Qd panel missing required 'vintage' column: {panel_candidate}")
            if "timestamp" not in panel_df.columns:
                raise ValueError(f"Qd panel missing required 'timestamp' column: {panel_candidate}")
            panel_df = panel_df.copy()
            panel_df["timestamp"] = pd.to_datetime(panel_df["timestamp"], errors="coerce")
            panel_df["vintage"] = panel_df["vintage"].astype(str)
            panel_df = panel_df.dropna(subset=["timestamp", "vintage"]).sort_values(["vintage", "timestamp"]).reset_index(drop=True)
            periods = sorted({pd.Period(v, freq="M") for v in panel_df["vintage"].unique()})
            self.vintage_periods = periods
            self._panel_df = panel_df
            self._uses_panel_source = True
            self.historical_qd_dir = panel_candidate

        self._cache: dict[pd.Period, pd.DataFrame] = {}
        self._selection_cache: dict[pd.Period, pd.Period | None] = {}

    @property
    def earliest_vintage(self) -> pd.Period:
        return self.vintage_periods[0]

    @property
    def latest_vintage(self) -> pd.Period:
        return self.vintage_periods[-1]

    def available_range_str(self) -> str:
        return f"{self.earliest_vintage}..{self.latest_vintage}"

    def select_vintage_period(self, cutoff_timestamp: pd.Timestamp, allow_fallback: bool | None = None) -> pd.Period | None:
        """Pick latest monthly vintage period <= cutoff month."""
        cutoff_period = pd.Period(pd.Timestamp(cutoff_timestamp), freq="M")
        if cutoff_period in self._selection_cache:
            cached = self._selection_cache[cutoff_period]
            if cached is not None:
                return cached

        allow_fallback = self.fallback_to_earliest if allow_fallback is None else bool(allow_fallback)
        eligible = [p for p in self.vintage_periods if p <= cutoff_period]
        if eligible:
            selected = eligible[-1]
            self._selection_cache[cutoff_period] = selected
            return selected

        if allow_fallback and self.vintage_periods:
            selected = self.vintage_periods[0]
            self._selection_cache[cutoff_period] = selected
            return selected

        self._selection_cache[cutoff_period] = None
        return None

    def compatible_window_count(self, task: Any) -> int:
        """Count trailing evaluation windows that are usable for this provider/task."""
        window_usable: list[bool] = []
        for window in task.iter_windows():
            try:
                input_data = window.get_input_data()
            except ValueError as exc:
                # fev may surface infeasible early windows when num_windows is larger than
                # the available history. Those windows must be excluded.
                if "too short" in str(exc):
                    window_usable.append(False)
                    continue
                raise

            if isinstance(input_data, tuple) and len(input_data) == 2:
                past_data, _ = input_data
            else:
                past_data = input_data

            cutoff = _extract_past_cutoff_timestamp(past_data=past_data, task=task)
            cutoff_actual = self.timestamp_mapping.get(pd.Timestamp(cutoff), pd.Timestamp(cutoff))
            selected = self.select_vintage_period(cutoff_actual, allow_fallback=not self.strict)
            has_vintage = selected is not None
            if self.strict:
                window_usable.append(has_vintage)
            else:
                window_usable.append(True)

        if not window_usable:
            return 0
        trailing = 0
        for is_valid in reversed(window_usable):
            if is_valid:
                trailing += 1
            else:
                break
        return trailing

    def adapt_past_data(self, past_data: Dataset, task: Any) -> Dataset:
        """Return a past_data Dataset reconstructed from the matching historical vintage."""
        id_col = _task_id_column(task)
        ts_col = _task_timestamp_column(task)
        target_col = _task_target_column(task)
        required_covars = _task_covariate_columns(task)

        rows: list[dict[str, Any]] = []
        for rec in past_data:
            ts_reindexed = pd.to_datetime(pd.Series(rec.get(ts_col, [])), errors="coerce")
            ts_reindexed = ts_reindexed.dropna().reset_index(drop=True)
            if ts_reindexed.empty:
                continue

            actual_ts = ts_reindexed.map(lambda t: self.timestamp_mapping.get(pd.Timestamp(t), pd.Timestamp(t)))
            actual_ts = pd.to_datetime(actual_ts, errors="coerce").reset_index(drop=True)
            if actual_ts.isna().all():
                continue
            actual_ts = actual_ts.where(actual_ts.notna(), ts_reindexed)

            cutoff = pd.Timestamp(actual_ts.iloc[-1])
            selected = self.select_vintage_period(cutoff_timestamp=cutoff, allow_fallback=not self.strict)
            if selected is None:
                raise ValueError(
                    "No historical FRED-QD vintage available for cutoff "
                    f"{cutoff.date()} (earliest vintage is {self.earliest_vintage})."
                )

            vintage_df = self._load_vintage_frame(vintage_period=selected)
            vintage_hist = vintage_df.loc[vintage_df["timestamp"] <= cutoff].copy()
            vintage_hist = vintage_hist.sort_values("timestamp").reset_index(drop=True)
            vintage_indexed = vintage_hist.set_index("timestamp", drop=True)

            aligned_target = pd.to_numeric(vintage_indexed.reindex(actual_ts)["target"], errors="coerce")
            fallback_target = _series_like_to_numeric(rec.get(target_col), expected_len=len(ts_reindexed))
            target_values = np.where(aligned_target.notna(), aligned_target, fallback_target).astype(float)

            row: dict[str, Any] = {
                id_col: rec.get(id_col, "__single_series__"),
                ts_col: ts_reindexed.tolist(),
                target_col: target_values.tolist(),
            }

            for cov in required_covars:
                fallback_cov = _series_like_to_numeric(rec.get(cov), expected_len=len(ts_reindexed))
                if cov in vintage_indexed.columns:
                    from_vintage = pd.to_numeric(vintage_indexed.reindex(actual_ts)[cov], errors="coerce")
                    cov_values = np.where(from_vintage.notna(), from_vintage, fallback_cov)
                else:
                    cov_values = fallback_cov.to_numpy(dtype=float)

                cov_series = _impute_covariate_series(
                    pd.Series(cov_values, dtype=float),
                    covariate_mode=self.covariate_mode,
                )
                row[cov] = cov_series.to_numpy(dtype=float).tolist()

            rows.append(row)

        if not rows:
            raise ValueError("HistoricalQuarterlyVintageProvider produced an empty past_data dataset")

        return Dataset.from_list(rows)

    def _load_vintage_frame(self, vintage_period: pd.Period) -> pd.DataFrame:
        if vintage_period in self._cache:
            return self._cache[vintage_period]

        if self._uses_panel_source:
            if self._panel_df is None:
                raise ValueError("Qd panel source is enabled but no panel dataframe is loaded.")
            vintage_key = f"{vintage_period.year:04d}-{vintage_period.month:02d}"
            wide = self._panel_df.loc[self._panel_df["vintage"] == vintage_key].copy()
            wide = wide.drop(columns=["vintage", "vintage_timestamp"], errors="ignore")
            transform_codes: dict[str, int] = {}
            apply_transforms = self.covariate_mode == "processed"
        else:
            csv_path = self.vintage_files[vintage_period]
            wide, transform_codes = load_historical_fred_qd_vintage_dataframe(
                csv_path=csv_path,
                return_transform_codes=True,
            )
            apply_transforms = self.covariate_mode == "processed"

        target_df, _ = build_real_gdp_target_series_from_time_rows(
            wide_df=wide,
            target_series_name=self.target_series_name,
            target_transform=self.target_transform,
            source_series_candidates=self.source_series_candidates,
            include_covariates=self.include_covariates,
            covariate_allowlist=self.covariate_columns or None,
            apply_fred_transforms=apply_transforms,
            fred_transform_codes=transform_codes,
            covariate_mode=self.covariate_mode,
        )

        if self.exclude_years_list:
            target_df = exclude_years(target_df, years=self.exclude_years_list)

        target_df = target_df.sort_values("timestamp").reset_index(drop=True)
        self._cache[vintage_period] = target_df
        return target_df


def load_fev_dataset(
    config: str | None = None,
    dataset_path: str = "data/panels/gdpc1_releases_first_second_third.csv",
    split: str = "train",
    dataset_revision: str | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
) -> Dataset:
    """Load a fev-compatible dataset from a local parquet file.

    Remote Hugging Face dataset pulls are intentionally disabled.
    """
    _ = config
    _ = split
    _ = dataset_revision
    _ = dataset_kwargs

    parquet_path = Path(dataset_path).expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(
            "Only local dataset paths are supported. "
            f"Expected parquet at: {parquet_path}"
        )
    suffix = parquet_path.suffix.lower()
    if suffix == ".parquet":
        return Dataset.from_parquet(str(parquet_path))
    if suffix == ".csv":
        frame = pd.read_csv(parquet_path)
        return Dataset.from_pandas(frame, preserve_index=False)
    raise ValueError(
        "Only local parquet/csv datasets are supported by load_fev_dataset. "
        f"Got: {parquet_path}"
    )


def find_gdp_column_candidates(dataset: Dataset, id_col: str = "id") -> dict[str, list[str]]:
    """Return likely GDP-related identifiers and columns."""
    pdf = dataset.to_pandas()
    columns = list(pdf.columns)

    id_candidates: list[str] = []
    id_column = id_col if id_col in pdf.columns else ("item_id" if "item_id" in pdf.columns else None)
    if id_column is not None:
        values = [str(v) for v in pd.Series(pdf[id_column]).dropna().unique().tolist()]
        id_candidates = [v for v in values if _looks_like_gdp(v)]

    column_candidates = [c for c in columns if _looks_like_gdp(c)]
    return {
        "id_candidates": sorted(id_candidates),
        "column_candidates": sorted(column_candidates),
        "columns": columns,
    }


def build_real_gdp_target_series(
    dataset: Dataset,
    target_series_name: str = DEFAULT_TARGET_SERIES_NAME,
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
    source_series_candidates: Sequence[str] | None = None,
    id_col: str = "item_id",
    timestamp_col: str = "timestamp",
    target_col: str = "target",
    include_covariates: bool = False,
    apply_fred_transforms: bool = False,
    fred_transform_codes: dict[str, int] | None = None,
    covariate_mode: Literal["unprocessed", "processed"] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build real-GDP target series and optional aligned covariates in long format.

    If target_series_name exists, it is used directly.
    Otherwise, target is computed from real GDP level series y_t using:
      - level: y_t
      - log_level: log(y_t)
      - saar_growth: 100 * ((y_t / y_{t-1})**4 - 1)

    Returned dataframe has columns:
        item_id, timestamp, target, [covariates...]
    """
    _validate_target_transform(target_transform)
    source_series_candidates = tuple(source_series_candidates or DEFAULT_SOURCE_SERIES_CANDIDATES)
    resolved_covariate_mode = (
        _normalize_covariate_mode(covariate_mode)
        if covariate_mode is not None
        else ("processed" if apply_fred_transforms else "unprocessed")
    )
    pdf = dataset.to_pandas()

    if id_col in pdf.columns and target_col in pdf.columns:
        return _build_from_long_format(
            pdf=pdf,
            target_series_name=target_series_name,
            target_transform=target_transform,
            source_series_candidates=source_series_candidates,
            id_col=id_col,
            timestamp_col=timestamp_col,
            target_col=target_col,
            include_covariates=include_covariates,
            apply_fred_transforms=apply_fred_transforms,
            fred_transform_codes=fred_transform_codes,
            covariate_mode=resolved_covariate_mode,
        )

    return _build_from_wide_format(
        pdf=pdf,
        target_series_name=target_series_name,
        target_transform=target_transform,
        source_series_candidates=source_series_candidates,
        timestamp_col=timestamp_col,
        include_covariates=include_covariates,
        apply_fred_transforms=apply_fred_transforms,
        fred_transform_codes=fred_transform_codes,
        covariate_mode=resolved_covariate_mode,
    )


def resolve_gdpc1_release_csv_path(path: str | Path | None = None) -> Path:
    """Resolve gdpc1 release CSV from explicit path or default candidate locations."""
    if path is not None and str(path).strip():
        csv_path = Path(path).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Release CSV does not exist: {csv_path}")
        return csv_path

    for candidate in DEFAULT_GDPC1_RELEASE_CSV_CANDIDATES:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved

    candidates = ", ".join(str(p) for p in DEFAULT_GDPC1_RELEASE_CSV_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate gdpc1 release CSV. Checked default paths: "
        f"{candidates}"
    )


def build_release_target_scaffold(
    release_csv_path: str | Path | None = None,
    target_series_name: str = DEFAULT_TARGET_SERIES_NAME,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build timestamp scaffold from gdpc1 release CSV for stage-specific truth overlays."""
    csv_path = resolve_gdpc1_release_csv_path(path=release_csv_path)
    releases = pd.read_csv(csv_path)

    if "observation_date" not in releases.columns:
        raise ValueError(f"Release CSV missing required column 'observation_date': {csv_path}")

    rel = releases.copy()
    rel["observation_date"] = pd.to_datetime(rel["observation_date"], errors="coerce")
    rel = rel.loc[rel["observation_date"].notna()].copy()
    if rel.empty:
        raise ValueError(f"Release CSV has no valid observation_date rows: {csv_path}")

    rel["quarter"] = pd.PeriodIndex(rel["observation_date"], freq="Q-DEC")
    rel = rel.sort_values("observation_date").drop_duplicates(subset=["quarter"], keep="last")

    out = pd.DataFrame(
        {
            "item_id": target_series_name,
            "timestamp": rel["observation_date"].to_numpy(),
            "target": np.nan,
        }
    )
    return out, {
        "source": "gdpc1_release_csv_scaffold",
        "release_csv_path": str(csv_path),
        "rows": int(len(out)),
    }


def format_gdp_item_id(
    base: str,
    release_metric: str,
    release_stage: str,
    target_transform: str,
) -> str:
    """Build a truth-aware GDP item id for evaluation datasets."""
    base_norm = str(base).strip().lower()
    metric_norm = str(release_metric).strip().lower()
    stage_norm = str(release_stage).strip().lower()
    transform_norm = str(target_transform).strip().lower()

    if metric_norm in {"realtime_qoq_saar", "realtime_saar", "realtime_qoq_saar_pct"}:
        return f"{base_norm}_qoq_saar_{stage_norm}_pct"
    return f"{base_norm}_{stage_norm}_{transform_norm}"


def apply_gdpc1_release_truth_target(
    dataset_df: pd.DataFrame,
    release_csv_path: str | Path | None = None,
    release_stage: str = "first",
    release_metric: str = "level",
    target_transform: str = DEFAULT_TARGET_TRANSFORM,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace dataset target values with selected GDP release-stage truth by quarter."""
    _validate_target_transform(target_transform)
    stage = _normalize_release_stage(release_stage)
    metric_kind = _normalize_release_metric(release_metric)

    if metric_kind == "realtime_qoq_saar":
        if stage not in RELEASE_STAGE_TO_REALTIME_SAAR_COLUMN:
            raise ValueError(
                "Realtime SAAR release metric supports stages first/second/third only; "
                f"got stage={stage!r}."
            )
        stage_col = RELEASE_STAGE_TO_REALTIME_SAAR_COLUMN[stage]
    else:
        stage_col = RELEASE_STAGE_TO_COLUMN[stage]

    csv_path = resolve_gdpc1_release_csv_path(path=release_csv_path)
    releases = pd.read_csv(csv_path)

    required = {"observation_date", stage_col}
    missing = sorted(required.difference(releases.columns))
    if missing:
        raise ValueError(f"Release CSV missing required columns: {missing}")

    rel = releases.copy()
    rel["observation_date"] = pd.to_datetime(rel["observation_date"], errors="coerce")
    rel[stage_col] = pd.to_numeric(rel[stage_col], errors="coerce")
    rel = rel.loc[rel["observation_date"].notna() & rel[stage_col].notna()].copy()
    if rel.empty:
        raise ValueError(
            f"Release CSV has no valid rows for stage '{stage}' ({stage_col}): {csv_path}"
        )

    rel["quarter"] = pd.PeriodIndex(rel["observation_date"], freq="Q-DEC")
    rel = rel.sort_values("observation_date").drop_duplicates(subset=["quarter"], keep="last")
    release_levels_by_quarter = rel.set_index("quarter")[stage_col].astype(float)

    out = dataset_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["quarter"] = pd.PeriodIndex(out["timestamp"], freq="Q-DEC")

    mapped_values = pd.to_numeric(out["quarter"].map(release_levels_by_quarter), errors="coerce")
    if metric_kind == "realtime_qoq_saar":
        # Values are already q/q SAAR percent and can be used directly as the target.
        out["target"] = mapped_values.astype(float)
    else:
        out["target"] = _transform_from_level(level_series=mapped_values, target_transform=target_transform).astype(float)

    new_item_id = format_gdp_item_id(
        base="gdpc1",
        release_metric=metric_kind,
        release_stage=stage,
        target_transform=target_transform,
    )
    out["item_id"] = new_item_id

    covariate_columns = [c for c in out.columns if c not in {"item_id", "timestamp", "target", "quarter"}]
    rows_with_release_target = int(np.isfinite(out["target"].to_numpy(dtype=float)).sum())
    out = out.drop(columns=["quarter"])
    out = _drop_invalid_and_fill_covariates(out=out, covariate_columns=covariate_columns)

    if metric_kind == "realtime_qoq_saar" or target_transform == "saar_growth":
        target_units = "pct_qoq_saar"
    elif target_transform == "log_level":
        target_units = "log_level"
    else:
        target_units = "level"

    return out, {
        "source": "gdpc1_release_csv",
        "item_id": new_item_id,
        "release_metric": metric_kind,
        "release_stage": stage,
        "release_column": stage_col,
        "target_transform": target_transform,
        "target_units": target_units,
        "release_csv_path": str(csv_path),
        "release_quarters_available": int(len(release_levels_by_quarter)),
        "rows_with_release_target": rows_with_release_target,
    }


def build_gdp_saar_growth_series(
    dataset: Dataset,
    target_series_name: str = "GDP_SAAR",
    source_series_candidates: Sequence[str] | None = None,
    id_col: str = "item_id",
    timestamp_col: str = "timestamp",
    target_col: str = "target",
    include_covariates: bool = False,
    apply_fred_transforms: bool = False,
    fred_transform_codes: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible wrapper for GDP q/q SAAR target construction."""
    return build_real_gdp_target_series(
        dataset=dataset,
        target_series_name=target_series_name,
        target_transform="saar_growth",
        source_series_candidates=source_series_candidates,
        id_col=id_col,
        timestamp_col=timestamp_col,
        target_col=target_col,
        include_covariates=include_covariates,
        apply_fred_transforms=apply_fred_transforms,
        fred_transform_codes=fred_transform_codes,
    )


def export_local_dataset_parquet(
    dataset_df: pd.DataFrame,
    output_path: str | Path,
    covariate_mode: Literal["unprocessed", "processed"] = "unprocessed",
) -> Path:
    """Write fev sequence-format parquet with optional covariates.

    Output schema:
      id, timestamp[list], target[list], [covariate_1[list], ...]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = ["item_id", "timestamp", "target"]
    missing = [c for c in required_columns if c not in dataset_df.columns]
    if missing:
        raise ValueError(f"Dataset dataframe missing required columns: {missing}")

    out = dataset_df.copy()
    resolved_covariate_mode = _normalize_covariate_mode(covariate_mode)
    if _is_sequence_dataset(out):
        rows = _coerce_sequence_rows(out, covariate_mode=resolved_covariate_mode)
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["target"] = pd.to_numeric(out["target"], errors="coerce")
        out = out.dropna(subset=["timestamp", "target"]).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
        rows = _aggregate_long_to_sequence_rows(out, covariate_mode=resolved_covariate_mode)

    if not rows:
        raise ValueError("No rows available after cleaning GDP series; cannot export dataset")

    Dataset.from_list(rows).to_parquet(str(output_path))
    return output_path


def exclude_years(dataset_df: pd.DataFrame, years: Sequence[int] | None = None) -> pd.DataFrame:
    """Exclude observations whose timestamp year is in `years`.

    Supports both long format (timestamp scalar) and sequence format (timestamp list).
    """
    years = sorted({int(y) for y in (years or [])})
    if not years:
        return dataset_df.copy()

    out = dataset_df.copy()

    if _is_sequence_dataset(out):
        rows: list[dict[str, Any]] = []
        for _, rec in out.iterrows():
            ts = pd.to_datetime(pd.Series(rec["timestamp"]), errors="coerce")
            y = pd.to_numeric(pd.Series(rec["target"]), errors="coerce")
            keep = ts.notna() & ~ts.dt.year.isin(years)

            row: dict[str, Any] = {
                "item_id": rec["item_id"],
                "timestamp": ts[keep].tolist(),
                "target": y[keep].astype(float).tolist(),
            }

            for col in [c for c in out.columns if c not in {"item_id", "timestamp", "target"}]:
                cov = _series_like_to_numeric(rec[col], expected_len=len(ts))
                row[col] = cov[keep].astype(float).tolist()

            if row["timestamp"]:
                rows.append(row)

        return pd.DataFrame(rows)

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    keep = out["timestamp"].notna() & ~out["timestamp"].dt.year.isin(years)
    out = out.loc[keep].sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    return out


def reindex_to_regular_frequency(
    dataset_df: pd.DataFrame,
    freq: str = "QS-DEC",
    timestamp_col: str = "timestamp",
    id_col: str = "item_id",
) -> pd.DataFrame:
    """Reassign timestamps to a regular frequency while preserving within-series order."""
    out = dataset_df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out = out.dropna(subset=[timestamp_col])

    if _is_sequence_dataset(out):
        rows: list[dict[str, Any]] = []
        for _, rec in out.iterrows():
            ts = pd.to_datetime(pd.Series(rec[timestamp_col]), errors="coerce")
            ts = ts.dropna().reset_index(drop=True)
            if ts.empty:
                continue
            new_ts = pd.date_range(start=ts.iloc[0], periods=len(ts), freq=freq)

            row: dict[str, Any] = {
                id_col: rec[id_col],
                timestamp_col: list(new_ts.to_pydatetime()),
                "target": rec["target"],
            }
            for col in [c for c in out.columns if c not in {id_col, timestamp_col, "target"}]:
                row[col] = rec[col]
            rows.append(row)
        return pd.DataFrame(rows)

    if id_col not in out.columns:
        out = out.sort_values(timestamp_col).reset_index(drop=True)
        out[timestamp_col] = pd.date_range(start=out[timestamp_col].iloc[0], periods=len(out), freq=freq)
        return out

    groups: list[pd.DataFrame] = []
    for _, grp in out.groupby(id_col, sort=False):
        g = grp.sort_values(timestamp_col).reset_index(drop=True).copy()
        g[timestamp_col] = pd.date_range(start=g[timestamp_col].iloc[0], periods=len(g), freq=freq)
        groups.append(g)

    return pd.concat(groups, axis=0, ignore_index=True)


def build_reindexed_to_actual_timestamp_map(
    actual_df: pd.DataFrame,
    reindexed_df: pd.DataFrame,
    id_col: str = "item_id",
    timestamp_col: str = "timestamp",
) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map reindexed timestamps back to original timestamps by within-series step."""
    actual = actual_df[[id_col, timestamp_col]].copy()
    reindexed = reindexed_df[[id_col, timestamp_col]].copy()

    actual[timestamp_col] = pd.to_datetime(actual[timestamp_col], errors="coerce")
    reindexed[timestamp_col] = pd.to_datetime(reindexed[timestamp_col], errors="coerce")

    actual = actual.dropna(subset=[timestamp_col]).sort_values([id_col, timestamp_col]).reset_index(drop=True)
    reindexed = reindexed.dropna(subset=[timestamp_col]).sort_values([id_col, timestamp_col]).reset_index(drop=True)

    actual["step"] = actual.groupby(id_col, sort=False).cumcount()
    reindexed["step"] = reindexed.groupby(id_col, sort=False).cumcount()

    merged = reindexed.merge(
        actual[[id_col, "step", timestamp_col]].rename(columns={timestamp_col: "actual_timestamp"}),
        on=[id_col, "step"],
        how="left",
    )

    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for _, row in merged.iterrows():
        reindexed_ts = pd.Timestamp(row[timestamp_col])
        actual_ts = pd.Timestamp(row["actual_timestamp"]) if pd.notna(row["actual_timestamp"]) else reindexed_ts
        mapping[reindexed_ts] = actual_ts
    return mapping


def _task_id_column(task: Any) -> str:
    return getattr(task, "id_column", getattr(task, "id_col", "id"))


def _task_timestamp_column(task: Any) -> str:
    return getattr(task, "timestamp_column", getattr(task, "timestamp_col", "timestamp"))


def _task_target_column(task: Any) -> str:
    target = getattr(task, "target", getattr(task, "target_col", "target"))
    if isinstance(target, list):
        if not target:
            raise ValueError("Task target list is empty")
        return str(target[0])
    return str(target)


def _task_covariate_columns(task: Any) -> list[str]:
    known = list(getattr(task, "known_dynamic_columns", []) or [])
    past = list(getattr(task, "past_dynamic_columns", []) or [])
    return list(dict.fromkeys([str(c) for c in [*past, *known]]))


def _extract_past_cutoff_timestamp(past_data: Dataset, task: Any) -> pd.Timestamp:
    ts_col = _task_timestamp_column(task)
    if len(past_data) == 0:
        raise ValueError("past_data is empty; cannot extract cutoff timestamp")

    rec = past_data[0]
    ts = pd.to_datetime(pd.Series(rec.get(ts_col, [])), errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        raise ValueError("past_data has no valid timestamps")

    return pd.Timestamp(ts.iloc[-1])


def _autodiscover_historical_qd_dir(preferred: Path) -> Path | None:
    search_roots: list[Path] = []
    if preferred.parent.exists():
        search_roots.append(preferred.parent.resolve())

    default_root = Path("data/historical").expanduser().resolve()
    if default_root.exists() and default_root not in search_roots:
        search_roots.append(default_root)

    for root in search_roots:
        files = sorted(root.rglob("FRED-QD_*.csv"))
        if not files:
            files = sorted(root.rglob("fred-qd_*.csv"))
        if not files:
            continue

        parent_counts: dict[Path, int] = {}
        for path in files:
            parent = path.parent.resolve()
            parent_counts[parent] = parent_counts.get(parent, 0) + 1

        if parent_counts:
            return max(parent_counts.items(), key=lambda kv: kv[1])[0]

    return None


def _parse_fred_qd_vintage_period(filename: str) -> pd.Period | None:
    match = FRED_QD_VINTAGE_PATTERN.search(filename)
    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        return None

    return pd.Period(f"{year:04d}-{month:02d}", freq="M")


def _looks_like_gdp(name: str) -> bool:
    upper = name.upper()
    tokens = ("GDP", "GDPC1", "RGDP", "REALGDP", "REAL_GDP", "GDP_SAAR", "LOG_REAL_GDP")
    return any(token in upper for token in tokens)


def _build_from_long_format(
    pdf: pd.DataFrame,
    target_series_name: str,
    target_transform: str,
    source_series_candidates: Sequence[str],
    id_col: str,
    timestamp_col: str,
    target_col: str,
    include_covariates: bool,
    apply_fred_transforms: bool,
    fred_transform_codes: dict[str, int] | None,
    covariate_mode: Literal["unprocessed", "processed"],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if timestamp_col not in pdf.columns:
        raise ValueError(f"Long-format dataset must include '{timestamp_col}'")

    work = pdf[[id_col, timestamp_col, target_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col])

    pivot = (
        work.pivot_table(index=timestamp_col, columns=id_col, values=target_col, aggfunc="last")
        .sort_index()
        .rename_axis(index="timestamp", columns=None)
    )
    pivot.columns = [str(c) for c in pivot.columns]

    available_ids = set(pivot.columns)
    computed = False

    if target_series_name in available_ids:
        source_series = target_series_name
        target = pd.to_numeric(pivot[target_series_name], errors="coerce")
    else:
        source_series = _pick_source_series(available_ids=available_ids, preferred=source_series_candidates)
        if source_series is None:
            raise ValueError(
                "Could not find real GDP level series. "
                f"Checked candidates={list(source_series_candidates)}; "
                f"available GDP-like IDs={sorted([s for s in available_ids if _looks_like_gdp(s)])}"
            )
        source = pd.to_numeric(pivot[source_series], errors="coerce")
        target = _transform_from_level(level_series=source, target_transform=target_transform)
        computed = target_transform != "level"

    out = pd.DataFrame(
        {
            "item_id": target_series_name,
            "timestamp": pivot.index,
            "target": target.to_numpy(dtype=float),
        }
    )

    resolved_covariate_mode = _normalize_covariate_mode(covariate_mode)
    use_fred_transforms = resolved_covariate_mode == "processed"

    covariate_columns: list[str] = []
    transformed_covariates: list[str] = []
    if include_covariates:
        excluded = {target_series_name, source_series}
        for cov in pivot.columns:
            if cov in excluded:
                continue
            cov_series = pd.to_numeric(pivot[cov], errors="coerce")
            if use_fred_transforms and fred_transform_codes and cov in fred_transform_codes:
                cov_series = apply_fred_transform_codes(
                    data_df=pd.DataFrame({cov: cov_series}),
                    transform_codes=fred_transform_codes,
                    columns=[cov],
                )[cov]
                transformed_covariates.append(cov)
            out[cov] = cov_series.to_numpy(dtype=float)
            covariate_columns.append(cov)

    out = _drop_invalid_and_fill_covariates(
        out=out,
        covariate_columns=covariate_columns,
        covariate_mode=resolved_covariate_mode,
    )
    return out, {
        "computed": computed,
        "source_series": source_series,
        "target_series": target_series_name,
        "target_transform": target_transform,
        "covariate_columns": covariate_columns,
        "apply_fred_transforms": bool(use_fred_transforms),
        "covariate_mode": resolved_covariate_mode,
        "transformed_covariates": transformed_covariates,
        "transform_code_count": len(fred_transform_codes or {}),
    }


def _build_from_wide_format(
    pdf: pd.DataFrame,
    target_series_name: str,
    target_transform: str,
    source_series_candidates: Sequence[str],
    timestamp_col: str,
    include_covariates: bool,
    apply_fred_transforms: bool,
    fred_transform_codes: dict[str, int] | None,
    covariate_mode: Literal["unprocessed", "processed"],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wide = pdf.copy()
    row = wide.iloc[0] if len(wide) else pd.Series(dtype=object)
    timestamps = _extract_timestamp_vector(row=row, timestamp_col=timestamp_col)
    if timestamps.empty:
        raise ValueError("Could not extract timestamp vector from dataset")

    computed = False
    if target_series_name in wide.columns:
        source_series = target_series_name
        target = _extract_numeric_vector(row=row, column=target_series_name)
    else:
        source_series = _pick_source_series(available_ids=set(map(str, wide.columns)), preferred=source_series_candidates)
        if source_series is None:
            raise ValueError(
                "Could not find GDP target/source in wide-format dataframe columns. "
                f"Checked candidates={list(source_series_candidates)}"
            )
        source = _extract_numeric_vector(row=row, column=source_series)
        target = _transform_from_level(level_series=source, target_transform=target_transform)
        computed = target_transform != "level"

    out = pd.DataFrame({"item_id": target_series_name, "timestamp": timestamps, "target": target})

    resolved_covariate_mode = _normalize_covariate_mode(covariate_mode)
    use_fred_transforms = resolved_covariate_mode == "processed"

    covariate_columns: list[str] = []
    transformed_covariates: list[str] = []
    if include_covariates:
        excluded = {timestamp_col, "id", target_series_name, source_series}
        covariate_data: dict[str, pd.Series] = {}
        for cov in wide.columns:
            if cov in excluded:
                continue
            if _is_numeric_sequence(row.get(cov, None), expected_len=len(timestamps)):
                cov_series = _extract_numeric_vector(row=row, column=cov)
                if use_fred_transforms and fred_transform_codes and cov in fred_transform_codes:
                    cov_series = apply_fred_transform_codes(
                        data_df=pd.DataFrame({cov: cov_series}),
                        transform_codes=fred_transform_codes,
                        columns=[cov],
                    )[cov]
                    transformed_covariates.append(cov)
                covariate_data[cov] = cov_series
                covariate_columns.append(cov)
        if covariate_data:
            out = pd.concat([out, pd.DataFrame(covariate_data)], axis=1)

    out = _drop_invalid_and_fill_covariates(
        out=out,
        covariate_columns=covariate_columns,
        covariate_mode=resolved_covariate_mode,
    )
    return out, {
        "computed": computed,
        "source_series": source_series,
        "target_series": target_series_name,
        "target_transform": target_transform,
        "covariate_columns": covariate_columns,
        "apply_fred_transforms": bool(use_fred_transforms),
        "covariate_mode": resolved_covariate_mode,
        "transformed_covariates": transformed_covariates,
        "transform_code_count": len(fred_transform_codes or {}),
    }


def _pick_source_series(available_ids: set[str], preferred: Sequence[str]) -> str | None:
    upper_to_original = {v.upper(): v for v in available_ids}

    for candidate in preferred:
        direct = upper_to_original.get(candidate.upper())
        if direct is not None:
            return direct

    for candidate in preferred:
        candidate_upper = candidate.upper()
        for series_id in available_ids:
            if candidate_upper in series_id.upper():
                return series_id

    return None


def _normalize_release_stage(stage: str) -> str:
    stage_norm = str(stage).strip().lower()
    aliases = {
        "first_release": "first",
        "second_release": "second",
        "third_release": "third",
        "latest_release": "latest",
    }
    stage_norm = aliases.get(stage_norm, stage_norm)
    if stage_norm not in RELEASE_STAGE_TO_COLUMN:
        raise ValueError(
            f"Unsupported release_stage={stage!r}. "
            f"Supported={sorted(RELEASE_STAGE_TO_COLUMN)}"
        )
    return stage_norm


def _normalize_release_metric(metric: str) -> str:
    metric_norm = str(metric).strip().lower()
    aliases = {
        "realtime_saar": "realtime_qoq_saar",
        "realtime_qoq_saar_pct": "realtime_qoq_saar",
    }
    metric_norm = aliases.get(metric_norm, metric_norm)
    allowed = {"level", "realtime_qoq_saar"}
    if metric_norm not in allowed:
        raise ValueError(
            f"Unsupported release_metric={metric!r}. "
            f"Supported={sorted(allowed)}"
        )
    return metric_norm


def _compute_saar_growth(level_series: pd.Series) -> pd.Series:
    return 100.0 * ((level_series / level_series.shift(1)) ** 4 - 1.0)


def _compute_log_level(level_series: pd.Series) -> pd.Series:
    level = pd.to_numeric(level_series, errors="coerce")
    positive = level.where(level > 0, np.nan)
    return np.log(positive)


def _transform_from_level(level_series: pd.Series, target_transform: str) -> pd.Series:
    if target_transform == "level":
        return pd.to_numeric(level_series, errors="coerce")
    if target_transform == "log_level":
        return _compute_log_level(level_series)
    if target_transform == "saar_growth":
        return _compute_saar_growth(pd.to_numeric(level_series, errors="coerce"))
    raise ValueError(f"Unsupported target_transform={target_transform}")


def _validate_target_transform(target_transform: str) -> None:
    if target_transform not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(
            f"Unsupported target_transform={target_transform}. "
            f"Supported={sorted(SUPPORTED_TARGET_TRANSFORMS)}"
        )


def _extract_timestamp_vector(row: pd.Series, timestamp_col: str) -> pd.Series:
    if timestamp_col in row.index:
        return pd.to_datetime(pd.Series(row[timestamp_col]), errors="coerce")
    return pd.Series(dtype="datetime64[ns]")


def _extract_numeric_vector(row: pd.Series, column: str) -> pd.Series:
    return pd.to_numeric(pd.Series(row[column]), errors="coerce")


def _is_sequence_dataset(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    sample_ts = df.iloc[0]["timestamp"]
    sample_target = df.iloc[0]["target"]
    return isinstance(sample_ts, (list, tuple, np.ndarray, pd.Series)) and isinstance(
        sample_target, (list, tuple, np.ndarray, pd.Series)
    )


def _coerce_sequence_rows(
    df: pd.DataFrame,
    covariate_mode: Literal["unprocessed", "processed"] = "unprocessed",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_cols = [c for c in df.columns if c not in {"item_id", "timestamp", "target"}]
    resolved_covariate_mode = _normalize_covariate_mode(covariate_mode)

    for _, rec in df.iterrows():
        item_id = str(rec["item_id"])
        ts = pd.to_datetime(pd.Series(rec["timestamp"]), errors="coerce")
        y = pd.to_numeric(pd.Series(rec["target"]), errors="coerce")
        mask = ts.notna() & y.notna() & np.isfinite(y.to_numpy(dtype=float))

        ts_clean = ts[mask].tolist()
        y_clean = y[mask].astype(float).tolist()
        if not ts_clean:
            continue

        row: dict[str, Any] = {"id": item_id, "timestamp": ts_clean, "target": y_clean}
        for col in extra_cols:
            cov = _series_like_to_numeric(value=rec[col], expected_len=len(ts))
            cov = _impute_covariate_series(
                cov[mask].astype(float),
                covariate_mode=resolved_covariate_mode,
            )
            row[col] = cov.tolist()

        rows.append(row)

    return rows


def _aggregate_long_to_sequence_rows(
    df: pd.DataFrame,
    covariate_mode: Literal["unprocessed", "processed"] = "unprocessed",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_cols = [c for c in df.columns if c not in {"item_id", "timestamp", "target"}]
    resolved_covariate_mode = _normalize_covariate_mode(covariate_mode)

    for item_id, grp in df.groupby("item_id", sort=False):
        g = grp.sort_values("timestamp")
        row: dict[str, Any] = {
            "id": str(item_id),
            "timestamp": g["timestamp"].tolist(),
            "target": g["target"].astype(float).tolist(),
        }

        for col in extra_cols:
            cov = _impute_covariate_series(
                pd.to_numeric(g[col], errors="coerce").astype(float),
                covariate_mode=resolved_covariate_mode,
            )
            row[col] = cov.tolist()

        rows.append(row)

    return rows


def _drop_invalid_and_fill_covariates(
    out: pd.DataFrame,
    covariate_columns: list[str],
    covariate_mode: Literal["unprocessed", "processed"] = "unprocessed",
) -> pd.DataFrame:
    out = out.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    resolved_covariate_mode = _normalize_covariate_mode(covariate_mode)

    mask = out["timestamp"].notna() & out["target"].notna() & np.isfinite(out["target"].to_numpy(dtype=float))
    out = out.loc[mask].reset_index(drop=True)

    for cov in covariate_columns:
        out[cov] = pd.to_numeric(out[cov], errors="coerce").astype(float)
        out[cov] = _impute_covariate_series(out[cov], covariate_mode=resolved_covariate_mode)

    return out


def _normalize_covariate_mode(mode: str) -> Literal["unprocessed", "processed"]:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in SUPPORTED_COVARIATE_MODES:
        raise ValueError(
            f"Unsupported covariate_mode={mode!r}. "
            f"Supported={sorted(SUPPORTED_COVARIATE_MODES)}"
        )
    return cast(Literal["unprocessed", "processed"], mode_norm)


def _impute_covariate_series(
    series: pd.Series,
    covariate_mode: Literal["unprocessed", "processed"],
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mode = _normalize_covariate_mode(covariate_mode)

    # Avoid hard zero-imputation in dataset construction, especially in processed mode
    # where transformation-induced missingness at tails can create artificial shocks.
    if mode == "processed":
        # TODO: move processed-mode imputation to model adapters (e.g., train-window median)
        # so dataset construction can preserve raw missingness end-to-end.
        return s.ffill().bfill()

    return s.ffill().bfill()


def _series_like_to_numeric(value: Any, expected_len: int) -> pd.Series:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        s = pd.to_numeric(pd.Series(value), errors="coerce")
    else:
        s = pd.to_numeric(pd.Series([value] * expected_len), errors="coerce")

    if len(s) < expected_len:
        if len(s) == 0:
            s = pd.Series([np.nan] * expected_len, dtype=float)
        else:
            pad = pd.Series([s.iloc[-1]] * (expected_len - len(s)), dtype=float)
            s = pd.concat([s, pad], ignore_index=True)
    elif len(s) > expected_len:
        s = s.iloc[:expected_len]

    return s.reset_index(drop=True)


def _is_numeric_sequence(value: Any, expected_len: int) -> bool:
    if not isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return False

    s = pd.to_numeric(pd.Series(value), errors="coerce")
    return len(s) == expected_len and s.notna().any()
