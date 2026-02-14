from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

DEFAULT_SOURCE_SERIES_CANDIDATES: tuple[str, ...] = (
    "GDPC1",
    "RGDP",
    "REALGDP",
    "REAL_GDP",
    "GDP",
)


def load_fev_dataset(
    config: str,
    dataset_path: str = "autogluon/fev_datasets",
    split: str = "train",
    dataset_revision: str | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
) -> Dataset:
    """Load a fev-compatible dataset split from Hugging Face or local parquet."""
    dataset_kwargs = dict(dataset_kwargs or {})

    if str(dataset_path).endswith(".parquet"):
        return load_dataset("parquet", data_files=str(dataset_path), split=split, **dataset_kwargs)

    dataset = load_dataset(
        dataset_path,
        name=config,
        revision=dataset_revision,
        split=split,
        **dataset_kwargs,
    )

    if isinstance(dataset, Dataset):
        return dataset
    if isinstance(dataset, DatasetDict):
        if split in dataset:
            return dataset[split]
        return dataset[next(iter(dataset.keys()))]

    raise TypeError(f"Unexpected dataset type: {type(dataset)}")


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


def build_gdp_saar_growth_series(
    dataset: Dataset,
    target_series_name: str = "GDP_SAAR",
    source_series_candidates: Sequence[str] | None = None,
    id_col: str = "item_id",
    timestamp_col: str = "timestamp",
    target_col: str = "target",
    include_covariates: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build GDP q/q SAAR series and optional aligned covariates in long format.

    If target_series_name exists, it is used directly.
    Otherwise, target is computed from a level series y_t as
        g_t = 100 * ((y_t / y_{t-1})**4 - 1)

    Returned dataframe has columns:
        item_id, timestamp, target, [covariates...]
    """
    source_series_candidates = tuple(source_series_candidates or DEFAULT_SOURCE_SERIES_CANDIDATES)
    pdf = dataset.to_pandas()

    if id_col in pdf.columns and target_col in pdf.columns:
        return _build_from_long_format(
            pdf=pdf,
            target_series_name=target_series_name,
            source_series_candidates=source_series_candidates,
            id_col=id_col,
            timestamp_col=timestamp_col,
            target_col=target_col,
            include_covariates=include_covariates,
        )

    return _build_from_wide_format(
        pdf=pdf,
        target_series_name=target_series_name,
        source_series_candidates=source_series_candidates,
        timestamp_col=timestamp_col,
        include_covariates=include_covariates,
    )


def export_local_dataset_parquet(dataset_df: pd.DataFrame, output_path: str | Path) -> Path:
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
    if _is_sequence_dataset(out):
        rows = _coerce_sequence_rows(out)
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["target"] = pd.to_numeric(out["target"], errors="coerce")
        out = out.dropna(subset=["timestamp", "target"]).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
        rows = _aggregate_long_to_sequence_rows(out)

    if not rows:
        raise ValueError("No rows available after cleaning GDP series; cannot export dataset")

    Dataset.from_list(rows).to_parquet(str(output_path))
    return output_path


def _looks_like_gdp(name: str) -> bool:
    upper = name.upper()
    tokens = ("GDP", "GDPC1", "RGDP", "REALGDP", "REAL_GDP", "GDP_SAAR")
    return any(token in upper for token in tokens)


def _build_from_long_format(
    pdf: pd.DataFrame,
    target_series_name: str,
    source_series_candidates: Sequence[str],
    id_col: str,
    timestamp_col: str,
    target_col: str,
    include_covariates: bool,
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
        target = _compute_saar_growth(pd.to_numeric(pivot[source_series], errors="coerce"))
        computed = True

    out = pd.DataFrame(
        {
            "item_id": target_series_name,
            "timestamp": pivot.index,
            "target": target.to_numpy(dtype=float),
        }
    )

    covariate_columns: list[str] = []
    if include_covariates:
        excluded = {target_series_name, source_series}
        for cov in pivot.columns:
            if cov in excluded:
                continue
            out[cov] = pd.to_numeric(pivot[cov], errors="coerce").to_numpy(dtype=float)
            covariate_columns.append(cov)

    out = _drop_invalid_and_fill_covariates(out=out, covariate_columns=covariate_columns)
    return out, {
        "computed": computed,
        "source_series": source_series,
        "target_series": target_series_name,
        "covariate_columns": covariate_columns,
    }


def _build_from_wide_format(
    pdf: pd.DataFrame,
    target_series_name: str,
    source_series_candidates: Sequence[str],
    timestamp_col: str,
    include_covariates: bool,
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
        target = _compute_saar_growth(source)
        computed = True

    out = pd.DataFrame({"item_id": target_series_name, "timestamp": timestamps, "target": target})

    covariate_columns: list[str] = []
    if include_covariates:
        excluded = {timestamp_col, "id", target_series_name, source_series}
        covariate_data: dict[str, pd.Series] = {}
        for cov in wide.columns:
            if cov in excluded:
                continue
            if _is_numeric_sequence(row.get(cov, None), expected_len=len(timestamps)):
                covariate_data[cov] = _extract_numeric_vector(row=row, column=cov)
                covariate_columns.append(cov)
        if covariate_data:
            out = pd.concat([out, pd.DataFrame(covariate_data)], axis=1)

    out = _drop_invalid_and_fill_covariates(out=out, covariate_columns=covariate_columns)
    return out, {
        "computed": computed,
        "source_series": source_series,
        "target_series": target_series_name,
        "covariate_columns": covariate_columns,
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


def _compute_saar_growth(level_series: pd.Series) -> pd.Series:
    return 100.0 * ((level_series / level_series.shift(1)) ** 4 - 1.0)


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


def _coerce_sequence_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_cols = [c for c in df.columns if c not in {"item_id", "timestamp", "target"}]

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
            cov = cov[mask].astype(float).ffill().bfill().fillna(0.0)
            row[col] = cov.tolist()

        rows.append(row)

    return rows


def _aggregate_long_to_sequence_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_cols = [c for c in df.columns if c not in {"item_id", "timestamp", "target"}]

    for item_id, grp in df.groupby("item_id", sort=False):
        g = grp.sort_values("timestamp")
        row: dict[str, Any] = {
            "id": str(item_id),
            "timestamp": g["timestamp"].tolist(),
            "target": g["target"].astype(float).tolist(),
        }

        for col in extra_cols:
            cov = pd.to_numeric(g[col], errors="coerce").astype(float).ffill().bfill().fillna(0.0)
            row[col] = cov.tolist()

        rows.append(row)

    return rows


def _drop_invalid_and_fill_covariates(out: pd.DataFrame, covariate_columns: list[str]) -> pd.DataFrame:
    out = out.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")

    mask = out["timestamp"].notna() & out["target"].notna() & np.isfinite(out["target"].to_numpy(dtype=float))
    out = out.loc[mask].reset_index(drop=True)

    for cov in covariate_columns:
        out[cov] = pd.to_numeric(out[cov], errors="coerce").astype(float)
        out[cov] = out[cov].ffill().bfill().fillna(0.0)

    return out


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
