from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

SUPPORTED_FRED_TCODES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)


def fred_transform(series: pd.Series | Sequence[float], tcode: int) -> pd.Series:
    """Apply FRED-MD/QD transform code to one series.

    Ported from FredMDQD.jl (`src/transforms.jl`):
    https://github.com/enweg/FredMDQD.jl
    """
    if tcode not in SUPPORTED_FRED_TCODES:
        raise ValueError(f"Unsupported FRED transform code: {tcode}. Supported={SUPPORTED_FRED_TCODES}")

    x = pd.to_numeric(pd.Series(series), errors="coerce").astype(float)

    if tcode == 1:
        return x
    if tcode == 2:
        return x.diff(1)
    if tcode == 3:
        return x.diff(1).diff(1)
    if tcode == 4:
        return np.log(x.where(x > 0.0, np.nan))
    if tcode == 5:
        return np.log(x.where(x > 0.0, np.nan)).diff(1)
    if tcode == 6:
        return np.log(x.where(x > 0.0, np.nan)).diff(1).diff(1)

    # tcode == 7: Delta(x_t / x_{t-1} - 1)
    growth = x / x.shift(1) - 1.0
    return growth.diff(1)


def extract_fred_transform_codes(
    raw_df: pd.DataFrame,
    first_col_name: str | None = None,
) -> dict[str, int]:
    """Extract per-series FRED transform codes from metadata row in a raw CSV dataframe."""
    if raw_df.empty:
        return {}

    date_or_key_col = first_col_name or str(raw_df.columns[0])
    if date_or_key_col not in raw_df.columns:
        return {}

    key = raw_df[date_or_key_col].astype(str).str.strip().str.lower()
    mask = key.isin({"transform", "transformation", "tcode", "tcodes"})
    if not mask.any():
        return {}

    row = raw_df.loc[mask].iloc[0]
    codes: dict[str, int] = {}
    for col in raw_df.columns:
        if col == date_or_key_col:
            continue
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.isna(val):
            continue
        code = int(val)
        if code in SUPPORTED_FRED_TCODES:
            codes[str(col)] = code

    return codes


def apply_fred_transform_codes(
    data_df: pd.DataFrame,
    transform_codes: Mapping[str, int] | None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply FRED transform codes to selected columns in a timestamp-aligned dataframe."""
    if not transform_codes:
        return data_df.copy()

    out = data_df.copy()

    if columns is None:
        target_columns = [c for c in out.columns if c in transform_codes]
    else:
        target_columns = [c for c in columns if c in out.columns and c in transform_codes]

    for col in target_columns:
        code = int(transform_codes[col])
        if code not in SUPPORTED_FRED_TCODES:
            continue
        out[col] = fred_transform(out[col], code)

    return out
