from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Literal, Sequence, cast

import pandas as pd

from .fred_transforms import apply_fred_transform_codes, extract_fred_transform_codes

VintagePanelMode = Literal["unprocessed", "processed"]
SUPPORTED_PANEL_MODES = {"unprocessed", "processed"}

_MD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})\.csv$", re.IGNORECASE),
    re.compile(r"^fred[-_]?md_(?P<year>\d{4})m(?P<month>\d{1,2})\.csv$", re.IGNORECASE),
)
_QD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^fred[-_]?qd_(?P<year>\d{4})m(?P<month>\d{1,2})\.csv$", re.IGNORECASE),
    re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})(?:-qd)?\.csv$", re.IGNORECASE),
)


def discover_md_vintage_files(md_dir: str | Path) -> dict[pd.Period, Path]:
    return _discover_vintage_files(root=md_dir, parser=_parse_md_period)


def discover_qd_vintage_files(qd_dir: str | Path) -> dict[pd.Period, Path]:
    return _discover_vintage_files(root=qd_dir, parser=_parse_qd_period)


def build_md_vintage_panel(
    md_dir: str | Path,
    *,
    mode: VintagePanelMode = "unprocessed",
    apply_fred_transforms: bool | None = None,
    exclude_from_transforms: Sequence[str] = ("timestamp",),
    transform_vintage_policy: Literal["per_file", "latest_file"] = "per_file",
) -> pd.DataFrame:
    files = discover_md_vintage_files(md_dir)
    normalized_mode = _normalize_mode(mode)
    do_transforms = _resolve_apply_transforms(normalized_mode, apply_fred_transforms)
    return build_panel_from_files(
        files,
        apply_transforms=do_transforms,
        exclude_from_transforms=exclude_from_transforms,
        transform_vintage_policy=transform_vintage_policy,
    )


def build_qd_vintage_panel(
    qd_dir: str | Path,
    *,
    mode: VintagePanelMode = "unprocessed",
    apply_fred_transforms: bool | None = None,
    exclude_from_transforms: Sequence[str] = ("timestamp", "GDPC1"),
    transform_vintage_policy: Literal["per_file", "latest_file"] = "per_file",
) -> pd.DataFrame:
    files = discover_qd_vintage_files(qd_dir)
    normalized_mode = _normalize_mode(mode)
    do_transforms = _resolve_apply_transforms(normalized_mode, apply_fred_transforms)
    return build_panel_from_files(
        files,
        apply_transforms=do_transforms,
        exclude_from_transforms=exclude_from_transforms,
        transform_vintage_policy=transform_vintage_policy,
    )


def build_panel_from_files(
    vintage_files: dict[pd.Period, Path],
    apply_transforms: bool = False,
    exclude_from_transforms: Sequence[str] = ("timestamp",),
    transform_vintage_policy: Literal["per_file", "latest_file"] = "per_file",
) -> pd.DataFrame:
    if transform_vintage_policy not in {"per_file", "latest_file"}:
        raise ValueError("transform_vintage_policy must be one of {'per_file', 'latest_file'}")

    latest_codes: dict[str, int] = {}
    if apply_transforms and transform_vintage_policy == "latest_file":
        latest_period = sorted(vintage_files)[-1]
        _, latest_codes = _load_data_rows(vintage_files[latest_period], apply_transforms=False)

    frames: list[pd.DataFrame] = []
    for period, csv_path in sorted(vintage_files.items(), key=lambda kv: kv[0]):
        df, file_codes = _load_data_rows(csv_path, apply_transforms=False)
        if df.empty:
            continue

        if apply_transforms:
            transform_codes = latest_codes if transform_vintage_policy == "latest_file" else file_codes
            transform_cols = [c for c in df.columns if c not in set(exclude_from_transforms)]
            df = apply_fred_transform_codes(df, transform_codes=transform_codes, columns=transform_cols)

        vintage_label = f"{period.year:04d}-{period.month:02d}"
        df.insert(0, "vintage", vintage_label)
        df.insert(1, "vintage_timestamp", pd.Timestamp(period.start_time))
        frames.append(df)

    if not frames:
        raise ValueError("No valid vintage data rows found.")

    panel = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    panel = panel.sort_values(["vintage", "timestamp"]).reset_index(drop=True)
    return panel


def _normalize_mode(mode: str) -> VintagePanelMode:
    value = str(mode).strip().lower()
    if value not in SUPPORTED_PANEL_MODES:
        raise ValueError(f"Unsupported mode={mode!r}. Supported={sorted(SUPPORTED_PANEL_MODES)}")
    return cast(VintagePanelMode, value)


def _resolve_apply_transforms(mode: VintagePanelMode, apply_fred_transforms: bool | None) -> bool:
    if apply_fred_transforms is None:
        return mode == "processed"
    return bool(apply_fred_transforms)


def write_panel(df: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif suffix == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {output.suffix}. Use .parquet or .csv")

    return output


def _discover_vintage_files(
    root: str | Path,
    parser: Callable[[str], pd.Period | None],
) -> dict[pd.Period, Path]:
    directory = Path(root).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Vintage directory does not exist: {directory}")

    files: dict[pd.Period, Path] = {}
    for path in sorted(directory.rglob("*.csv")):
        period = parser(path.name)
        if period is None:
            continue
        files[period] = path

    if not files:
        raise FileNotFoundError(f"No recognized vintage CSV files under: {directory}")

    return dict(sorted(files.items(), key=lambda kv: kv[0]))


def _parse_md_period(filename: str) -> pd.Period | None:
    return _parse_period(filename, _MD_PATTERNS)


def _parse_qd_period(filename: str) -> pd.Period | None:
    return _parse_period(filename, _QD_PATTERNS)


def _parse_period(filename: str, patterns: tuple[re.Pattern[str], ...]) -> pd.Period | None:
    for pattern in patterns:
        match = pattern.search(filename)
        if not match:
            continue

        year = int(match.group("year"))
        month = int(match.group("month"))
        if month < 1 or month > 12:
            return None
        return pd.Period(f"{year:04d}-{month:02d}", freq="M")

    return None


def _load_data_rows(csv_path: Path, apply_transforms: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(csv_path)
    if raw.empty:
        return pd.DataFrame(columns=["timestamp"]), {}

    first_col = str(raw.columns[0])
    transform_codes = extract_fred_transform_codes(raw_df=raw, first_col_name=first_col)
    # FRED vintage files use U.S. month/day/year dates for observation rows.
    date_values = pd.to_datetime(raw[first_col], format="%m/%d/%Y", errors="coerce")
    data = raw.loc[date_values.notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=["timestamp"]), transform_codes

    data["timestamp"] = pd.to_datetime(data[first_col], format="%m/%d/%Y", errors="coerce")
    data = data.drop(columns=[first_col])

    for col in data.columns:
        if col == "timestamp":
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_values("timestamp").reset_index(drop=True)
    ordered_cols = ["timestamp"] + [c for c in data.columns if c != "timestamp"]
    out = data[ordered_cols]
    if apply_transforms:
        transform_cols = [c for c in out.columns if c != "timestamp"]
        out = apply_fred_transform_codes(out, transform_codes=transform_codes, columns=transform_cols)
    return out, transform_codes
