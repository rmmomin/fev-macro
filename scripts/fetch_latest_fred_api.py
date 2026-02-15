#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_MD_TEMPLATE = "data/historical/md/vintages_1999_2026/2026-01.csv"
DEFAULT_QD_TEMPLATE = "data/historical/qd/vintages_2018_2026/FRED-QD_2026m1.csv"
DEFAULT_OUTPUT_DIR = "data/latest"
USER_AGENT = "fev-macro-fred-fetch/1.0"
FRED_PAGE_LIMIT_MAX = 100_000
FRED_EARLIEST_DATE = "1776-07-04"
FRED_FAR_FUTURE_DATE = "9999-12-31"

MANUAL_SERIES_ALIASES: dict[str, tuple[str, ...]] = {
    "CLAIMSx": ("ICSA",),
    "S&P 500": ("SP500",),
    "S&P div yield": ("SPDIVY",),
    "S&P PE ratio": ("SP500PE", "SPEARN"),
}


@dataclass
class TemplateSchema:
    path: Path
    first_col: str
    series_columns: list[str]
    metadata_rows: pd.DataFrame
    data_rows: pd.DataFrame


@dataclass
class FetchResult:
    variable_name: str
    resolved_series_id: str | None
    series: pd.Series | None
    error: str | None


class FredAPIError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch latest FRED observations for MD/QD template variables and write refreshed CSVs under data/latest/."
        )
    )
    parser.add_argument("--env_file", type=str, default=".env", help="Path to dotenv file containing FRED_API_KEY.")
    parser.add_argument("--api_key", type=str, default=None, help="Optional explicit API key override.")
    parser.add_argument("--md_template", type=str, default=DEFAULT_MD_TEMPLATE)
    parser.add_argument("--qd_template", type=str, default=DEFAULT_QD_TEMPLATE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--md_output_name", type=str, default="fred_md_latest.csv")
    parser.add_argument("--qd_output_name", type=str, default="fred_qd_latest.csv")
    parser.add_argument("--coverage_report_name", type=str, default="fred_latest_coverage_gaps.json")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--timeout_seconds", type=float, default=30.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_backoff_seconds", type=float, default=1.5)
    parser.add_argument(
        "--page_limit",
        type=int,
        default=FRED_PAGE_LIMIT_MAX,
        help=f"FRED observations pagination page size (max {FRED_PAGE_LIMIT_MAX}).",
    )
    parser.add_argument(
        "--retry_rate_limited_passes",
        type=int,
        default=1,
        help="Number of extra sequential retry passes for variables that failed with HTTP 429.",
    )
    parser.add_argument(
        "--retry_rate_limited_sleep_seconds",
        type=float,
        default=3.0,
        help="Delay between sequential retries in the rate-limit retry pass.",
    )
    parser.add_argument(
        "--series_limit",
        type=int,
        default=None,
        help="Optional cap (for debugging) on number of variables fetched from each template.",
    )
    parser.add_argument(
        "--fail_on_unresolved",
        action="store_true",
        help="Exit non-zero if any series cannot be resolved/fetched.",
    )
    return parser.parse_args()


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return str(args.api_key).strip()

    from_env = str(os.getenv("FRED_API_KEY", "")).strip()
    if from_env:
        return from_env

    env_file = Path(args.env_file).expanduser().resolve()
    parsed = _load_dotenv(env_file)
    from_file = str(parsed.get("FRED_API_KEY", "")).strip()
    if from_file:
        os.environ["FRED_API_KEY"] = from_file
        return from_file

    raise ValueError(
        "Missing FRED_API_KEY. Set it in environment, pass --api_key, "
        f"or add it to {env_file}."
    )


def read_template_schema(path: str | Path) -> TemplateSchema:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Template CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, dtype=str)
    if raw.empty:
        raise ValueError(f"Template CSV is empty: {csv_path}")

    first_col = str(raw.columns[0])
    series_columns = [str(c) for c in raw.columns[1:] if str(c) and not str(c).startswith("Unnamed:")]

    parsed_dates = pd.to_datetime(raw[first_col], format="%m/%d/%Y", errors="coerce")
    metadata_rows = raw.loc[parsed_dates.isna()].copy()
    metadata_rows = metadata_rows.dropna(how="all").reset_index(drop=True)
    data_rows = raw.loc[parsed_dates.notna()].copy().reset_index(drop=True)
    if not data_rows.empty:
        data_rows[first_col] = parsed_dates.loc[parsed_dates.notna()].values

    keep_cols = [first_col, *series_columns]
    for col in keep_cols:
        if col not in metadata_rows.columns:
            metadata_rows[col] = ""
    metadata_rows = metadata_rows[keep_cols]
    for col in keep_cols:
        if col not in data_rows.columns:
            data_rows[col] = pd.NA
    data_rows = data_rows[keep_cols]

    return TemplateSchema(
        path=csv_path,
        first_col=first_col,
        series_columns=series_columns,
        metadata_rows=metadata_rows,
        data_rows=data_rows,
    )


def _candidate_series_ids(variable_name: str) -> list[str]:
    candidates: list[str] = []

    manual = MANUAL_SERIES_ALIASES.get(variable_name)
    if manual:
        candidates.extend(list(manual))

    candidates.append(variable_name)

    if variable_name.endswith("x") and len(variable_name) > 1:
        candidates.append(variable_name[:-1])

    if " " in variable_name:
        candidates.append(variable_name.replace(" ", ""))

    if variable_name.endswith("x") and " " in variable_name:
        candidates.append(variable_name[:-1].replace(" ", ""))

    deduped: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        c = str(cand).strip()
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    return deduped


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%m/%d/%Y", errors="coerce")
    if parsed.notna().sum() > 0:
        return parsed
    return pd.to_datetime(values, errors="coerce")


def _prepare_numeric_panel(df: pd.DataFrame, date_col: str, series_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[date_col, *series_columns])

    working = df.copy()
    missing_cols = [col for col in series_columns if col not in working.columns]
    if missing_cols:
        missing_df = pd.DataFrame({col: pd.NA for col in missing_cols}, index=working.index)
        working = pd.concat([working, missing_df], axis=1)

    parsed_dates = _parse_dates(working[date_col])
    out = working.loc[parsed_dates.notna(), [date_col, *series_columns]].copy()
    if out.empty:
        return pd.DataFrame(columns=[date_col, *series_columns])

    out[date_col] = parsed_dates.loc[parsed_dates.notna()].values
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    for col in series_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _iso_date_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    return str(ts.date())


def _infer_period_frequency(dates: pd.Series) -> str:
    if dates.empty:
        return "M"
    unique_dates = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
    if len(unique_dates) < 3:
        return "M"
    diffs_days = unique_dates.diff().dropna().dt.days
    if diffs_days.empty:
        return "M"
    median_diff = float(diffs_days.median())
    if median_diff >= 80.0:
        return "Q"
    return "M"


def _presence_mask_by_period(df: pd.DataFrame, date_col: str, series_columns: list[str], period_freq: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=series_columns)
    tmp = df[[date_col, *series_columns]].copy()
    tmp["_period"] = tmp[date_col].dt.to_period(period_freq)
    mask = tmp[series_columns].notna()
    mask.index = tmp["_period"]
    grouped = mask.groupby(level=0).any().sort_index()
    return grouped


def _period_to_timestamp(period_index: pd.PeriodIndex, period_freq: str) -> pd.DatetimeIndex:
    # Always emit month-end timestamps for aligned outputs.
    month_periods = period_index.asfreq("M", how="end")
    return pd.DatetimeIndex(month_periods.to_timestamp(how="end").normalize())


def _normalize_dates_to_month_end(
    df: pd.DataFrame,
    date_col: str,
    period_freq: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    periods = dates.dt.to_period(period_freq)
    month_periods = periods.dt.asfreq("M", how="end")
    out[date_col] = month_periods.dt.to_timestamp(how="end").dt.normalize()
    out = out.loc[out[date_col].notna()].sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    return out.reset_index(drop=True)


def _align_series_to_template_frequency(series: pd.Series, period_freq: str) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float, name=getattr(series, "name", None))

    aligned = pd.to_numeric(series, errors="coerce")
    aligned.index = pd.to_datetime(aligned.index, errors="coerce")
    aligned = aligned.loc[aligned.index.notna()].sort_index().dropna()
    if aligned.empty:
        return pd.Series(dtype=float, name=getattr(series, "name", None))

    grouped = aligned.groupby(aligned.index.to_period(period_freq)).mean()
    grouped.index = _period_to_timestamp(grouped.index, period_freq)
    grouped = grouped.sort_index()
    grouped.name = getattr(series, "name", None)
    return grouped.astype(float)


def _fetch_dependency_series(
    series_id: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    page_limit: int,
    period_freq: str,
    cache: dict[str, pd.Series],
) -> tuple[pd.Series | None, str | None]:
    if series_id in cache:
        return cache[series_id], None
    try:
        fetched = _fetch_observations_for_series_id(
            series_id=series_id,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            page_limit=page_limit,
        )
        aligned = _align_series_to_template_frequency(fetched, period_freq)
        cache[series_id] = aligned
        return aligned, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _construct_missing_series(
    requested_vars: list[str],
    template_history: pd.DataFrame,
    date_col: str,
    unresolved_map: dict[str, str],
    resolved_map: dict[str, str],
    series_by_var: dict[str, pd.Series],
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    page_limit: int,
    period_freq: str,
    label: str,
) -> None:
    dependency_cache: dict[str, pd.Series] = {}
    template_numeric = _prepare_numeric_panel(template_history, date_col, requested_vars)
    if not template_numeric.empty:
        template_numeric = _normalize_dates_to_month_end(template_numeric, date_col, period_freq)
        template_numeric = template_numeric.set_index(date_col)
    else:
        template_numeric = pd.DataFrame(columns=requested_vars)

    def calibrate_to_template(
        target_var: str,
        expr: pd.Series,
        description: str,
    ) -> tuple[pd.Series | None, str | None, float]:
        expr_clean = pd.to_numeric(expr, errors="coerce")
        expr_clean.index = pd.to_datetime(expr_clean.index, errors="coerce")
        expr_clean = expr_clean.loc[expr_clean.index.notna()].sort_index()
        expr_clean = expr_clean.where(expr_clean.replace([float("inf"), float("-inf")], pd.NA).notna())
        expr_clean = expr_clean.dropna()
        expr_clean = _align_series_to_template_frequency(expr_clean, period_freq).dropna()
        if expr_clean.empty:
            return None, f"{description}: expression produced no usable values", float("inf")

        if target_var not in template_numeric.columns:
            return expr_clean, f"{description} [no template target to calibrate]", float("inf")

        target = pd.to_numeric(template_numeric[target_var], errors="coerce").dropna()
        if target.empty:
            return expr_clean, f"{description} [template target empty; no calibration]", float("inf")

        common_idx = target.index.intersection(expr_clean.index)
        if len(common_idx) < 8:
            return expr_clean, f"{description} [insufficient overlap={len(common_idx)}; no calibration]", float("inf")

        target_common = target.loc[common_idx]
        expr_common = expr_clean.loc[common_idx]
        valid = target_common.notna() & expr_common.notna() & (expr_common != 0.0) & (target_common != 0.0)
        if int(valid.sum()) < 8:
            return expr_clean, f"{description} [insufficient valid overlap={int(valid.sum())}; no calibration]", float("inf")

        t = target_common.loc[valid]
        e = expr_common.loc[valid]
        scale_candidates = [1.0]
        ratio = t / e
        ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).dropna()
        if not ratio.empty:
            scale_candidates.append(float(ratio.median()))

        best_scale = 1.0
        best_mape = float("inf")
        for scale in scale_candidates:
            pred = e * scale
            mape = float(((pred - t).abs() / t.abs()).median() * 100.0)
            if mape < best_mape:
                best_mape = mape
                best_scale = float(scale)

        calibrated = expr_clean * best_scale
        detail = (
            f"{description} [calibrated_scale={best_scale:.10g}, "
            f"overlap_n={int(valid.sum())}, overlap_mape={best_mape:.4f}%]"
        )
        return calibrated, detail, float(best_mape)

    def ensure_series(candidates: list[str]) -> tuple[pd.Series | None, str | None]:
        errors: list[str] = []
        for cand in candidates:
            if cand in series_by_var and series_by_var[cand] is not None:
                return series_by_var[cand], None
        for cand in candidates:
            if cand in dependency_cache and dependency_cache[cand] is not None:
                return dependency_cache[cand], None
        for cand in candidates:
            fetched, err = _fetch_dependency_series(
                series_id=cand,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                page_limit=page_limit,
                period_freq=period_freq,
                cache=dependency_cache,
            )
            if fetched is not None and not fetched.empty:
                return fetched, None
            if err:
                errors.append(f"{cand}: {err}")
        if errors:
            return None, " | ".join(errors)
        return None, "No candidates provided"

    def select_best_calibrated(
        target_var: str,
        candidate_exprs: list[tuple[str, pd.Series | None]],
    ) -> tuple[pd.Series | None, str | None, str | None]:
        best_series: pd.Series | None = None
        best_detail: str | None = None
        best_score = float("inf")
        errors: list[str] = []

        for desc, expr in candidate_exprs:
            if expr is None:
                errors.append(f"{desc}: expression missing")
                continue
            calibrated, detail, score = calibrate_to_template(target_var, expr, desc)
            if calibrated is None:
                if detail:
                    errors.append(detail)
                continue
            candidate_score = score if score == score else float("inf")
            if best_series is None or candidate_score < best_score:
                best_series = calibrated
                best_detail = detail
                best_score = candidate_score

        if best_series is not None:
            return best_series, best_detail, None
        if errors:
            return None, None, " | ".join(errors)
        return None, None, "No candidate expressions provided"

    for var in [v for v in requested_vars if v in unresolved_map]:
        constructed: pd.Series | None = None
        detail = ""
        build_error: str | None = None

        if var == "CLAIMSx":
            claims, build_error = ensure_series(["CLAIMSx", "ICSA"])
            if claims is not None:
                constructed = claims
                detail = "constructed: ICSA aggregated to template frequency"

        elif var in {"HWI", "HWIx"}:
            jts, err_jts = ensure_series(["JTSJOL"])
            old_hwi, err_old = ensure_series(["M0882BUSM350NNBR"])
            best_hwi, best_detail, sel_err = select_best_calibrated(
                var,
                [
                    ("constructed: JTSJOL", jts),
                    ("constructed: M0882BUSM350NNBR", old_hwi),
                ],
            )
            base_template = None
            if var in template_numeric.columns:
                base_template = pd.to_numeric(template_numeric[var], errors="coerce").dropna()
            if best_hwi is not None:
                if base_template is not None and not base_template.empty:
                    stitched = pd.concat([base_template, best_hwi], axis=0).groupby(level=0).last().sort_index()
                    constructed = stitched
                    detail = f"{best_detail}; stitched with template history"
                else:
                    constructed = best_hwi
                    detail = best_detail or "constructed: help-wanted proxy"
            else:
                build_error = " | ".join([msg for msg in [err_jts, err_old, sel_err] if msg])

        elif var in {"HWIURATIO", "HWIURATIOx"}:
            hwi_var = "HWIx" if var.endswith("x") else "HWI"
            hwi_series = series_by_var.get(hwi_var)
            if hwi_series is None:
                jts, _ = ensure_series(["JTSJOL"])
                if jts is not None:
                    hwi_series, _, _ = calibrate_to_template(
                        hwi_var,
                        jts,
                        f"constructed: JTSJOL proxy for {hwi_var}",
                    )
            unemp, err_unemp = ensure_series(["UNEMPLOY"])
            if hwi_series is not None and unemp is not None:
                expr = hwi_series / unemp.where(unemp != 0.0)
                best_ratio, best_detail, sel_err = select_best_calibrated(
                    var,
                    [(f"constructed: {hwi_var} / UNEMPLOY", expr)],
                )
                if best_ratio is not None:
                    base_template = None
                    if var in template_numeric.columns:
                        base_template = pd.to_numeric(template_numeric[var], errors="coerce").dropna()
                    if base_template is not None and not base_template.empty:
                        stitched = pd.concat([base_template, best_ratio], axis=0).groupby(level=0).last().sort_index()
                        constructed = stitched
                        detail = f"{best_detail}; stitched with template history"
                    else:
                        constructed = best_ratio
                        detail = best_detail or "constructed: HWI / UNEMPLOY"
                else:
                    build_error = sel_err
            else:
                build_error = "Missing inputs for HWIURATIO construction"
                if err_unemp:
                    build_error = f"{build_error} | {err_unemp}"

        elif var in {"CONSPI", "CONSPIx"}:
            nonrev, err_nonrev = ensure_series(["NONREVSL", "NONREVSLx"])
            candidate_exprs: list[tuple[str, pd.Series | None]] = []
            if nonrev is not None:
                if var == "CONSPI":
                    rpi, err_rpi = ensure_series(["RPI"])
                    pcepi, err_pcepi = ensure_series(["PCEPI"])
                    if rpi is not None and pcepi is not None:
                        denom = rpi * pcepi / 100.0
                        denom = denom.where(denom != 0.0)
                        candidate_exprs.append(("constructed: NONREVSL / (RPI * PCEPI / 100)", nonrev / denom))
                    else:
                        build_error = " | ".join([msg for msg in [err_rpi, err_pcepi] if msg])
                else:
                    dpi, err_dpi = ensure_series(["DPI"])
                    dpic, err_dpic = ensure_series(["DPIC96"])
                    if dpi is not None:
                        candidate_exprs.append(("constructed: NONREVSLx / DPI", nonrev / dpi.where(dpi != 0.0)))
                    if dpic is not None:
                        candidate_exprs.append(("constructed: NONREVSLx / DPIC96", nonrev / dpic.where(dpic != 0.0)))
                    if not candidate_exprs:
                        build_error = " | ".join([msg for msg in [err_dpi, err_dpic] if msg])
            else:
                build_error = err_nonrev

            if candidate_exprs:
                best_series, best_detail, sel_err = select_best_calibrated(var, candidate_exprs)
                if best_series is not None:
                    constructed = best_series
                    detail = best_detail or "constructed: CONSPI proxy"
                else:
                    build_error = " | ".join([msg for msg in [build_error, sel_err] if msg])

        elif var == "AMDMNOx":
            acog, err_acog = ensure_series(["ACOGNO", "ACOGNOx"])
            anden, err_anden = ensure_series(["ANDENO", "ANDENOx"])
            if acog is not None and anden is not None:
                best_series, best_detail, sel_err = select_best_calibrated(
                    var,
                    [("constructed: ACOGNO + ANDENO", acog + anden)],
                )
                if best_series is not None:
                    constructed = best_series
                    detail = best_detail or "constructed: durable orders proxy"
                else:
                    build_error = sel_err
            else:
                build_error = " | ".join([msg for msg in [err_acog, err_anden] if msg])

        elif var in {"LIABPIx", "NWPIx"}:
            dpi, err_dpi = ensure_series(["DPI"])
            dpic, err_dpic = ensure_series(["DPIC96"])
            num_map = {
                "LIABPIx": ["TLBSHNOx", "TLBSHNO"],
                "NWPIx": ["TNWBSHNOx", "TNWBSHNO"],
            }
            num, err_num = ensure_series(num_map[var])
            if num is not None:
                candidate_exprs: list[tuple[str, pd.Series | None]] = []
                if dpi is not None:
                    candidate_exprs.append((f"constructed: 100 * {num_map[var][0]} / DPI", 100.0 * num / dpi.where(dpi != 0.0)))
                if dpic is not None:
                    candidate_exprs.append((f"constructed: 100 * {num_map[var][0]} / DPIC96", 100.0 * num / dpic.where(dpic != 0.0)))
                if not candidate_exprs:
                    build_error = " | ".join([msg for msg in [err_dpi, err_dpic] if msg])
                    continue
                best_series, best_detail, sel_err = select_best_calibrated(
                    var,
                    candidate_exprs,
                )
                if best_series is not None:
                    constructed = best_series
                    detail = best_detail or f"constructed: {var} from personal-income denominator"
                else:
                    build_error = sel_err
            else:
                build_error = err_num

        elif var == "TARESAx":
            tfa, err_tfa = ensure_series(["TFAABSHNOx", "TFAABSHNO"])
            pce_core, _ = ensure_series(["PCEPILFE"])
            candidate_exprs: list[tuple[str, pd.Series | None]] = []
            if tfa is not None:
                candidate_exprs.append(("constructed: scaled TFAABSHNOx", tfa))
                if pce_core is not None:
                    denom = pce_core.where(pce_core != 0.0)
                    candidate_exprs.append(("constructed: 100 * TFAABSHNOx / PCEPILFE", 100.0 * tfa / denom))
            if candidate_exprs:
                best_series, best_detail, sel_err = select_best_calibrated(var, candidate_exprs)
                if best_series is not None:
                    constructed = best_series
                    detail = best_detail or "constructed: TARESA proxy"
                else:
                    build_error = sel_err
            else:
                build_error = err_tfa

        elif var in {"TLBSNNCBBDIx", "TNWMVBSNNCBBDIx", "TLBSNNBBDIx", "TNWBSNNBBDIx"}:
            num_map = {
                "TLBSNNCBBDIx": ["TLBSNNCBx", "TLBSNNCB"],
                "TNWMVBSNNCBBDIx": ["TNWMVBSNNCBx", "TNWMVBSNNCB"],
                "TLBSNNBBDIx": ["TLBSNNBx", "TLBSNNB"],
                "TNWBSNNBBDIx": ["TNWBSNNBx", "TNWBSNNB"],
            }
            num, err_num = ensure_series(num_map[var])
            denom_ids = [
                "BOGZ1FA106012005Q",
                "BOGZ1FA106012095Q",
                "BOGZ1FU106012005Q",
                "A464RC1Q027SBEA",
            ]
            if num is not None:
                candidate_exprs: list[tuple[str, pd.Series | None]] = []
                for denom_id in denom_ids:
                    denom, _ = ensure_series([denom_id])
                    if denom is None:
                        continue
                    denom_safe = denom.where(denom != 0.0)
                    candidate_exprs.append((f"constructed: 100 * {num_map[var][0]} / {denom_id}", 100.0 * num / denom_safe))
                if candidate_exprs:
                    best_series, best_detail, sel_err = select_best_calibrated(var, candidate_exprs)
                    if best_series is not None:
                        constructed = best_series
                        detail = best_detail or "constructed: business-income ratio proxy"
                    else:
                        build_error = sel_err
                else:
                    build_error = "No denominator series available for business-income ratio construction"
            else:
                build_error = err_num

        elif var in {"COMPAPFFx", "COMPAPFF"}:
            cp3m, err_cp = ensure_series(["CP3Mx", "CP3M"])
            ff, err_ff = ensure_series(["FEDFUNDS"])
            if cp3m is not None and ff is not None:
                constructed = cp3m - ff
                detail = "constructed: CP3M minus FEDFUNDS"
            else:
                build_error = " | ".join([msg for msg in [err_cp, err_ff] if msg])

        elif var == "MORTG10YRx":
            m30, err_m = ensure_series(["MORTGAGE30US"])
            g10, err_g = ensure_series(["GS10"])
            if m30 is not None and g10 is not None:
                constructed = m30 - g10
                detail = "constructed: MORTGAGE30US minus GS10"
            else:
                build_error = " | ".join([msg for msg in [err_m, err_g] if msg])

        elif var == "TB6M3Mx":
            tb6, err_tb6 = ensure_series(["TB6MS"])
            tb3, err_tb3 = ensure_series(["TB3MS"])
            if tb6 is not None and tb3 is not None:
                constructed = tb6 - tb3
                detail = "constructed: TB6MS minus TB3MS"
            else:
                build_error = " | ".join([msg for msg in [err_tb6, err_tb3] if msg])

        elif var == "GS1TB3Mx":
            gs1, err_gs1 = ensure_series(["GS1"])
            tb3, err_tb3 = ensure_series(["TB3MS"])
            if gs1 is not None and tb3 is not None:
                constructed = gs1 - tb3
                detail = "constructed: GS1 minus TB3MS"
            else:
                build_error = " | ".join([msg for msg in [err_gs1, err_tb3] if msg])

        elif var == "GS10TB3Mx":
            gs10, err_gs10 = ensure_series(["GS10"])
            tb3, err_tb3 = ensure_series(["TB3MS"])
            if gs10 is not None and tb3 is not None:
                constructed = gs10 - tb3
                detail = "constructed: GS10 minus TB3MS"
            else:
                build_error = " | ".join([msg for msg in [err_gs10, err_tb3] if msg])

        elif var == "CPF3MTB3Mx":
            cp3m, err_cp = ensure_series(["CP3M", "CP3Mx"])
            tb3, err_tb3 = ensure_series(["TB3MS"])
            if cp3m is not None and tb3 is not None:
                constructed = cp3m - tb3
                detail = "constructed: CP3M minus TB3MS"
            else:
                build_error = " | ".join([msg for msg in [err_cp, err_tb3] if msg])

        elif var == "BOGMBASEREALx":
            bogm, err_bogm = ensure_series(["BOGMBASE"])
            cpi, err_cpi = ensure_series(["CPIAUCSL"])
            if bogm is not None and cpi is not None:
                safe_cpi = cpi.copy()
                safe_cpi[safe_cpi == 0.0] = pd.NA
                constructed = 100.0 * bogm / safe_cpi
                detail = "constructed: 100 * BOGMBASE / CPIAUCSL"
            else:
                build_error = " | ".join([msg for msg in [err_bogm, err_cpi] if msg])

        elif var == "UNRATELTx":
            ce16, err_ce = ensure_series(["CE16OV"])
            unrate, err_ur = ensure_series(["UNRATE"])
            u27, err_u27 = ensure_series(["UEMP27OV"])
            if ce16 is not None and unrate is not None and u27 is not None:
                lf = ce16 / (1.0 - unrate / 100.0)
                lf = lf.where(lf != 0.0)
                constructed = 100.0 * u27 / lf
                detail = "constructed: 100 * UEMP27OV / (CE16OV / (1 - UNRATE/100))"
            else:
                build_error = " | ".join([msg for msg in [err_ce, err_ur, err_u27] if msg])

        elif var == "UNRATESTx":
            ce16, err_ce = ensure_series(["CE16OV"])
            unrate, err_ur = ensure_series(["UNRATE"])
            u5, err_u5 = ensure_series(["UEMPLT5"])
            u14, err_u14 = ensure_series(["UEMP5TO14"])
            u26, err_u26 = ensure_series(["UEMP15T26"])
            if ce16 is not None and unrate is not None and u5 is not None and u14 is not None and u26 is not None:
                lf = ce16 / (1.0 - unrate / 100.0)
                lf = lf.where(lf != 0.0)
                constructed = 100.0 * (u5 + u14 + u26) / lf
                detail = "constructed: 100 * (UEMPLT5 + UEMP5TO14 + UEMP15T26) / (CE16OV / (1 - UNRATE/100))"
            else:
                build_error = " | ".join([msg for msg in [err_ce, err_ur, err_u5, err_u14, err_u26] if msg])

        if constructed is not None:
            cleaned = pd.to_numeric(constructed, errors="coerce")
            cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
            cleaned = cleaned.loc[cleaned.index.notna()].dropna().sort_index()
            cleaned = _align_series_to_template_frequency(cleaned, period_freq).dropna()
            if not cleaned.empty:
                series_by_var[var] = cleaned
                resolved_map[var] = detail
                unresolved_map.pop(var, None)
                print(f"[{label}] CONSTRUCT OK: {var} ({len(cleaned)} obs) [{detail}]")
                continue
            build_error = "Constructed series was empty after cleaning"

        if build_error:
            unresolved_map[var] = f"{unresolved_map[var]} | construct: {build_error}"


def compute_coverage_report(
    schema: TemplateSchema,
    fetched_full_df: pd.DataFrame,
    unresolved_map: dict[str, str],
) -> dict[str, object]:
    template_df = _prepare_numeric_panel(schema.data_rows, schema.first_col, schema.series_columns)
    fetched_df = _prepare_numeric_panel(fetched_full_df, schema.first_col, schema.series_columns)
    period_freq = _infer_period_frequency(
        template_df[schema.first_col] if not template_df.empty else pd.Series(dtype="datetime64[ns]")
    )
    template_df = _normalize_dates_to_month_end(template_df, schema.first_col, period_freq)
    fetched_df = _normalize_dates_to_month_end(fetched_df, schema.first_col, period_freq)

    template_series_counts = template_df[schema.series_columns].notna().sum() if not template_df.empty else pd.Series(dtype=int)
    fetched_series_counts = fetched_df[schema.series_columns].notna().sum() if not fetched_df.empty else pd.Series(dtype=int)

    series_missing_in_fetched = [
        col
        for col in schema.series_columns
        if int(template_series_counts.get(col, 0)) > 0 and int(fetched_series_counts.get(col, 0)) == 0
    ]

    common_dates_count = 0
    template_non_missing_points = 0
    matched_non_missing_points = 0
    missing_non_missing_points = 0
    coverage_pct: float | None = None
    low_coverage_series: list[dict[str, object]] = []
    common_dates_min = None
    common_dates_max = None

    common_periods_count = 0
    common_periods_min = None
    common_periods_max = None
    template_period_non_missing_points = 0
    matched_period_non_missing_points = 0
    missing_period_non_missing_points = 0
    period_coverage_pct: float | None = None
    low_coverage_period_series: list[dict[str, object]] = []

    if not template_df.empty and not fetched_df.empty:
        template_idx = template_df.set_index(schema.first_col)
        fetched_idx = fetched_df.set_index(schema.first_col)
        common_dates = template_idx.index.intersection(fetched_idx.index).sort_values()
        common_dates_count = int(len(common_dates))

        if common_dates_count > 0:
            common_dates_min = _iso_date_or_none(common_dates.min())
            common_dates_max = _iso_date_or_none(common_dates.max())

            template_common = template_idx.loc[common_dates, schema.series_columns]
            fetched_common = fetched_idx.loc[common_dates, schema.series_columns]

            template_mask = template_common.notna()
            fetched_mask = fetched_common.notna()

            template_non_missing_points = int(template_mask.to_numpy().sum())
            matched_non_missing_points = int((template_mask & fetched_mask).to_numpy().sum())
            missing_non_missing_points = int(template_non_missing_points - matched_non_missing_points)
            if template_non_missing_points > 0:
                coverage_pct = round(100.0 * matched_non_missing_points / template_non_missing_points, 4)

            for col in schema.series_columns:
                template_points = int(template_mask[col].sum())
                if template_points <= 0:
                    continue
                matched_points = int((template_mask[col] & fetched_mask[col]).sum())
                if matched_points < template_points:
                    missing_points = int(template_points - matched_points)
                    col_cov = round(100.0 * matched_points / template_points, 4)
                    low_coverage_series.append(
                        {
                            "series": col,
                            "coverage_pct": col_cov,
                            "template_points": template_points,
                            "missing_points": missing_points,
                        }
                    )

        template_period_mask = _presence_mask_by_period(template_df, schema.first_col, schema.series_columns, period_freq)
        fetched_period_mask = _presence_mask_by_period(fetched_df, schema.first_col, schema.series_columns, period_freq)
        common_periods = template_period_mask.index.intersection(fetched_period_mask.index).sort_values()
        common_periods_count = int(len(common_periods))
        if common_periods_count > 0:
            common_periods_min = str(common_periods.min())
            common_periods_max = str(common_periods.max())
            template_period_common = template_period_mask.loc[common_periods, schema.series_columns]
            fetched_period_common = fetched_period_mask.loc[common_periods, schema.series_columns]
            template_period_non_missing_points = int(template_period_common.to_numpy().sum())
            matched_period_non_missing_points = int((template_period_common & fetched_period_common).to_numpy().sum())
            missing_period_non_missing_points = int(template_period_non_missing_points - matched_period_non_missing_points)
            if template_period_non_missing_points > 0:
                period_coverage_pct = round(100.0 * matched_period_non_missing_points / template_period_non_missing_points, 4)

            for col in schema.series_columns:
                template_points = int(template_period_common[col].sum())
                if template_points <= 0:
                    continue
                matched_points = int((template_period_common[col] & fetched_period_common[col]).sum())
                if matched_points < template_points:
                    missing_points = int(template_points - matched_points)
                    col_cov = round(100.0 * matched_points / template_points, 4)
                    low_coverage_period_series.append(
                        {
                            "series": col,
                            "coverage_pct": col_cov,
                            "template_points": template_points,
                            "missing_points": missing_points,
                        }
                    )

    low_coverage_series.sort(key=lambda row: (row["coverage_pct"], -row["missing_points"], row["series"]))
    low_coverage_period_series.sort(key=lambda row: (row["coverage_pct"], -row["missing_points"], row["series"]))

    unresolved_series = sorted(unresolved_map.keys())
    report = {
        "template_data_rows": int(len(template_df)),
        "fetched_data_rows": int(len(fetched_df)),
        "template_date_min": _iso_date_or_none(template_df[schema.first_col].min() if not template_df.empty else None),
        "template_date_max": _iso_date_or_none(template_df[schema.first_col].max() if not template_df.empty else None),
        "fetched_date_min": _iso_date_or_none(fetched_df[schema.first_col].min() if not fetched_df.empty else None),
        "fetched_date_max": _iso_date_or_none(fetched_df[schema.first_col].max() if not fetched_df.empty else None),
        "template_series_with_any_data": int((template_series_counts > 0).sum()),
        "fetched_series_with_any_data": int((fetched_series_counts > 0).sum()),
        "unresolved_series_count": int(len(unresolved_series)),
        "unresolved_series": unresolved_series,
        "series_missing_in_fetched_count": int(len(series_missing_in_fetched)),
        "series_missing_in_fetched": series_missing_in_fetched,
        "common_template_dates_count": int(common_dates_count),
        "common_template_date_min": common_dates_min,
        "common_template_date_max": common_dates_max,
        "template_non_missing_points_on_common_dates": int(template_non_missing_points),
        "matched_non_missing_points_on_common_dates": int(matched_non_missing_points),
        "missing_non_missing_points_on_common_dates": int(missing_non_missing_points),
        "coverage_pct_on_template_non_missing_points": coverage_pct,
        "series_with_point_gaps_on_common_dates_count": int(len(low_coverage_series)),
        "series_with_point_gaps_on_common_dates_top20": low_coverage_series[:20],
        "period_alignment_frequency": period_freq,
        "common_template_periods_count": int(common_periods_count),
        "common_template_period_min": common_periods_min,
        "common_template_period_max": common_periods_max,
        "template_non_missing_points_on_common_periods": int(template_period_non_missing_points),
        "matched_non_missing_points_on_common_periods": int(matched_period_non_missing_points),
        "missing_non_missing_points_on_common_periods": int(missing_period_non_missing_points),
        "coverage_pct_on_template_non_missing_points_period_aligned": period_coverage_pct,
        "series_with_point_gaps_on_common_periods_count": int(len(low_coverage_period_series)),
        "series_with_point_gaps_on_common_periods_top20": low_coverage_period_series[:20],
    }
    return report


def _fetch_observations_for_series_id(
    series_id: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    page_limit: int,
) -> pd.Series:
    limit = max(1, min(int(page_limit), FRED_PAGE_LIMIT_MAX))
    offset = 0
    values: dict[pd.Timestamp, float] = {}

    while True:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "asc",
            "observation_start": FRED_EARLIEST_DATE,
            "observation_end": FRED_FAR_FUTURE_DATE,
            "limit": limit,
            "offset": offset,
        }
        url = f"{FRED_OBSERVATIONS_URL}?{urllib.parse.urlencode(params)}"

        payload: dict[str, object] | None = None
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                request = urllib.request.Request(url=url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                status = int(getattr(exc, "code", 0))
                body_msg = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                    parsed = json.loads(body)
                    if isinstance(parsed, dict) and parsed.get("error_message"):
                        body_msg = str(parsed["error_message"])
                    else:
                        body_msg = body[:200]
                except Exception:
                    body_msg = str(exc)

                transient = status == 429 or status >= 500
                if transient and attempt < max_retries:
                    sleep_s = retry_backoff_seconds * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                raise FredAPIError(f"HTTP {status} for series_id={series_id}: {body_msg}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < max_retries:
                    sleep_s = retry_backoff_seconds * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                break

        if payload is None:
            raise FredAPIError(f"Network error for series_id={series_id}: {last_error}")

        if "error_code" in payload:
            message = payload.get("error_message") or f"FRED API error code={payload.get('error_code')}"
            raise FredAPIError(str(message))

        observations = payload.get("observations", [])
        if not isinstance(observations, list):
            raise FredAPIError(f"Invalid observations payload for series_id={series_id}")

        for obs in observations:
            if not isinstance(obs, dict):
                continue
            date_raw = obs.get("date")
            val_raw = obs.get("value")
            if date_raw in (None, ""):
                continue
            if val_raw in (None, "", "."):
                continue

            try:
                val = float(val_raw)
            except (TypeError, ValueError):
                continue

            ts = pd.to_datetime(date_raw, errors="coerce")
            if pd.isna(ts):
                continue
            values[pd.Timestamp(ts)] = val

        if not observations:
            break

        count_raw = payload.get("count")
        try:
            count = int(count_raw) if count_raw is not None else None
        except (TypeError, ValueError):
            count = None

        next_offset = offset + len(observations)
        if count is not None and next_offset >= count:
            break
        if count is None and len(observations) < limit:
            break
        if next_offset <= offset:
            break
        offset = next_offset

    if not values:
        raise FredAPIError(f"No observations returned for series_id={series_id}")

    out = pd.Series(values, dtype=float, name=series_id)
    out = out.sort_index()
    return out


def _fetch_one_variable(
    variable_name: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    page_limit: int,
) -> FetchResult:
    attempts: list[str] = []
    for candidate_id in _candidate_series_ids(variable_name):
        try:
            series = _fetch_observations_for_series_id(
                series_id=candidate_id,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                page_limit=page_limit,
            )
            return FetchResult(
                variable_name=variable_name,
                resolved_series_id=candidate_id,
                series=series,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{candidate_id}: {exc}")

    return FetchResult(
        variable_name=variable_name,
        resolved_series_id=None,
        series=None,
        error=" | ".join(attempts) if attempts else "No candidate series IDs generated",
    )


def fetch_template_dataset(
    schema: TemplateSchema,
    api_key: str,
    max_workers: int,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    page_limit: int,
    retry_rate_limited_passes: int,
    retry_rate_limited_sleep_seconds: float,
    series_limit: int | None,
    label: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    requested_vars = list(schema.series_columns)
    if series_limit is not None:
        requested_vars = requested_vars[: int(series_limit)]
    template_period_freq = _infer_period_frequency(
        schema.data_rows[schema.first_col] if not schema.data_rows.empty else pd.Series(dtype="datetime64[ns]")
    )

    resolved_map: dict[str, str] = {}
    unresolved_map: dict[str, str] = {}
    series_by_var: dict[str, pd.Series] = {}

    print(f"[{label}] Fetching {len(requested_vars)} variables from FRED API...")
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(
                _fetch_one_variable,
                variable_name,
                api_key,
                timeout_seconds,
                max_retries,
                retry_backoff_seconds,
                page_limit,
            ): variable_name
            for variable_name in requested_vars
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result.series is not None and result.resolved_series_id is not None:
                aligned = _align_series_to_template_frequency(result.series, template_period_freq)
                if aligned.empty:
                    unresolved_map[result.variable_name] = (
                        f"Resolved {result.resolved_series_id} but no usable data after {template_period_freq} alignment"
                    )
                    print(
                        f"[{label}] {completed}/{len(requested_vars)} FAIL: {result.variable_name} "
                        f"(empty after alignment)"
                    )
                else:
                    resolved_map[result.variable_name] = result.resolved_series_id
                    series_by_var[result.variable_name] = aligned
                    print(
                        f"[{label}] {completed}/{len(requested_vars)} OK: {result.variable_name} "
                        f"<- {result.resolved_series_id} ({len(aligned)} aligned obs)"
                    )
            else:
                unresolved_map[result.variable_name] = result.error or "Unknown fetch error"
                print(f"[{label}] {completed}/{len(requested_vars)} FAIL: {result.variable_name}")

    extra_passes = max(0, int(retry_rate_limited_passes))
    for pass_num in range(1, extra_passes + 1):
        rate_limited_vars = [
            var
            for var, err in unresolved_map.items()
            if err and "HTTP 429" in err
        ]
        if not rate_limited_vars:
            break

        print(
            f"[{label}] Rate-limit retry pass {pass_num}/{extra_passes} "
            f"for {len(rate_limited_vars)} variables (sequential)."
        )
        for idx, variable_name in enumerate(rate_limited_vars, start=1):
            result = _fetch_one_variable(
                variable_name=variable_name,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                page_limit=page_limit,
            )
            if result.series is not None and result.resolved_series_id is not None:
                aligned = _align_series_to_template_frequency(result.series, template_period_freq)
                if aligned.empty:
                    unresolved_map[result.variable_name] = (
                        f"Resolved {result.resolved_series_id} but no usable data after {template_period_freq} alignment"
                    )
                    print(f"[{label}] retry {idx}/{len(rate_limited_vars)} FAIL: {result.variable_name} (empty)")
                else:
                    resolved_map[result.variable_name] = result.resolved_series_id
                    series_by_var[result.variable_name] = aligned
                    unresolved_map.pop(result.variable_name, None)
                    print(
                        f"[{label}] retry {idx}/{len(rate_limited_vars)} OK: "
                        f"{result.variable_name} <- {result.resolved_series_id} ({len(aligned)} aligned obs)"
                    )
            else:
                unresolved_map[result.variable_name] = result.error or "Unknown fetch error"
                print(f"[{label}] retry {idx}/{len(rate_limited_vars)} FAIL: {result.variable_name}")
            if retry_rate_limited_sleep_seconds > 0:
                time.sleep(float(retry_rate_limited_sleep_seconds))

    _construct_missing_series(
        requested_vars=requested_vars,
        template_history=schema.data_rows,
        date_col=schema.first_col,
        unresolved_map=unresolved_map,
        resolved_map=resolved_map,
        series_by_var=series_by_var,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        page_limit=page_limit,
        period_freq=template_period_freq,
        label=label,
    )

    if series_by_var:
        data_df = pd.DataFrame(series_by_var)
        data_df.index = pd.to_datetime(data_df.index, errors="coerce")
        data_df = data_df.loc[data_df.index.notna()].sort_index()
        data_df = data_df.reset_index().rename(columns={"index": schema.first_col})
        data_df[schema.first_col] = pd.to_datetime(data_df[schema.first_col], errors="coerce").map(
            lambda ts: f"{ts.month}/{ts.day}/{ts.year}" if pd.notna(ts) else ""
        )
    else:
        data_df = pd.DataFrame(columns=[schema.first_col])

    for col in requested_vars:
        if col not in data_df.columns:
            data_df[col] = pd.NA

    data_df = data_df[[schema.first_col, *requested_vars]]

    metadata_rows = schema.metadata_rows.copy()
    for col in data_df.columns:
        if col not in metadata_rows.columns:
            metadata_rows[col] = ""
    metadata_rows = metadata_rows[data_df.columns]

    full = pd.concat([metadata_rows, data_df.astype(object)], ignore_index=True)
    return full, resolved_map, unresolved_map


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep="")


def main() -> int:
    args = parse_args()
    api_key = resolve_api_key(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_schema = read_template_schema(args.md_template)
    qd_schema = read_template_schema(args.qd_template)

    md_df, md_resolved, md_unresolved = fetch_template_dataset(
        schema=md_schema,
        api_key=api_key,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        page_limit=args.page_limit,
        retry_rate_limited_passes=args.retry_rate_limited_passes,
        retry_rate_limited_sleep_seconds=args.retry_rate_limited_sleep_seconds,
        series_limit=args.series_limit,
        label="MD",
    )
    qd_df, qd_resolved, qd_unresolved = fetch_template_dataset(
        schema=qd_schema,
        api_key=api_key,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        page_limit=args.page_limit,
        retry_rate_limited_passes=args.retry_rate_limited_passes,
        retry_rate_limited_sleep_seconds=args.retry_rate_limited_sleep_seconds,
        series_limit=args.series_limit,
        label="QD",
    )

    md_output = output_dir / args.md_output_name
    qd_output = output_dir / args.qd_output_name
    write_csv(md_df, md_output)
    write_csv(qd_df, qd_output)

    md_coverage = compute_coverage_report(
        schema=md_schema,
        fetched_full_df=md_df,
        unresolved_map=md_unresolved,
    )
    qd_coverage = compute_coverage_report(
        schema=qd_schema,
        fetched_full_df=qd_df,
        unresolved_map=qd_unresolved,
    )
    coverage_report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "md": md_coverage,
        "qd": qd_coverage,
    }
    coverage_report_path = output_dir / args.coverage_report_name
    coverage_report_path.write_text(json.dumps(coverage_report, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "md": {
            "template": str(md_schema.path),
            "output": str(md_output),
            "series_requested": len(md_schema.series_columns if args.series_limit is None else md_schema.series_columns[: args.series_limit]),
            "series_resolved": len(md_resolved),
            "series_unresolved": len(md_unresolved),
            "resolved_map": md_resolved,
            "unresolved_map": md_unresolved,
            "coverage_vs_template": md_coverage,
        },
        "qd": {
            "template": str(qd_schema.path),
            "output": str(qd_output),
            "series_requested": len(qd_schema.series_columns if args.series_limit is None else qd_schema.series_columns[: args.series_limit]),
            "series_resolved": len(qd_resolved),
            "series_unresolved": len(qd_unresolved),
            "resolved_map": qd_resolved,
            "unresolved_map": qd_unresolved,
            "coverage_vs_template": qd_coverage,
        },
    }
    manifest_path = output_dir / "fred_latest_fetch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved MD latest file: {md_output}")
    print(f"Saved QD latest file: {qd_output}")
    print(f"Saved fetch manifest: {manifest_path}")
    print(f"Saved coverage report: {coverage_report_path}")
    print(
        "Summary: "
        f"MD resolved={len(md_resolved)} unresolved={len(md_unresolved)} | "
        f"QD resolved={len(qd_resolved)} unresolved={len(qd_unresolved)}"
    )
    print(
        "Coverage vs template: "
        f"MD={md_coverage.get('coverage_pct_on_template_non_missing_points')}% "
        f"(gaps={md_coverage.get('series_missing_in_fetched_count')}) | "
        f"QD={qd_coverage.get('coverage_pct_on_template_non_missing_points')}% "
        f"(gaps={qd_coverage.get('series_missing_in_fetched_count')})"
    )
    print(
        "Coverage vs template (period-aligned): "
        f"MD={md_coverage.get('coverage_pct_on_template_non_missing_points_period_aligned')}% "
        f"[{md_coverage.get('period_alignment_frequency')}] | "
        f"QD={qd_coverage.get('coverage_pct_on_template_non_missing_points_period_aligned')}% "
        f"[{qd_coverage.get('period_alignment_frequency')}]"
    )

    total_unresolved = len(md_unresolved) + len(qd_unresolved)
    if args.fail_on_unresolved and total_unresolved > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
