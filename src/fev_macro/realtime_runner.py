from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, cast

RealtimeMode = Literal["unprocessed", "processed"]

SUPPORTED_REALTIME_MODES = {"unprocessed", "processed"}

DEFAULT_QD_PANEL_PATHS: dict[RealtimeMode, Path] = {
    "unprocessed": Path("data/panels/fred_qd_vintage_panel.parquet"),
    "processed": Path("data/panels/fred_qd_vintage_panel_processed.parquet"),
}

DEFAULT_MD_PANEL_PATHS: dict[RealtimeMode, Path] = {
    "unprocessed": Path("data/panels/fred_md_vintage_panel.parquet"),
    "processed": Path("data/panels/fred_md_vintage_panel_processed.parquet"),
}

DEFAULT_LATEST_FORECAST_OUTPUTS: dict[RealtimeMode, Path] = {
    "unprocessed": Path("results/realtime_latest_vintage_forecast_unprocessed.csv"),
    "processed": Path("results/realtime_latest_vintage_forecast_processed.csv"),
}

DEFAULT_REALTIME_OOS_OUTPUT_DIRS: dict[RealtimeMode, Path] = {
    "unprocessed": Path("results/realtime_oos_unprocessed"),
    "processed": Path("results/realtime_oos_processed"),
}

DEFAULT_BASELINE_BY_MODE: dict[RealtimeMode, str] = {
    "unprocessed": "rw_drift_log",
    "processed": "naive_last_growth",
}

DEFAULT_MODELS_UNPROCESSED_LL: list[str] = [
    "rw_drift_log",
    "ar4_growth",
    "auto_ets",
    "theta",
    "drift",
    "auto_arima",
    "local_trend_ssm",
    "random_forest",
    "xgboost",
    "chronos2",
    "factor_pca_qd",
    "bvar_minnesota_8",
    "bvar_minnesota_20",
    "mixed_freq_dfm_md",
    "ensemble_avg_top3",
    "ensemble_weighted_top5",
    "naive_last",
]

DEFAULT_MODELS_PROCESSED_G: list[str] = [
    "naive_last_growth",
    "ar4_growth",
    "auto_arima_growth",
    "auto_ets_growth",
    "theta_growth",
    "local_trend_ssm_growth",
    "random_forest_growth",
    "xgboost_growth",
    "chronos2_growth",
    "factor_pca_qd_growth",
    "bvar_minnesota_growth_8",
    "bvar_minnesota_growth_20",
    "mixed_freq_dfm_md_growth",
    "ensemble_avg_top3_processed_g",
    "ensemble_weighted_top5_processed_g",
]

DEFAULT_MODELS_BY_MODE: dict[RealtimeMode, list[str]] = {
    "unprocessed": DEFAULT_MODELS_UNPROCESSED_LL,
    "processed": DEFAULT_MODELS_PROCESSED_G,
}


def normalize_mode(value: str | None) -> RealtimeMode:
    mode = str(value or "unprocessed").strip().lower()
    if mode not in SUPPORTED_REALTIME_MODES:
        raise ValueError(f"Unsupported mode={value!r}. Supported={sorted(SUPPORTED_REALTIME_MODES)}")
    return cast(RealtimeMode, mode)


def resolve_qd_panel_path(mode: RealtimeMode, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return DEFAULT_QD_PANEL_PATHS[mode].resolve()


def resolve_md_panel_path(mode: RealtimeMode, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return DEFAULT_MD_PANEL_PATHS[mode].resolve()


def resolve_models(mode: RealtimeMode, requested_models: Sequence[str] | None) -> list[str]:
    if requested_models:
        return _dedupe([str(m).strip() for m in requested_models if str(m).strip()])
    return list(DEFAULT_MODELS_BY_MODE[mode])


def resolve_baseline_model(mode: RealtimeMode, explicit_baseline: str | None) -> str:
    if explicit_baseline and str(explicit_baseline).strip():
        return str(explicit_baseline).strip()
    return DEFAULT_BASELINE_BY_MODE[mode]


def resolve_latest_output_path(mode: RealtimeMode, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return DEFAULT_LATEST_FORECAST_OUTPUTS[mode].resolve()


def resolve_oos_output_dir(mode: RealtimeMode, explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve()
    return DEFAULT_REALTIME_OOS_OUTPUT_DIRS[mode].resolve()


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


__all__ = [
    "DEFAULT_BASELINE_BY_MODE",
    "DEFAULT_MODELS_BY_MODE",
    "DEFAULT_MODELS_PROCESSED_G",
    "DEFAULT_MODELS_UNPROCESSED_LL",
    "DEFAULT_MD_PANEL_PATHS",
    "DEFAULT_QD_PANEL_PATHS",
    "DEFAULT_REALTIME_OOS_OUTPUT_DIRS",
    "DEFAULT_LATEST_FORECAST_OUTPUTS",
    "RealtimeMode",
    "normalize_mode",
    "resolve_baseline_model",
    "resolve_latest_output_path",
    "resolve_md_panel_path",
    "resolve_models",
    "resolve_oos_output_dir",
    "resolve_qd_panel_path",
]
