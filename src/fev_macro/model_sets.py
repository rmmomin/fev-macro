from __future__ import annotations

from typing import Literal, Sequence, cast

from .models import normalize_model_names

MODELS_LL_UNPROCESSED: list[str] = [
    "naive_last",
    "mean",
    "drift",
    "seasonal_naive",
    "auto_arima",
    "auto_ets",
    "theta",
    "local_trend_ssm",
    "random_forest",
    "xgboost",
    "bvar_minnesota_8",
    "bvar_minnesota_20",
    "factor_pca_qd",
    "mixed_freq_dfm_md",
    "chronos2",
    "ensemble_avg_top3",
]

MODELS_G_PROCESSED: list[str] = [
    "naive_last",
    "mean",
    "ar4",
    "auto_arima",
    "local_trend_ssm",
    "random_forest",
    "xgboost",
    "bvar_minnesota_growth_8",
    "bvar_minnesota_growth_20",
    "factor_pca_qd",
    "mixed_freq_dfm_md",
    "ensemble_avg_top3",
]

SUPPORTED_MODEL_SETS = {"auto", "ll", "g"}


def resolve_model_names(
    requested_models: Sequence[str] | None,
    *,
    covariate_mode: Literal["unprocessed", "processed"],
    model_set: Literal["auto", "ll", "g"] = "auto",
) -> list[str]:
    if requested_models:
        cleaned = [str(m).strip() for m in requested_models if str(m).strip()]
        return normalize_model_names(cleaned)

    mode = _normalize_model_set(model_set)
    if mode == "auto":
        mode = "ll" if covariate_mode == "unprocessed" else "g"

    defaults = MODELS_LL_UNPROCESSED if mode == "ll" else MODELS_G_PROCESSED
    return normalize_model_names(defaults)


def _normalize_model_set(value: str) -> Literal["auto", "ll", "g"]:
    v = str(value).strip().lower()
    if v not in SUPPORTED_MODEL_SETS:
        raise ValueError(f"Unsupported model_set={value!r}. Supported={sorted(SUPPORTED_MODEL_SETS)}")
    return cast(Literal["auto", "ll", "g"], v)
