from __future__ import annotations

from collections.abc import Callable, Sequence

from .ar_models import AR4Model
from .base import BaseModel
from .baselines import Drift, Mean, NaiveLast, SeasonalNaive
from .bvar_minnesota import (
    BVARMinnesota20Model,
    BVARMinnesota8Model,
    BVARMinnesotaGrowth20Model,
    BVARMinnesotaGrowth8Model,
)
from .chronos2 import Chronos2Model
from .ensemble import EnsembleAvgTop3Model
from .factor_models import MixedFrequencyDFMModel, QuarterlyFactorPCAModel
from .gdpnow import AtlantaFedGDPNowModel
from .lstm_models import LSTMMultivariateModel, LSTMUnivariateModel
from .random_forest import RandomForestModel
from .randoms import RandomNormal, RandomPermutation, RandomUniform
from .state_space import LocalTrendStateSpaceModel
from .statsforecast_models import AutoARIMAModel, AutoETSModel, ThetaModel
from .xgboost_model import XGBoostModel

ModelBuilder = Callable[[int], BaseModel]
MODEL_NAME_ALIASES: dict[str, str] = {
    "gdpnow": "atlantafed_gdpnow",
}


def _no_seed(factory: Callable[[], BaseModel]) -> ModelBuilder:
    def _builder(seed: int) -> BaseModel:
        _ = seed
        return factory()

    return _builder


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "naive_last": _no_seed(NaiveLast),
    "mean": _no_seed(Mean),
    "ar4": _no_seed(AR4Model),
    "drift": _no_seed(Drift),
    "seasonal_naive": _no_seed(lambda: SeasonalNaive(season_length=4)),
    "random_normal": lambda seed: RandomNormal(seed=seed),
    "random_uniform": lambda seed: RandomUniform(seed=seed),
    "random_permutation": lambda seed: RandomPermutation(seed=seed),
    "random_forest": lambda seed: RandomForestModel(seed=seed),
    "xgboost": lambda seed: XGBoostModel(seed=seed),
    "lstm_univariate": lambda seed: LSTMUnivariateModel(seed=seed),
    "lstm_multivariate": lambda seed: LSTMMultivariateModel(seed=seed),
    "local_trend_ssm": _no_seed(LocalTrendStateSpaceModel),
    "bvar_minnesota_8": _no_seed(BVARMinnesota8Model),
    "bvar_minnesota_20": _no_seed(BVARMinnesota20Model),
    "bvar_minnesota_growth_8": _no_seed(BVARMinnesotaGrowth8Model),
    "bvar_minnesota_growth_20": _no_seed(BVARMinnesotaGrowth20Model),
    "factor_pca_qd": lambda seed: QuarterlyFactorPCAModel(seed=seed),
    "mixed_freq_dfm_md": lambda seed: MixedFrequencyDFMModel(seed=seed),
    "atlantafed_gdpnow": _no_seed(AtlantaFedGDPNowModel),
    "gdpnow": _no_seed(AtlantaFedGDPNowModel),
    "ensemble_avg_top3": _no_seed(EnsembleAvgTop3Model),
    "auto_arima": _no_seed(lambda: AutoARIMAModel(season_length=4)),
    "auto_ets": _no_seed(lambda: AutoETSModel(season_length=4)),
    "theta": _no_seed(lambda: ThetaModel(season_length=4)),
    "chronos2": _no_seed(lambda: Chronos2Model(device_map="cpu")),
}


def register_model(name: str, builder: ModelBuilder) -> None:
    """Register a model factory under a stable CLI name."""
    MODEL_REGISTRY[name] = builder


def canonicalize_model_name(name: str) -> str:
    key = str(name).strip().lower()
    return MODEL_NAME_ALIASES.get(key, key)


def normalize_model_names(model_names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in model_names:
        canonical = canonicalize_model_name(str(name))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def available_models(*, include_aliases: bool = False) -> list[str]:
    if include_aliases:
        return sorted(MODEL_REGISTRY.keys())
    return sorted(normalize_model_names(MODEL_REGISTRY.keys()))


def build_models(model_names: list[str], seed: int = 0) -> dict[str, BaseModel]:
    unknown = [m for m in model_names if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available={available_models()}")

    models: dict[str, BaseModel] = {}
    for idx, model_name in enumerate(model_names):
        model_seed = seed + idx
        models[model_name] = MODEL_REGISTRY[model_name](model_seed)
    return models
