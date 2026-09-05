"""Strict model implementations. Inputs are arrays from ONE verified origin.

No legacy adapters, data files, leaderboards, automatic COVID interventions, or
silent fallback models are used. Optional numerical libraries are loaded lazily.
All returned paths are quarterly log growth, including models fitted to levels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import warnings

import numpy as np
import pandas as pd

from .fred_transforms import fred_transform
from .pit import PITError

CORE_MODELS = ("naive_last", "last_growth", "mean_growth", "ar4", "bridge_ridge")
CATALOG_MODELS = (
    *CORE_MODELS, "mean", "drift", "seasonal_naive", "random_normal", "random_uniform",
    "random_permutation", "auto_arima", "auto_ets", "theta", "local_trend_ssm",
    "random_forest", "xgboost", "bvar_minnesota_8", "bvar_minnesota_20",
    "bvar_minnesota_growth_8", "bvar_minnesota_growth_20", "factor_pca_qd",
    "mixed_freq_dfm_md", "lstm_univariate", "lstm_multivariate",
    "ensemble_avg_top3", "ensemble_weighted_top5", "chronos2",
)
ENSEMBLES = ("ensemble_avg_top3", "ensemble_weighted_top5")
COVARIATE_MODELS = {
    "bridge_ridge", "random_forest", "xgboost", "factor_pca_qd", "mixed_freq_dfm_md",
    "lstm_multivariate", *[m for m in CATALOG_MODELS if m.startswith("bvar_")],
}


class UnsupportedModel(PITError):
    """A requested model has no admissible specification/evidence at this origin."""


@dataclass
class ModelData:
    y: pd.Series  # contiguous quarterly GDP levels, through last released GDP
    features: pd.DataFrame  # growth-quarter index through target; includes counts/missingness
    covariates: tuple[str, ...]
    metadata: dict  # raw, verified origin records (needed for native monthly DFM inputs)
    steps: int
    seed: int = 0
    chronos_checkpoint: str | None = None
    origin: str | None = None

    @property
    def g(self):
        return np.diff(np.log(self.y.to_numpy()))


def model_seed(seed: int, name: str, origin: str, target: str) -> int:
    """Stable across catalog order, failed models and parallel scheduling."""
    digest = hashlib.sha256(f"{seed}|{name}|{origin}|{target}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Training-only mean imputation and scaling, preserving column positions."""
    train, other = np.asarray(train, float), np.asarray(other, float)
    valid = np.isfinite(train)
    count = valid.sum(axis=0)
    mean = np.divide(np.where(valid, train, 0).sum(axis=0), count,
                     out=np.zeros(train.shape[1]), where=count > 0)
    filled = np.where(valid, train, mean)
    scale = filled.std(axis=0)
    scale[scale < 1e-12] = 1.
    return ((filled - mean) / scale, (np.where(np.isfinite(other), other, mean) - mean) / scale,
            dict(imputation_mean=mean.tolist(), scale=scale.tolist(), observed_counts=count.tolist()))


def _level_to_growth(log_levels, last_level):
    return np.diff(np.r_[np.log(last_level), np.asarray(log_levels, float)])


def _supervised(g, features, lags):
    if len(g) < lags + 8:
        raise PITError(f"Need at least {lags + 8} growth observations")
    x = np.array([np.r_[g[t-lags:t][::-1], features[t]] for t in range(lags, len(g))])
    return x, g[lags:]


def _regression(d: ModelData, name: str):
    g = d.g
    lags = 8 if name in {"random_forest", "xgboost"} else 4
    f = d.features.to_numpy(float)
    details = dict(lags=lags, feature_columns=list(d.features.columns))
    if name == "factor_pca_qd":
        # Fit PCA only on quarters with observed training GDP, never target-quarter data.
        from sklearn.decomposition import PCA
        columns = list(d.covariates)
        raw = d.features[columns].to_numpy(float)
        train, rest, scaling = standardize(raw[:len(g)], raw[len(g):])
        active = np.std(train, axis=0) > 1e-12
        if not active.any():
            raise PITError("PCA has no varying observed training covariate")
        n = min(6, int(active.sum()), len(train) - 1)
        pca = PCA(n_components=n, svd_solver="full")
        factors = np.vstack([pca.fit_transform(train[:, active]), pca.transform(rest[:, active])])
        indicators = d.features.drop(columns=columns).to_numpy(float)
        f = np.column_stack([factors, indicators])
        details.update(pca_n_factors=n, pca_columns=np.asarray(columns)[active].tolist(),
                       pca_loadings=pca.components_.tolist(), pca_scaling=scaling)
    x, target = _supervised(g, f, lags)
    train, _, scaling = standardize(x, x[:0])
    details.update(preprocessing=scaling, fit_rows=len(target))
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        params = dict(n_estimators=120, max_depth=12, max_features="sqrt", min_samples_leaf=1,
                      n_jobs=1, random_state=d.seed)
        estimator = RandomForestRegressor(**params).fit(train, target)
    elif name == "xgboost":
        from xgboost import XGBRegressor
        params = dict(n_estimators=200, max_depth=3, learning_rate=.05, objective="reg:squarederror",
                      subsample=1., colsample_bytree=1., n_jobs=1, random_state=d.seed, tree_method="hist")
        estimator = XGBRegressor(**params).fit(train, target)
    else:
        from sklearn.linear_model import Ridge
        params = dict(alpha=1.)
        estimator = Ridge(**params).fit(train, target)
    details["parameters"] = params
    history = list(g)
    for j in range(d.steps):
        row = np.r_[history[-lags:][::-1], f[len(g)+j]]
        row = np.where(np.isfinite(row), row, scaling["imputation_mean"])
        row = (row - scaling["imputation_mean"]) / scaling["scale"]
        history.append(float(estimator.predict(row.reshape(1, -1))[0]))
    return history[-d.steps:], details


def _bvar(d: ModelData, name: str):
    cap = 7 if name.endswith("_8") else 19
    covs = list(d.covariates[:cap])
    is_growth = "growth" in name
    target = d.g if is_growth else np.log(d.y.to_numpy()[1:])
    # Partial monthly means are NOT treated as complete quarterly VAR states.
    cov = d.features[covs].copy()
    for c in covs:
        frequency = next(r["frequency"] for r in d.metadata["inputs"] if r["variable"] == c)
        expected = 3 if frequency == "M" else 1
        cov.loc[d.features[c + "__count"] < expected, c] = np.nan
    history = np.column_stack([target, cov.iloc[:len(target)].to_numpy()])
    scaled, future, scaling = standardize(history, np.column_stack([
        np.full(d.steps, np.nan), cov.iloc[len(target):].to_numpy()]))
    k, lags = scaled.shape[1], 2
    x = np.array([np.r_[1., scaled[t-1], scaled[t-2]] for t in range(lags, len(scaled))])
    y = scaled[lags:]
    betas = []
    shrink = 6. if cap == 7 else 7.
    for eq in range(k):
        prior = np.zeros(x.shape[1])
        prior[1+eq] = 0. if is_growth else 1.
        weights = [.2]
        for lag in range(1, lags+1):
            weights.extend([(.3 if j == eq and lag == 1 else 1. if j == eq else 2.) * lag**1.5
                            for j in range(k)])
        penalty = np.diag(shrink * np.square(weights))
        betas.append(np.linalg.solve(x.T @ x + penalty, x.T @ y[:, eq] + penalty @ prior))
    beta = np.column_stack(betas)
    extended = list(scaled)
    future_observed = np.isfinite(cov.iloc[len(target):].to_numpy())
    for j in range(d.steps):
        prediction = np.r_[1., extended[-1], extended[-2]] @ beta
        # Observed complete covariates condition later steps, not the contemporaneous GDP equation.
        prediction[1:] = np.where(future_observed[j], future[j, 1:], prediction[1:])
        extended.append(prediction)
    forecast = np.asarray(extended[-d.steps:])[:, 0] * scaling["scale"][0] + scaling["imputation_mean"][0]
    path = forecast if is_growth else _level_to_growth(forecast, d.y.iloc[-1])
    return path, dict(covariates=covs, dimension=k, max_dimension=cap+1, lags=lags,
                      shrinkage=shrink, own_lag1_prior=0. if is_growth else 1., preprocessing=scaling,
                      estimator="Minnesota-style penalized posterior mode; no posterior simulation",
                      partial_quarters="training mean imputed; future VAR states forecast")


def monthly_inputs(d: ModelData) -> pd.DataFrame:
    series = {}
    for c in d.covariates:
        rows = [r for r in d.metadata["inputs"] if r["variable"] == c]
        if not rows or rows[0]["frequency"] != "M":
            raise UnsupportedModel("Mixed-frequency DFM requires explicitly monthly covariates")
        s = pd.Series([r["value"] for r in rows], index=pd.PeriodIndex([r["obs_date"] for r in rows], freq="M"), dtype=float).sort_index()
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
        # Match provider ordering: native transformation precedes temporal alignment.
        values = fred_transform(s, rows[0]["tcode"]).to_numpy()
        series[c] = pd.Series(values, index=s.index).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(series)


def _dfm(d: ModelData):
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
    monthly = monthly_inputs(d)
    start = d.y.index[1].asfreq("M", "start")
    train_end = d.y.index[-1].asfreq("M", "end")
    target_end = (d.y.index[-1] + d.steps).asfreq("M", "end")
    cutoff_month = pd.Period(d.metadata["information_cutoff"], freq="M")
    index = pd.period_range(start, max(target_end, cutoff_month), freq="M")
    monthly = monthly.reindex(index)
    training = monthly.loc[:train_end]
    if (training.count() < 24).any() or (training.std() < 1e-12).any():
        raise PITError("DFM requires at least 24 nonconstant monthly training observations per series")
    quarterly = pd.Series(d.g, index=d.y.index[1:], name="GDPC1")
    params = dict(factors=1, factor_orders=1, idiosyncratic_ar1=False, standardize=True)
    model = DynamicFactorMQ(training, endog_quarterly=quarterly, **params)
    result = model.fit_em(maxiter=5000, tolerance=1e-6, em_initialization=False, disp=False)
    llf = np.asarray(result.mle_retvals["llf"], float)
    delta = 2 * abs(llf[-1] - llf[-2]) / max(abs(llf[-1]) + abs(llf[-2]), 1e-12)
    if not np.isfinite(llf).all() or delta > 1e-6:
        raise PITError(f"DFM EM did not converge after {result.mle_retvals['iter']} iterations (relative likelihood change={delta:g})")
    # Reuse training parameters/scaling; condition on ALL data known at this origin.
    updated = result.apply(monthly, endog_quarterly=quarterly, refit=False,
                           retain_standardization=True, copy_initialization=True)
    predictions = updated.get_prediction(information_set="smoothed").predicted_mean
    dates = [(d.y.index[-1] + j).asfreq("M", "end") for j in range(1, d.steps+1)]
    return predictions.loc[dates, "GDPC1"].to_numpy(), dict(
        parameters=params, em_initialization=False, em_maxiter=5000, em_iterations=int(result.mle_retvals["iter"]), em_relative_change=float(delta),
        training_months=[str(start), str(train_end)], conditioning_cutoff=d.metadata["information_cutoff"],
        monthly_counts=monthly.count().to_dict(), monthly_input_sha256=_array_hash(monthly.to_numpy()),
        quarterly_measurement="Mariano-Murasawa quarterly log-growth aggregation",
        fit_params=np.asarray(result.params).tolist(),
        prediction="smoothed only within origin information set; future observations remain missing")


def _array_hash(a):
    return hashlib.sha256(np.asarray(a, dtype="<f8").tobytes()).hexdigest()


def _lstm(d: ModelData, multivariate: bool):
    import torch
    g, lags = d.g, 8
    if len(g) < lags + 8:
        raise PITError("LSTM requires at least 16 growth observations")
    f = d.features.to_numpy(float) if multivariate else np.empty((len(d.features), 0))
    # Row for quarter s contains g[s-1] and covariates for s known at the outer origin.
    def sequence(history, t):
        return np.array([np.r_[history[s-1], f[s]] for s in range(t-lags+1, t+1)])
    raw = np.array([sequence(g, t) for t in range(lags, len(g))])
    x, _, scaling = standardize(raw.reshape(-1, raw.shape[-1]), np.empty((0, raw.shape[-1])))
    x = x.reshape(raw.shape)
    target = g[lags:]
    ym, ys = float(target.mean()), max(float(target.std()), 1e-12)
    epochs, hidden = 80, 16
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(d.seed)
            class Net(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = torch.nn.LSTM(x.shape[-1], hidden, batch_first=True)
                    self.head = torch.nn.Linear(hidden, 1)
                def forward(self, inputs):
                    return self.head(self.lstm(inputs)[0][:, -1]).squeeze(-1)
            net = Net().cpu()
            opt = torch.optim.Adam(net.parameters(), lr=.01)
            tx, ty = torch.tensor(x, dtype=torch.float32), torch.tensor((target-ym)/ys, dtype=torch.float32)
            for _ in range(epochs):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(tx), ty)
                if not torch.isfinite(loss):
                    raise PITError("LSTM nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                opt.step()
            net.eval()
            history = list(g)
            with torch.no_grad():
                for j in range(d.steps):
                    row = sequence(history, len(g)+j)
                    row = np.where(np.isfinite(row), row, scaling["imputation_mean"])
                    row = (row - scaling["imputation_mean"]) / scaling["scale"]
                    prediction = float(net(torch.tensor(row[None], dtype=torch.float32)).item()) * ys + ym
                    history.append(prediction)
            details = dict(lags=lags, hidden_size=hidden, epochs=epochs, learning_rate=.01,
                           device="cpu", training_loss=float(loss.detach()), preprocessing=scaling,
                           target_mean=ym, target_scale=ys, feature_columns=list(d.features.columns) if multivariate else [],
                           selection="fixed epochs; no evaluation outcomes or early stopping")
            return history[-d.steps:], details
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)


def forecast_model(name: str, d: ModelData) -> tuple[np.ndarray, dict]:
    """Run a catalog member; fail explicitly on unavailable evidence or a failed fit."""
    if name in COVARIATE_MODELS and not d.covariates:
        raise UnsupportedModel(f"{name} requires predeclared verified covariates")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path, details = _dispatch(name, d)
    path = np.asarray(path, float)
    if path.shape != (d.steps,) or not np.isfinite(path).all():
        raise PITError(f"{name} produced invalid quarterly growth path")
    details.update(seed=d.seed, warnings=sorted(set(str(w.message) for w in caught)),
                   prediction_target="quarterly log GDP growth", growth_path=path.tolist())
    return path, details


def _dispatch(name, d):
    g, logy, h = d.g, np.log(d.y.to_numpy()), d.steps
    rng = np.random.default_rng(d.seed)
    if name == "mean":
        return _level_to_growth(np.repeat(logy.mean(), h), d.y.iloc[-1]), {"fit_target": "log GDP level"}
    if name == "drift":
        return np.repeat((logy[-1]-logy[0])/(len(logy)-1), h), {"fit_target": "log GDP level"}
    if name == "seasonal_naive":
        levels = list(logy)
        for _ in range(h):
            levels.append(levels[-4])
        return _level_to_growth(levels[-h:], d.y.iloc[-1]), {"season_length": 4, "fit_target": "log GDP level"}
    if name in {"random_normal", "random_uniform", "random_permutation"}:
        if name == "random_normal":
            path = rng.normal(g.mean(), g.std(), h)
        elif name == "random_uniform":
            path = rng.uniform(g.min(), g.max(), h)
        else:
            path = np.concatenate([rng.permutation(g) for _ in range((h+len(g)-1)//len(g))])[:h]
        return path, {"fit_target": "quarterly log GDP growth", "role": "randomized control, not a substantive nowcast"}
    if name in {"auto_arima", "auto_ets", "theta"}:
        from statsforecast.models import AutoARIMA, AutoETS, Theta
        if name == "auto_arima":
            params = dict(season_length=4, ic="aicc", stepwise=True, approximation=False)
            estimator = AutoARIMA(**params)
        elif name == "auto_ets":
            params = dict(season_length=4, model="ZZZ")
            estimator = AutoETS(**params)
        else:
            params = dict(season_length=4, decomposition_type="additive")
            estimator = Theta(**params)
        estimator.fit(logy)
        forecast = estimator.predict(h=h)["mean"]
        fitted = estimator.model_
        details = dict(parameters=params, fit_target="log GDP level", selection="training-only likelihood/information criterion")
        for key in ("arma", "method", "components"):
            if key in fitted:
                details["selected_"+key] = np.asarray(fitted[key]).tolist()
        return _level_to_growth(forecast, d.y.iloc[-1]), details
    if name == "local_trend_ssm":
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        estimator = UnobservedComponents(logy, level="local linear trend")
        result = estimator.fit(disp=False, maxiter=1000)
        if not result.mle_retvals.get("converged", False):
            raise PITError("Local trend state-space optimizer did not converge")
        return _level_to_growth(result.forecast(h), d.y.iloc[-1]), dict(
            fit_target="log GDP level", level="local linear trend", cycle=False,
            fitted_parameters=result.params.tolist(), converged=True)
    if name in {"random_forest", "xgboost", "factor_pca_qd"}:
        return _regression(d, name)
    if name.startswith("bvar_"):
        return _bvar(d, name)
    if name == "mixed_freq_dfm_md":
        return _dfm(d)
    if name in {"lstm_univariate", "lstm_multivariate"}:
        return _lstm(d, name == "lstm_multivariate")
    if name == "chronos2":
        if not d.chronos_checkpoint:
            raise UnsupportedModel("Chronos-2 requires --chronos-checkpoint with publisher publication evidence and pinned artifact hashes")
        from .pit_checkpoint import validate_checkpoint
        try:
            directory, evidence = validate_checkpoint(d.chronos_checkpoint, d.origin)
        except (PITError, OSError, KeyError, ValueError) as exc:
            raise UnsupportedModel(f"Inadmissible checkpoint: {exc}") from exc
        from chronos import Chronos2Pipeline
        import torch
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(d.seed)
                pipeline = Chronos2Pipeline.from_pretrained(str(directory), device_map="cpu", local_files_only=True)
                quantiles, _ = pipeline.predict_quantiles([logy.astype(np.float32)], prediction_length=h, quantile_levels=[.5])
        finally:
            torch.set_num_threads(old_threads)
        forecast = quantiles[0][0, :, 0].numpy(force=True)
        return _level_to_growth(forecast, d.y.iloc[-1]), dict(
            checkpoint=evidence, fit_target="log GDP level", point_statistic="median",
            device="cpu", covariates=[], context_length=len(logy), zero_shot=True)
    raise UnsupportedModel(f"No strict implementation for {name}")
