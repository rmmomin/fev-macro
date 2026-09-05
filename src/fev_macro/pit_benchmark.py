"""Auditable catalog with an explicit information set at every origin.

No archive panels, scoring releases or stored leaderboards enter fitting.
Ensembles replay earlier PIT origins; pretrained checkpoints need dated evidence.
"""
from __future__ import annotations

import json
import math
from typing import Sequence

from .pit_models import (CATALOG_MODELS, COVARIATE_MODELS, ENSEMBLES,
                         ModelData, UnsupportedModel, forecast_model, model_seed)

import numpy as np
import pandas as pd

from .asof_provider import AsofVintageProvider
from .asof_store import AsofStore
from .pit import PITError, content_hash, information_date

MODELS = CATALOG_MODELS
STAGES = ("first", "second", "third")


def growth(current: float, previous: float, *, annualized: bool = True) -> float:
    if not np.isfinite([current, previous]).all() or min(current, previous) <= 0:
        raise PITError("GDP growth requires finite positive levels")
    return float(100 * np.expm1((4 if annualized else 1) * (np.log(current) - np.log(previous))))


def validate_release_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    required = {"quarter", "stage", "release_date", "source_url"}
    if not required.issubset(calendar):
        raise PITError(f"Release calendar requires {sorted(required)}")
    c = calendar.copy()
    c["quarter"] = pd.PeriodIndex(c.quarter, freq="Q-DEC")
    c["release_date"] = pd.to_datetime(c.release_date, errors="raise")
    if c[list(required)].isna().any().any() or not c.stage.isin(STAGES).all():
        raise PITError("Invalid release stage/calendar value")
    if not c.source_url.astype(str).str.startswith("https://www.bea.gov/").all():
        raise PITError("Stage labels require a sourced BEA release calendar")
    if c.duplicated(["quarter", "stage"]).any() or c.duplicated(["quarter", "release_date"]).any():
        raise PITError("Duplicate release stage/date")
    for q, rows in c.groupby("quarter"):
        rows = rows.assign(order=rows.stage.map({s: i for i, s in enumerate(STAGES)})).sort_values("order")
        if not rows.release_date.is_monotonic_increasing or (rows.release_date <= q.end_time).any():
            raise PITError("GDP stages must be ordered and follow the observation quarter")
    return c.sort_values(["quarter", "release_date"]).reset_index(drop=True)


def build_release_truth(store: AsofStore, calendar: pd.DataFrame) -> pd.DataFrame:
    """Read q and q-1 from the SAME release-date snapshot, including unchanged releases."""
    if not store.strict_pit:
        raise PITError("Truth requires a strict ALFRED store")
    rows = []
    for r in validate_release_calendar(calendar).itertuples():
        q, d = r.quarter, r.release_date
        snap = store.snapshot_long(asof_ts=d + pd.Timedelta(days=1), series_ids=["GDPC1"],
                                   obs_start=(q - 1).start_time, obs_end=q.start_time,
                                   include_asof_used=True)
        snap = snap.set_index("obs_ts")
        if q.start_time not in snap.index or (q - 1).start_time not in snap.index:
            raise PITError(f"Missing same-vintage GDP pair for {q} {r.stage} on {d.date()}")
        current, previous = float(snap.loc[q.start_time, "value"]), float(snap.loc[(q - 1).start_time, "value"])
        saar = growth(current, previous)
        if hasattr(r, "expected_saar") and pd.notna(r.expected_saar) and abs(saar - float(r.expected_saar)) > 0.06:
            raise PITError(f"ALFRED/BEA growth disagreement for {q} {r.stage}")
        rows.append(dict(target_quarter=str(q), release_stage=r.stage, release_date=d.date().isoformat(),
                         source_url=r.source_url, y_true_level=current, truth_previous_level=previous,
                         g_true_qoq=growth(current, previous, annualized=False), g_true_saar=saar,
                         truth_vintages=json.dumps([str(v.date()) for v in snap.asof_used]),
                         truth_provenance_ids=json.dumps(sorted(set(snap.provenance_id)))))
    return pd.DataFrame(rows)


def _regression_path(g: np.ndarray, steps: int, features: pd.DataFrame | None) -> list[float]:
    # Four fixed lags; no hyperparameter or feature selection using test losses.
    lags = 4
    if len(g) <= lags + 2:
        raise PITError("Insufficient contiguous GDP history for AR(4)")
    x = np.array([g[t-lags:t][::-1] for t in range(lags, len(g))])
    y = g[lags:]
    if features is not None:
        f = features.to_numpy(dtype=float)
        x = np.column_stack([x, f[lags:len(g)]])
    # Fit every imputation and scale parameter on training rows only.
    x = np.where(np.isfinite(x), x, np.nan)
    finite_counts = np.isfinite(x).sum(axis=0)
    means = np.divide(np.nansum(x, axis=0), finite_counts, out=np.zeros(x.shape[1]), where=finite_counts > 0)
    x = np.where(np.isfinite(x), x, means)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1
    design = np.column_stack([np.ones(len(x)), (x - means) / scale])
    penalty = np.eye(design.shape[1]) * (1.0 if features is not None else 0.0)
    penalty[0, 0] = 0
    beta = (np.linalg.solve(design.T @ design + penalty, design.T @ y) if features is not None
            else np.linalg.lstsq(design, y, rcond=None)[0])
    history = list(g)
    for j in range(steps):
        row = np.asarray(history[-lags:][::-1])
        if features is not None:
            row = np.r_[row, f[len(g) + j]]
        row = np.where(np.isfinite(row), row, means)
        prediction = float(np.r_[1., (row - means) / scale] @ beta)
        if not np.isfinite(prediction):
            raise PITError("Nonfinite regression forecast")
        history.append(prediction)
    return history[-steps:]


def _ensemble_validation(provider, origin, target, y, *, candidates, windows, covariates,
                         min_train, rolling_size, seed):
    """Rebuild historical validation inputs at THEIR origins, score only known GDP.

    Validation truth is the GDP vintage available at the CURRENT selection origin,
    not an eventual revision or a claimed first release. Every candidate uses the
    same quarters. Failure of a candidate fails selection; none is silently dropped.
    """
    quarters = y.index[-windows:]
    if len(quarters) != windows or quarters[0] - 1 not in y.index:
        raise PITError("Insufficient released GDP for ensemble validation")
    requests = pd.DataFrame([dict(origin_date=(origin - pd.DateOffset(months=3*(target.ordinal-q.ordinal))).isoformat(),
                                  target_quarter=str(q)) for q in quarters])
    if (pd.to_datetime(requests.origin_date) >= origin).any():
        raise PITError("Ensemble validation origin is not earlier than forecast origin")
    inner, audits = run_pit_backtest(provider, requests, models=candidates, covariates=covariates,
                                    min_train=min_train, rolling_size=rolling_size, seed=seed)
    errors = {}
    truth = {str(q): growth(float(y.loc[q]), float(y.loc[q-1])) for q in quarters}
    for name in candidates:
        sample = inner.loc[inner.model == name].sort_values("target_quarter")
        if len(sample) != windows or not np.isfinite(sample.g_hat_saar).all():
            raise PITError("Ensemble requires a complete matched validation sample")
        errors[name] = float(np.sqrt(np.mean([(r.g_hat_saar - truth[r.target_quarter])**2
                                              for r in sample.itertuples()])))
    return dict(candidates=list(candidates), windows=windows, rmse_saar=errors,
                truth_saar=truth, truth_vintage_rule="same current-origin GDP snapshot as outer training",
                origin_rule="same calendar day/time shifted by whole quarters", validation_audits=audits)


def run_pit_backtest(
    provider: AsofVintageProvider, origins: pd.DataFrame, *,
    models: Sequence[str] = ("naive_last", "mean_growth", "ar4"),
    covariates: Sequence[str] = (), min_train: int = 24, rolling_size: int | None = None,
    seed: int = 0, on_model_error: str = "raise", ensemble_windows: int = 8,
    chronos_checkpoint: str | None = None,
    ensemble_candidates: Sequence[str] = ("ar4", "auto_arima", "random_forest", "bridge_ridge", "mean_growth", "last_growth"),
) -> tuple[pd.DataFrame, list[dict]]:
    """Forecast at explicit origins, using only their verified information sets.

    Optional error recording preserves a row for every requested model, with NaN
    predictions and pit_validated=False on failure. No fallback or silent omission.
    Hyperparameter selection is training-only; ensemble selection replays earlier
    PIT origins. Scoring releases are never passed into this forecasting function.
    """
    if not provider.strict_pit:
        raise PITError("Strict benchmark requires strict_pit=True")
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise PITError(f"Predeclared supported models required: {MODELS}")
    if on_model_error not in {"raise", "record"}:
        raise PITError("on_model_error must be raise or record")
    if len(set(covariates)) != len(covariates) or "GDPC1" in covariates:
        raise PITError("Covariates must be unique and cannot include the GDP target")
    if any(provider.series_specs.get(c, {}).get("series_id", c) == "GDPC1" for c in covariates):
        raise PITError("A covariate cannot alias the GDP target")
    if min_train < 8 or (rolling_size is not None and rolling_size < min_train):
        raise PITError("Window must contain at least min_train >= 8 GDP levels")
    if not {"origin_date", "target_quarter"}.issubset(origins) or origins.empty:
        raise PITError("Explicit forecast origins and target quarters are required")
    if origins.duplicated(["origin_date", "target_quarter"]).any():
        raise PITError("Duplicate forecast request")
    if set(models) & set(ENSEMBLES):
        if (ensemble_windows < 4 or len(set(ensemble_candidates)) != len(ensemble_candidates)
                or len(ensemble_candidates) < 5 or set(ensemble_candidates) - (set(MODELS) - set(ENSEMBLES) - {"chronos2"})):
            raise PITError("Ensembles require >=4 earlier quarters and >=5 unique non-ensemble candidates")
    rows, audits = [], []
    config = dict(models=list(models), covariates=list(covariates), min_train=min_train,
                  rolling_size=rolling_size, series_specs=provider.series_specs,
                  covariate_mode=provider.covariate_mode, ridge_penalty=1., ar_lags=4,
                  model_selection="fixed specifications; training-only automatic orders; nested PIT ensembles",
                  seed=seed, on_model_error=on_model_error, chronos_checkpoint=chronos_checkpoint,
                  ensemble_windows=ensemble_windows, ensemble_candidates=list(ensemble_candidates))
    for request in origins.sort_values("origin_date").itertuples():
        origin, target = pd.Timestamp(request.origin_date), pd.Period(request.target_quarter, freq="Q-DEC")
        cutoff = information_date(origin)
        panel, meta = provider.build_panel_asof(asof_ts=origin, target_col="GDPC1",
                                               covariate_columns=covariates, obs_end=cutoff)
        panel = panel.set_index("quarter").sort_index()
        y = panel.GDPC1.dropna()
        if not np.isfinite(y).all() or (y <= 0).any():
            raise PITError("Invalid GDP training levels")
        if rolling_size is not None:
            y = y.iloc[-rolling_size:]
        if y.empty or not y.index.equals(pd.period_range(y.index.min(), y.index.max(), freq="Q-DEC")):
            raise PITError("Internal GDP gap: do not compress missing calendar quarters")
        if len(y) < min_train:
            raise PITError(f"Insufficient PIT GDP history at {origin.date()}: {len(y)} levels")
        last = y.index[-1]
        steps = target.ordinal - last.ordinal
        if steps < 1:
            raise PITError(f"{target} already available at origin; not an out-of-sample forecast")
        if steps > 8:
            raise PITError("Horizon exceeds the supported eight-quarter limit")
        g = np.diff(np.log(y.to_numpy()))
        index = pd.period_range(y.index[1], target, freq="Q-DEC")
        features = panel.reindex(index=index, columns=list(covariates)).copy()
        for variable in covariates:
            counts = meta["quarterly_observation_counts"].get(variable, {})
            features[variable + "__count"] = [counts.get(str(q), 0) for q in index]
            features[variable + "__missing"] = features[variable].isna().astype(float)
        fits, selection = {}, None
        def fit(name):
            nonlocal selection
            if name in fits:
                return fits[name]
            if name in COVARIATE_MODELS and (not covariates or meta["unresolved_variables"]):
                raise UnsupportedModel(f"{name} requires all predeclared covariates in the PIT store")
            if name in ENSEMBLES:
                if selection is None:
                    selection = _ensemble_validation(provider, origin, target, y, candidates=ensemble_candidates,
                        windows=ensemble_windows, covariates=covariates, min_train=min_train,
                        rolling_size=rolling_size, seed=seed)
                n = 3 if name == "ensemble_avg_top3" else 5
                selected = sorted(ensemble_candidates, key=lambda m: (selection["rmse_saar"][m], m))[:n]
                weights = np.ones(n) if n == 3 else 1. / np.maximum([selection["rmse_saar"][m] for m in selected], 1e-8)
                weights /= weights.sum()
                # Combine GDP levels and derive a coherent growth path, including h > 1.
                levels = np.array([float(y.iloc[-1]) * np.exp(np.cumsum(fit(m)[0])) for m in selected])
                average = weights @ levels
                path = np.diff(np.log(np.r_[float(y.iloc[-1]), average]))
                details = dict(selected_members=selected, weights=weights.tolist(),
                               averaging="GDP levels", selection=selection)
            elif name == "naive_last":
                path, details = [0.] * steps, {}
            elif name == "last_growth":
                path, details = [float(g[-1])] * steps, {}
            elif name == "mean_growth":
                path, details = [float(g.mean())] * steps, {}
            elif name in {"ar4", "bridge_ridge"}:
                path = _regression_path(g, steps, features if name == "bridge_ridge" else None)
                details = dict(lags=4, ridge_penalty=1. if name == "bridge_ridge" else 0.)
            else:
                data = ModelData(y=y, features=features, covariates=tuple(covariates), metadata=meta,
                                 steps=steps, seed=model_seed(seed, name, cutoff.date().isoformat(), str(target)),
                                 chronos_checkpoint=chronos_checkpoint, origin=origin.isoformat())
                path, details = forecast_model(name, data)
            path = np.asarray(path, float)
            if path.shape != (steps,) or not np.isfinite(path).all():
                raise PITError(f"Invalid forecast path from {name}")
            details["growth_path"] = path.tolist()
            fits[name] = path, details
            return fits[name]
        def input_columns(name):
            if name in ENSEMBLES:
                return list(dict.fromkeys(c for m in fits[name][1]["selected_members"] for c in input_columns(m)))
            if name.startswith("bvar_") and name in fits:
                return ["GDPC1", *fits[name][1]["covariates"]]
            return ["GDPC1", *covariates] if name in COVARIATE_MODELS else ["GDPC1"]
        pending, errors = [], {}
        for model in models:
            try:
                path, _ = fit(model)
                with np.errstate(over="raise", invalid="raise"):
                    level = float(y.iloc[-1] * np.exp(np.sum(path)))
                    saar = float(100 * np.expm1(4 * path[-1]))
                if not np.isfinite([level, saar]).all() or level <= 0:
                    raise PITError(f"Invalid forecast from {model}")
                status, error = "ok", ""
            except (PITError, ImportError, OSError, ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                if on_model_error == "raise":
                    raise PITError(f"{model} at {origin.date()}: {exc}") from exc
                level, saar = np.nan, np.nan
                status = "unsupported" if isinstance(exc, (UnsupportedModel, ImportError)) else "failed"
                error = f"{type(exc).__name__}: {exc}"
                errors[model] = dict(status=status, error=error)
            pending.append(dict(model=model, origin_date=origin.isoformat(), target_quarter=str(target),
                horizon=steps, information_cutoff=cutoff.date().isoformat(),
                training_min_quarter=str(y.index[0]), training_max_quarter=str(last), n_train=len(y),
                y_hat_level=level, g_hat_saar=saar, status=status, error=error,
                max_vintage_used=max(v["vintage_date"] for v in meta["inputs"]
                    if v["variable"] in input_columns(model) and
                    (v["variable"] != "GDPC1" or pd.Period(v["obs_date"], freq="Q") in y.index)) if status == "ok" else None,
                max_vintage_inspected=max(v["vintage_date"] for v in meta["inputs"]), pit_validated=status == "ok"))
        audit = dict(origin_date=origin.isoformat(), target_quarter=str(target), config=config,
                     training_start=str(y.index[0]), training_end=str(last), training_levels=len(y), **meta)
        audit["input_scope"] = "snapshot records inspected; fitted GDP range is training_start..training_end; nested audits retain ensemble origins"
        audit["model_input_columns"] = {m: input_columns(m) if m not in errors else [] for m in models}
        audit["model_fits"] = {name: detail for name, (_, detail) in fits.items()}
        audit["model_errors"] = errors
        audit["audit_id"] = content_hash(audit)
        audits.append(audit)
        for row in pending:
            row["audit_id"] = audit["audit_id"]
        rows.extend(pending)
    return pd.DataFrame(rows), audits


def provenance_ids(value):
    """Collect response evidence recursively, including nested ensemble validation."""
    ids = set()
    if isinstance(value, dict):
        if "provenance_id" in value:
            ids.add(value["provenance_id"])
        for child in value.values():
            ids.update(provenance_ids(child))
    elif isinstance(value, list):
        for child in value:
            ids.update(provenance_ids(child))
    return ids


def score_forecasts(forecasts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if truth.duplicated(["target_quarter", "release_stage"]).any():
        raise PITError("Duplicate release truth")
    scored = forecasts.merge(truth, on="target_quarter", how="left", validate="many_to_many")
    if scored.release_stage.isna().any():
        raise PITError("Missing truth for requested target quarter")
    origin_dates = scored.origin_date.map(lambda o: (information_date(o) + pd.Timedelta(days=1)).date())
    # Forecast API already rejects targets observed at origin. The calendar is
    # an additional check, and is used only after all forecasts have been made.
    if (pd.to_datetime(scored.release_date).dt.date < origin_dates).any():
        raise PITError("Release being scored predates forecast origin")
    return scored


def paired_metrics(scored: pd.DataFrame, baseline: str = "naive_last") -> pd.DataFrame:
    keys = ["origin_date", "target_quarter", "horizon", "release_stage"]
    if scored.duplicated(["model", *keys]).any():
        raise PITError("Duplicate forecast/score rows would overweight observations")
    base = scored.loc[scored.model == baseline, [*keys, "g_hat_saar", "g_true_saar"]]
    if base.empty:
        raise PITError("Benchmark forecasts are missing")
    rows = []
    for (model, h, stage), data in scored.groupby(["model", "horizon", "release_stage"]):
        matched = data.merge(base, on=keys, suffixes=("", "_baseline"), validate="one_to_one")
        columns = ["g_hat_saar", "g_hat_saar_baseline", "g_true_saar", "g_true_saar_baseline"]
        valid = matched.loc[np.isfinite(matched[columns]).all(axis=1)]
        if not np.allclose(valid.g_true_saar, valid.g_true_saar_baseline):
            raise PITError("Model and benchmark truth differ")
        if valid.empty:
            continue
        e, b = valid.g_hat_saar - valid.g_true_saar, valid.g_hat_saar_baseline - valid.g_true_saar
        rmse, brmse = float(np.sqrt(np.mean(e**2))), float(np.sqrt(np.mean(b**2)))
        rows.append(dict(model=model, horizon=h, release_stage=stage, n=len(valid),
                         n_requested=len(data), rmse=rmse, mae=float(np.mean(abs(e))),
                         baseline_rmse=brmse, rel_rmse=rmse / brmse if brmse else np.nan))
    return pd.DataFrame(rows)


def paired_dm(scored: pd.DataFrame, model: str, baseline: str, *, horizon: int, stage: str) -> dict:
    """Asymptotic squared-error DM with Bartlett HAC, one forecast per quarter.

    Refuse small/irregular/repeated-target samples. Monthly/daily nowcast losses
    need a different dependence treatment and cannot be pooled into this test.
    No multiple-comparison adjustment is claimed.
    """
    if horizon < 1 or stage not in STAGES or model == baseline:
        raise PITError("Invalid DM comparison")
    part = scored.loc[(scored.horizon == horizon) & (scored.release_stage == stage)]
    keys = ["origin_date", "target_quarter"]
    left, right = (part.loc[part.model == name] for name in (model, baseline))
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise PITError("Duplicate DM forecast")
    pairs = left.merge(right, on=keys, suffixes=("", "_baseline"), validate="one_to_one")
    pairs = pairs.sort_values("target_quarter")
    cols = ["g_hat_saar", "g_hat_saar_baseline", "g_true_saar", "g_true_saar_baseline"]
    if not np.isfinite(pairs[cols]).all().all() or len(pairs) < max(20, 3*horizon):
        raise PITError("DM requires at least 20 finite matched quarterly observations")
    q = pd.PeriodIndex(pairs.target_quarter, freq="Q-DEC")
    if not q.equals(pd.period_range(q.min(), q.max(), freq="Q-DEC")):
        raise PITError("DM requires unique consecutive target quarters")
    if not np.allclose(pairs.g_true_saar, pairs.g_true_saar_baseline):
        raise PITError("DM truth differs between model and benchmark")
    d = ((pairs.g_hat_saar - pairs.g_true_saar)**2 -
         (pairs.g_hat_saar_baseline - pairs.g_true_saar)**2).to_numpy()
    centered = d - d.mean()
    # Include dependence beyond mechanical h-step overlap via a fixed bandwidth rule.
    bandwidth = min(len(d)-1, max(horizon-1, int(np.floor(4*(len(d)/100)**(2/9)))))
    variance = float(centered @ centered / len(d))
    for lag in range(1, bandwidth+1):
        variance += 2 * (1-lag/(bandwidth+1)) * float(centered[lag:] @ centered[:-lag] / len(d))
    if variance <= np.finfo(float).eps:
        raise PITError("Degenerate loss differential: no valid DM statistic")
    statistic = float(d.mean() / np.sqrt(variance / len(d)))
    return dict(n=len(d), statistic=statistic, p_value=math.erfc(abs(statistic)/np.sqrt(2)),
                hac_lags=bandwidth, inference="asymptotic normal, unadjusted, squared loss")
