"""Optional foundation adapters: pinned local inference on a single PIT snapshot.

Forecast-horizon covariates mean *features already known at the origin*, not
future releases. A partial-quarter mean is accompanied by its release count and
missing indicator. Unavailable values stay NaN; no completion is fetched.
"""
from __future__ import annotations

from contextlib import contextmanager
import os

import numpy as np
import pandas as pd

from .pit import PITError
from .pit_checkpoint import CHECKPOINT_SPECS, validate_checkpoint
from .pit_models import ModelData, UnsupportedModel, _array_hash, _level_to_growth, _supervised


@contextmanager
def local_cpu(seed):
    # Set before optional imports too: TabPFN-TS otherwise enables telemetry.
    values = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TABPFN_DISABLE_TELEMETRY='1', TABPFN_NO_BROWSER='1')
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        import torch
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                yield
        finally:
            torch.set_num_threads(threads)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _checkpoint(name, d):
    key = 'chronos2' if name == 'chronos2_covariates' else name
    manifest = d.chronos_checkpoint if key == 'chronos2' else (d.foundation_checkpoints or {}).get(key)
    if not manifest:
        flag = 'chronos' if key == 'chronos2' else key.replace('_', '-')
        raise UnsupportedModel(f'{name} requires --{flag}-checkpoint with pinned publisher evidence')
    try:
        return validate_checkpoint(manifest, d.origin, model=key, model_use=d.model_use)
    except (PITError, OSError, KeyError, TypeError, ValueError) as exc:
        raise UnsupportedModel(f'Inadmissible {name} checkpoint: {exc}') from exc


def _features(d):
    """An explicit regular quarterly design; monthly slots retain calendar position.

    Raw observations already come from the verified origin snapshot. Transform
    on the complete native calendar BEFORE slicing the training window. Never
    fill unpublished months, interpolate GDP, or compress gaps in that calendar.
    """
    expected = pd.period_range(d.y.index[1], d.y.index[-1] + d.steps, freq='Q-DEC')
    columns = [*d.covariates, *[v + suffix for v in d.covariates for suffix in ('__count', '__missing')]]
    if not d.features.index.equals(expected) or list(d.features.columns) != columns:
        raise PITError('Foundation covariates must have explicit aligned quarters, counts and missing indicators')
    if not d.covariates:
        raise UnsupportedModel('Verified covariates are required')
    if d.foundation_features == 'quarterly':
        frame = d.features.copy()
    elif d.foundation_features == 'monthly_slots':
        from .fred_transforms import fred_transform
        frame = pd.DataFrame(index=expected)
        for variable in d.covariates:
            records = [v for v in d.metadata.get('inputs', []) if v['variable'] == variable]
            if not records:
                raise UnsupportedModel(f'No verified native observations for {variable}')
            specs = {(r['frequency'], r['tcode']) for r in records}
            if len(specs) != 1:
                raise PITError(f'Inconsistent native specification for {variable}')
            frequency, code = specs.pop()
            if frequency not in {'M', 'Q'}:
                raise UnsupportedModel(f'Unsupported native frequency for {variable}')
            cutoff = d.metadata['information_cutoff']
            if any(r['vintage_date'] > cutoff or r['obs_date'] > cutoff for r in records):
                raise PITError('Native features exceed the information cutoff')
            native = pd.Series([r['value'] for r in records],
                index=pd.PeriodIndex([r['obs_date'] for r in records], freq=frequency), dtype=float).sort_index()
            if native.index.has_duplicates:
                raise PITError(f'Duplicate native period for {variable}')
            native = native.reindex(pd.period_range(native.index.min(), native.index.max(), freq=frequency))
            transformed = fred_transform(native, code)
            for slot in range(1, 4) if frequency == 'M' else [0]:
                periods = expected.asfreq('M', how='start') + slot - 1 if slot else expected
                name = f'{variable}__m{slot}' if slot else f'{variable}__q'
                frame[name] = transformed.reindex(periods).to_numpy()
                frame[name + '__missing'] = frame[name].isna().astype(float)
    else:
        raise PITError('Unknown foundation feature representation')
    if np.isinf(frame.to_numpy(float)).any():
        raise PITError('Infinite foundation feature')
    return frame


def _frame_evidence(d, frame):
    f = frame.to_numpy(float)
    return dict(feature_representation=d.foundation_features, feature_columns=list(frame.columns),
                feature_quarters=frame.index.astype(str).tolist(),
                feature_sha256=_array_hash(f),
                context_feature_sha256=_array_hash(f[:len(d.g)]),
                forecast_features=frame.iloc[len(d.g):].astype(object).where(
                    frame.iloc[len(d.g):].notna(), None).to_dict(orient='list'),
                covariate_rule=('native transformation then calendar month 1/2/3 values and missing indicators; '
                    'quarterly covariates remain quarterly; no filling or future releases'
                    if d.foundation_features == 'monthly_slots' else
                    'origin snapshot partial-quarter summaries, counts and missing indicators; no future releases'))


def _tabpfn_config(directory, name, seed):
    filename = next(f for f in CHECKPOINT_SPECS[name]['files'] if f.endswith('.ckpt'))
    return dict(model_path=str(directory / filename), device='cpu', random_state=seed,
                n_estimators=4, n_preprocessing_jobs=1, fit_mode='fit_preprocessors')


def _tabpfn_bridge(d, directory):
    from tabpfn import TabPFNRegressor
    frame, lags = _features(d), 4
    f = frame.to_numpy(float)
    x, target = _supervised(d.g, f, lags)
    params = _tabpfn_config(directory, 'tabpfn_bridge', d.seed)
    estimator = TabPFNRegressor(**params).fit(x, target)
    history = list(d.g)
    for j in range(d.steps):
        row = np.r_[history[-lags:][::-1], f[len(d.g)+j]]
        value = np.asarray(estimator.predict(row[None], output_type='median'), float)
        if value.shape != (1,) or not np.isfinite(value).all():
            raise PITError('TabPFN bridge produced an invalid growth prediction')
        history.append(float(value[0]))
    return history[-d.steps:], dict(parameters=params, lags=lags, fit_rows=len(target),
        fit_target='quarterly log GDP growth', point_statistic='median',
        training_design_sha256=_array_hash(x), training_target_sha256=_array_hash(target),
        preprocessing='TabPFN preprocessing fitted on training rows; no external scaling or imputation',
        **_frame_evidence(d, frame))


def _tabpfn_ts(d, directory):
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode
    from tabpfn_time_series.features import RunningIndexFeature, CalendarFeature, AutoSeasonalFeature
    params = _tabpfn_config(directory, 'tabpfn_ts', d.seed)
    pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL, tabpfn_model_config=params,
        max_context_length=len(d.y), tabpfn_output_selection='median',
        temporal_features=[RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()])
    context = pd.DataFrame(dict(item_id='GDPC1', timestamp=d.y.index.to_timestamp(), target=np.log(d.y.to_numpy())))
    quarters = pd.period_range(d.y.index[-1]+1, periods=d.steps, freq='Q-DEC')
    future = pd.DataFrame(dict(item_id='GDPC1', timestamp=quarters.to_timestamp()))
    result = pipeline.predict_df(context, future_df=future, quantiles=[.5]).reset_index()
    # Match explicit calendar timestamps; never rely on the wrapper's returned order.
    if len(result) != d.steps or result.duplicated(['item_id', 'timestamp']).any():
        raise PITError('TabPFN-TS returned duplicate or missing forecast rows')
    if set(result.item_id) != {'GDPC1'} or set(pd.to_datetime(result.timestamp)) != set(future.timestamp):
        raise PITError('TabPFN-TS returned the wrong target calendar')
    values = result.set_index('timestamp').loc[future.timestamp, 'target'].to_numpy(float)
    return _level_to_growth(values, d.y.iloc[-1]), dict(parameters=params,
        fit_target='log GDP level', point_statistic='median', covariates=[], context_length=len(context),
        temporal_features=['RunningIndexFeature', 'CalendarFeature', 'AutoSeasonalFeature'],
        preprocessing='official TabPFN-TS context-only feature extraction; contiguous GDP; explicit future calendar',
        context_sha256=_array_hash(context.target), forecast_quarters=quarters.astype(str).tolist())


def chronos_covariate_task(d):
    """Quarterly growth and origin-known features on both sides of last GDP release."""
    frame, n = _features(d), len(d.g)
    f = frame.to_numpy(float)
    return dict(target=d.g.astype(np.float32),
        past_covariates={c: f[:n, i].astype(np.float32) for i, c in enumerate(frame)},
        future_covariates={c: f[n:, i].astype(np.float32) for i, c in enumerate(frame)}), frame


def _chronos_covariates(d, directory):
    from chronos import Chronos2Pipeline
    task, f = chronos_covariate_task(d)
    pipeline = Chronos2Pipeline.from_pretrained(str(directory), device_map='cpu', local_files_only=True)
    quantiles, _ = pipeline.predict_quantiles([task], prediction_length=d.steps, quantile_levels=[.5])
    path = quantiles[0][0, :, 0].numpy(force=True)
    return path, dict(fit_target='quarterly log GDP growth', point_statistic='median', zero_shot=True,
                     context_length=len(d.g), context_sha256=_array_hash(task['target']),
                     alignment='quarterly target grid with explicit origin-known covariate channels',
                     **_frame_evidence(d, f))


def _timesfm3(d, directory):
    from timesfm3 import TimesFM3Forecaster
    context = np.log(d.y.to_numpy()).astype(np.float32)
    model = TimesFM3Forecaster.from_pretrained(str(directory), device='cpu', local_files_only=True,
                                             per_core_batch_size=1)
    result = model.predict(context=context, horizon=d.steps, return_quantiles=False,
        use_symmetric_averaging=False, make_positive=False, sort_quantiles=True,
        use_znorm=False, padding_mode='none')
    values = np.asarray(result.forecast, float)
    if values.shape != (d.steps,):
        raise PITError(f'TimesFM 3 returned unexpected forecast shape: {values.shape}')
    return _level_to_growth(values, d.y.iloc[-1]), dict(fit_target='log GDP level',
        point_statistic='median', zero_shot=True, covariates=[], context_length=len(context),
        context_sha256=_array_hash(context),
        parameters=dict(per_core_batch_size=1, use_symmetric_averaging=False, make_positive=False,
                        sort_quantiles=True, use_znorm=False, padding_mode='none'),
        scope='univariate GDP comparator; no macro covariates or monthly interpolation')


def foundation_forecast(name: str, d: ModelData):
    directory, evidence = _checkpoint(name, d)  # admission precedes optional imports/inference
    adapters = dict(tabpfn_bridge=_tabpfn_bridge, tabpfn_ts=_tabpfn_ts,
                    timesfm3=_timesfm3, chronos2_covariates=_chronos_covariates)
    with local_cpu(d.seed):
        path, details = adapters[name](d, directory)
    details.update(checkpoint=evidence, device='cpu', inference_mode='local pinned weights',
                   information_cutoff=d.metadata['information_cutoff'])
    return path, details
