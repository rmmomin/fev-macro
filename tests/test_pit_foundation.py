"""Foundation adapter contracts without downloading weights or using a cloud API.

Synthetic checkpoint bytes and recording backends test routing/leakage, not model
accuracy. An opt-in test below runs the actual pinned packages and real weights.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_pit_contract import ROOT, FIXTURES, store, real_store, provider_for, ingest, obs
from fev_macro.pit import PITError
from fev_macro.pit_benchmark import run_pit_backtest
from fev_macro.pit_checkpoint import CHECKPOINT_SPECS, validate_checkpoint
from fev_macro.pit_models import ModelData, forecast_model
from fev_macro import pit_foundation as fm

NEW_MODELS = ['tabpfn_bridge', 'tabpfn_ts', 'timesfm3', 'chronos2_covariates']


def make_checkpoint(root, model, published='2010-01-01T12:00:00Z'):
    """Synthetic evidence only, never usable for real model inference."""
    spec = CHECKPOINT_SPECS[model]
    directory = root / model / 'checkpoint'
    directory.mkdir(parents=True)
    files = {}
    for name in spec['files']:
        body = b'synthetic test bytes: ' + name.encode()
        (directory / name).write_bytes(body)
        sha = hashlib.sha256(body).hexdigest()
        files[name] = dict(sha256=sha, publisher_lfs_sha256=sha,
            publisher_git_blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest())
    revision = 'a'*40
    record = dict(repository=spec['repository'], revision=revision, directory='checkpoint', files=files,
        publication_evidence=dict(commit_id=revision, created_at=published,
            source_url=f"https://huggingface.co/{spec['repository']}/commit/{revision}"))
    manifest = directory.parent / 'manifest.json'
    manifest.write_text(json.dumps(record))
    return str(manifest)


@pytest.fixture
def manifests(tmp_path):
    return {name: make_checkpoint(tmp_path, name) for name in CHECKPOINT_SPECS}


def options(manifests):
    return dict(chronos_checkpoint=manifests['chronos2'],
                foundation_checkpoints={k:v for k,v in manifests.items() if k != 'chronos2'},
                model_use='research')


@pytest.fixture
def recordings(monkeypatch):
    seen = dict(bridge_fit=[], bridge_predict=[], ts=[], timesfm=[], chronos=[])
    monkeypatch.setattr(fm, 'local_cpu', lambda seed: nullcontext())

    class Regressor:
        def __init__(self, **kwargs): self.params = kwargs
        def fit(self, x, y):
            seen['bridge_fit'].append((x.copy(), y.copy(), deepcopy(self.params)))
            self.mean = float(y.mean())
            return self
        def predict(self, x, **kwargs):
            assert kwargs == dict(output_type='median')
            seen['bridge_predict'].append(x.copy())
            return self.mean + np.nansum(x, axis=1)*1e-5

    class TSPipeline:
        def __init__(self, **kwargs): self.params = kwargs
        def predict_df(self, context, future_df, quantiles):
            seen['ts'].append((context.copy(), future_df.copy(), self.params))
            result = future_df.copy()
            result['target'] = context.target.iloc[-1] + .005*np.arange(1,len(result)+1)
            return result.set_index(['item_id','timestamp']).iloc[::-1]

    class TimesFM:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert Path(path).is_dir() and kwargs['local_files_only'] and kwargs['device'] == 'cpu'
            return cls()
        def predict(self, **kwargs):
            seen['timesfm'].append(deepcopy(kwargs))
            return SimpleNamespace(forecast=kwargs['context'][-1]+.004*np.arange(1,kwargs['horizon']+1))

    class Tensor:
        def __init__(self, a): self.a = np.asarray(a)
        def __getitem__(self, key): return Tensor(self.a[key])
        def numpy(self, force=False): return self.a

    class Chronos:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert Path(path).is_dir() and kwargs == dict(device_map='cpu', local_files_only=True)
            return cls()
        def predict_quantiles(self, tasks, prediction_length, quantile_levels):
            seen['chronos'].append(deepcopy(tasks[0]))
            task = tasks[0]
            p = np.full(prediction_length, np.mean(task['target']))
            for a in task['future_covariates'].values(): p += np.nan_to_num(a)*1e-5
            return [Tensor(p[None,:,None])], None

    monkeypatch.setitem(sys.modules, 'tabpfn', SimpleNamespace(TabPFNRegressor=Regressor))
    monkeypatch.setitem(sys.modules, 'tabpfn_time_series', SimpleNamespace(TabPFNTSPipeline=TSPipeline, TabPFNMode=SimpleNamespace(LOCAL='local')))
    monkeypatch.setitem(sys.modules, 'tabpfn_time_series.features', SimpleNamespace(
        RunningIndexFeature=lambda:'index', CalendarFeature=lambda:'calendar', AutoSeasonalFeature=lambda:'season'))
    monkeypatch.setitem(sys.modules, 'timesfm3', SimpleNamespace(TimesFM3Forecaster=TimesFM))
    monkeypatch.setitem(sys.modules, 'chronos', SimpleNamespace(Chronos2Pipeline=Chronos))
    return seen


@pytest.mark.parametrize('model', NEW_MODELS)
@pytest.mark.parametrize('representation', ['quarterly', 'monthly_slots'])
def test_new_adapters_ignore_future_revisions_and_same_day_releases(real_store, manifests, recordings, model, representation):
    p = provider_for(real_store, covariate_mode='processed')
    origin = pd.DataFrame([dict(origin_date='2019-02-15', target_quarter='2019Q1')])
    kwargs = dict(models=[model], covariates=['UNRATE'], foundation_features=representation, **options(manifests))
    try:
        before, a = run_pit_backtest(p, origin, **kwargs)
        ingest(real_store, 'GDPC1', [obs('2018-07-01','2019-02-15',1e8),obs('2019-01-01','2021-01-01',1e9)])
        ingest(real_store, 'UNRATE', [obs('2019-02-01','2019-02-15',1e9),obs('2018-10-01','2026-01-01',1e9)])
        after, b = run_pit_backtest(p, origin, **kwargs)
        pd.testing.assert_frame_equal(before, after)
        assert a == b and before.status.tolist() == ['ok']
        path = a[0]['model_fits'][model]['growth_path']
        assert before.checkpoint_revision.iloc[0] == 'a'*40
        assert before.model_use.iloc[0] == 'research'
        assert before.horizon.iloc[0] == len(path) == 2
        assert before.g_hat_saar.iloc[0] == pytest.approx(100*np.expm1(4*path[-1]))
        json.dumps(a, allow_nan=False)
    finally:
        p.close()


def test_bridge_fits_only_training_rows_and_recurses_without_future_targets(real_store, manifests, recordings):
    p = provider_for(real_store, covariate_mode='processed')
    origin = pd.DataFrame([dict(origin_date='2019-02-15',target_quarter='2019Q1')])
    kwargs = dict(models=['tabpfn_bridge'], covariates=['UNRATE'], rolling_size=32, **options(manifests))
    try:
        f,a = run_pit_backtest(p,origin,**kwargs)
        x,y,params = recordings['bridge_fit'][0]
        assert x.shape == (27,7) and y.shape == (27,)
        assert params['device'] == 'cpu' and params['random_state'] is not None
        assert np.array_equal(x[1:,0], y[:-1])  # most recent lag precedes its own target
        first,second = recordings['bridge_predict'][:2]
        assert second[0,0] == pytest.approx(a[0]['model_fits']['tabpfn_bridge']['growth_path'][0])
        ingest(real_store, 'UNRATE', [obs('2019-01-01','2019-02-14',1e5)])
        changed,b = run_pit_backtest(p,origin,**kwargs)
        np.testing.assert_equal(x,recordings['bridge_fit'][1][0])
        np.testing.assert_equal(y,recordings['bridge_fit'][1][1])
        assert a[0]['model_fits']['tabpfn_bridge']['training_design_sha256'] == b[0]['model_fits']['tabpfn_bridge']['training_design_sha256']
        assert changed.g_hat_saar.iloc[0] != f.g_hat_saar.iloc[0]
    finally:
        p.close()


def test_chronos_passes_ragged_known_features_and_preserves_unavailable_values(real_store, manifests, recordings):
    p = provider_for(real_store, covariate_mode='processed')
    origin = pd.DataFrame([dict(origin_date='2019-02-15',target_quarter='2019Q2')])
    try:
        _,a = run_pit_backtest(p,origin,models=['chronos2_covariates'],covariates=['UNRATE'],**options(manifests))
        task = recordings['chronos'][0]
        assert set(task['past_covariates']) == set(task['future_covariates']) == {'UNRATE','UNRATE__count','UNRATE__missing'}
        assert all(len(v)==len(task['target']) for v in task['past_covariates'].values())
        # Q4 GDP is still unreleased; Q1 has January only; Q2 has no observations.
        assert task['future_covariates']['UNRATE__count'].tolist() == [3.,1.,0.]
        assert np.isnan(task['future_covariates']['UNRATE'][-1])
        assert task['future_covariates']['UNRATE__missing'][-1] == 1
        assert a[0]['model_input_columns']['chronos2_covariates'] == ['GDPC1','UNRATE']
        detail = a[0]['model_fits']['chronos2_covariates']
        assert detail['forecast_features']['UNRATE'][-1] is None
        assert detail['fit_target'] == 'quarterly log GDP growth'
    finally:
        p.close()


@pytest.mark.parametrize('model', ['tabpfn_ts','timesfm3'])
def test_univariate_comparators_never_consume_macro_features(real_store, manifests, recordings, model):
    p = provider_for(real_store)
    origin = pd.read_csv(FIXTURES/'origins.csv').iloc[:1]
    try:
        before,a = run_pit_backtest(p,origin,models=[model],covariates=['UNRATE'],**options(manifests))
        ingest(real_store,'UNRATE',[obs('2019-01-01','2019-04-24',1e9)])
        after,b = run_pit_backtest(p,origin,models=[model],covariates=['UNRATE'],**options(manifests))
        assert before.g_hat_saar.iloc[0] == after.g_hat_saar.iloc[0]
        assert a[0]['model_input_columns'][model] == ['GDPC1']
        assert a[0]['model_fits'][model]['covariates'] == []
        if model == 'tabpfn_ts':
            context,future,params = recordings['ts'][0]
            assert list(context.columns) == ['item_id','timestamp','target']
            assert list(future.columns) == ['item_id','timestamp']
            assert params['tabpfn_mode'] == 'local'
    finally:
        p.close()


@pytest.mark.parametrize('model', NEW_MODELS)
def test_missing_checkpoint_records_unsupported_without_inference(real_store, model, recordings):
    p = provider_for(real_store)
    try:
        f,a = run_pit_backtest(p,pd.read_csv(FIXTURES/'origins.csv').iloc[:1],models=[model],
            covariates=['UNRATE'],on_model_error='record',model_use='research')
        assert f.status.tolist() == ['unsupported'] and f.g_hat_saar.isna().all()
        assert not f.pit_validated.any() and not any(recordings.values())
    finally:
        p.close()


@pytest.mark.parametrize('model', list(CHECKPOINT_SPECS))
def test_checkpoint_date_gate_including_same_new_york_day(tmp_path, model):
    path = make_checkpoint(tmp_path,model,published='2026-09-03T23:00:00Z')
    with pytest.raises(PITError,match='not available'):
        validate_checkpoint(path,'2026-09-03T23:30:00-04:00',model=model,model_use='research')
    _,evidence = validate_checkpoint(path,'2026-09-04',model=model,model_use='research')
    assert evidence['manifest']['publication_evidence']['created_at'] == '2026-09-03T23:00:00Z'


@pytest.mark.parametrize('model', ['tabpfn_bridge','tabpfn_ts','timesfm3'])
def test_restricted_checkpoints_cannot_enter_production_even_with_edited_manifest(manifests, model):
    path = Path(manifests[model]);record=json.loads(path.read_text())
    record['research_only'] = False;record['license'] = 'Apache-2.0';path.write_text(json.dumps(record))
    with pytest.raises(PITError,match='model_use=research'):
        validate_checkpoint(path,'2026-09-04',model=model)


@pytest.mark.parametrize('model', list(CHECKPOINT_SPECS))
def test_checkpoint_tampering_and_extra_files_rejected(manifests, model):
    path = Path(manifests[model]);directory=path.parent/'checkpoint'
    (directory/'unrecorded-config').mkdir()
    with pytest.raises(PITError,match='unrecorded'):
        validate_checkpoint(path,'2026-09-04',model=model,model_use='research')
    (directory/'unrecorded-config').rmdir()
    filename=next(f for f in CHECKPOINT_SPECS[model]['files'] if f.endswith(('.safetensors','.ckpt')))
    (directory/filename).write_bytes(b'tampered')
    with pytest.raises(PITError,match='hash mismatch'):
        validate_checkpoint(path,'2026-09-04',model=model,model_use='research')


def test_tabpfn_ts_cannot_substitute_the_generic_regressor_checkpoint(manifests):
    with pytest.raises(PITError,match='Exact checkpoint files'):
        validate_checkpoint(manifests['tabpfn_bridge'],'2026-09-04',model='tabpfn_ts',model_use='research')


@pytest.mark.parametrize('field,value', [('publication_evidence',None),('files',None),('directory',None),('revision',None)])
def test_malformed_checkpoint_metadata_fails_closed(manifests, field, value):
    path=Path(manifests['timesfm3']);record=json.loads(path.read_text())
    record[field]=value;path.write_text(json.dumps(record))
    with pytest.raises(PITError):
        validate_checkpoint(path,'2026-09-04',model='timesfm3',model_use='research')


@pytest.mark.parametrize('bad', ['shape','nonfinite','wrong_calendar','duplicate'])
def test_invalid_backend_outputs_remain_failed_rows(real_store, manifests, recordings, monkeypatch, bad):
    if bad in {'shape','nonfinite'}:
        name='timesfm3'
        def predict(self, **kwargs):
            h=kwargs['horizon']
            return SimpleNamespace(forecast=np.zeros((h,1)) if bad=='shape' else np.full(h,np.nan))
        monkeypatch.setattr(sys.modules['timesfm3'].TimesFM3Forecaster,'predict',predict)
    else:
        name='tabpfn_ts'
        def predict(self, context, future_df, quantiles):
            result=future_df.copy();result['target']=1.
            if bad=='wrong_calendar': result['timestamp']+=pd.Timedelta(days=1)
            else: result=pd.concat([result,result.iloc[:1]])
            return result.set_index(['item_id','timestamp'])
        monkeypatch.setattr(sys.modules['tabpfn_time_series'].TabPFNTSPipeline,'predict_df',predict)
    p=provider_for(real_store)
    try:
        f,a=run_pit_backtest(p,pd.read_csv(FIXTURES/'origins.csv').iloc[:1],models=[name],
                            on_model_error='record',**options(manifests))
        assert f.status.tolist()==['failed'] and f.g_hat_saar.isna().all()
        assert not f.pit_validated.any() and name in a[0]['model_errors']
    finally:
        p.close()


def test_download_hash_failure_cannot_create_checkpoint_manifest(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from prepare_foundation_checkpoint import prepare_checkpoint
    spec=CHECKPOINT_SPECS['timesfm3'];revision='a'*40
    files=[]
    for name in spec['files']:
        path=tmp_path/name;path.write_bytes(b'wrong bytes')
        files.append(SimpleNamespace(rfilename=name,blob_id='a'*40,
            lfs=SimpleNamespace(sha256='b'*64) if name.endswith('safetensors') else None))
    class API:
        def list_repo_commits(self, repo, revision):
            return [SimpleNamespace(commit_id=revision,created_at=datetime(2020,1,1,tzinfo=timezone.utc))]
        def model_info(self,*a,**k):return SimpleNamespace(siblings=files)
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(HfApi=API,
        hf_hub_download=lambda repo,name,**kw:str(tmp_path/name)))
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='Publisher hash mismatch'):
        prepare_checkpoint('timesfm3',revision,'2026-09-04',out,model_use='research')
    assert not out.exists()


def test_cli_registers_all_adapters_and_explicit_research_use():
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_pit_backtest.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0
    for name in NEW_MODELS: assert name in result.stdout
    assert '--model-use' in result.stdout and '--tabpfn-ts-checkpoint' in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize('model,representation', [(m,'quarterly') for m in NEW_MODELS] +
                         [(m,'monthly_slots') for m in ['tabpfn_bridge','chronos2_covariates']])
def test_actual_pinned_weights_with_network_disabled(model, representation, monkeypatch):
    """FEV_FOUNDATION_CHECKPOINTS points to a JSON map of existing local manifests."""
    config=os.environ.get('FEV_FOUNDATION_CHECKPOINTS')
    if not config: pytest.skip('Optional real foundation checkpoints not configured')
    manifests=json.loads(Path(config).read_text())
    # Disable even incidental socket traffic during imports and inference.
    def offline(*a,**k): raise AssertionError('Network access during pinned local inference')
    monkeypatch.setattr(socket.socket,'connect',offline)
    monkeypatch.setattr(socket.socket,'connect_ex',offline)
    rng=np.random.default_rng(25)
    q=pd.period_range('2010Q1',periods=64,freq='Q-DEC')
    y=pd.Series(100*np.exp(np.cumsum(.005+rng.normal(0,.002,len(q)))),index=q)
    index=pd.period_range(q[1],q[-1]+2,freq='Q-DEC')
    features=pd.DataFrame({'X':rng.normal(size=len(index)), 'X__count':3.,'X__missing':0.},index=index)
    features.iloc[-1]=[np.nan,0.,1.]
    monthly = pd.period_range(q[0].asfreq('M','start'), (q[-1]+1).asfreq('M','end'), freq='M')
    inputs = [dict(variable='X',obs_date=str(m.start_time.date()),vintage_date=str((m+1).start_time.date()),
                   frequency='M',tcode=1,value=float(rng.normal())) for m in monthly]
    data=ModelData(y,features,('X',),{'information_cutoff':'2026-09-03','inputs':inputs},2,
        seed=13,origin='2026-09-04',foundation_features=representation,**options(manifests))
    path,evidence=forecast_model(model,data)
    assert path.shape==(2,) and np.isfinite(path).all()
    assert evidence['checkpoint']['manifest']['revision'] != 'a'*40
    assert evidence['device']=='cpu'
