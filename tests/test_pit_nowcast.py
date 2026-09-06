"""Calendar-preserving features and independently checked data freshness."""
from copy import deepcopy
import json
import sys

import numpy as np
import pandas as pd
import pytest

from test_pit_contract import store, real_store, provider_for, ingest, obs, sync, FIXTURES
from test_pit_foundation import manifests, recordings, options
from fev_macro.pit import PITError
from fev_macro.pit_models import ModelData
from fev_macro.pit_benchmark import run_pit_backtest
from fev_macro.pit_foundation import _features
from fev_macro.pit_freshness import check_freshness


def design(records, code=1, frequency='M'):
    y = pd.Series([100., 101.], index=pd.period_range('2020Q3', periods=2, freq='Q'))
    index = pd.period_range('2020Q4', '2021Q2', freq='Q')
    frame = pd.DataFrame({'X': 3., 'X__count': 3., 'X__missing': 0.}, index=index)
    inputs = [dict(variable='X', obs_date=d, vintage_date=v, value=x, frequency=frequency, tcode=code)
              for d, v, x in records]
    return ModelData(y, frame, ('X',), dict(information_cutoff='2021-04-30', inputs=inputs),
                     2, foundation_features='monthly_slots')


def test_slots_preserve_information_destroyed_by_quarterly_mean():
    a = design([(f'2021-0{i}-01', '2021-04-20', x) for i, x in enumerate([1., 3., 5.], 1)])
    b = deepcopy(a)
    for r, x in zip(b.metadata['inputs'], [5., 3., 1.]): r['value'] = x
    pd.testing.assert_frame_equal(a.features, b.features)
    fa, fb = _features(a), _features(b)
    assert fa.loc['2021Q1', 'X__m1'] == 1. and fb.loc['2021Q1', 'X__m1'] == 5.
    assert not fa.equals(fb)
    assert fa.loc['2021Q2', 'X__m1__missing'] == 1.


def test_native_growth_crosses_quarter_boundary_and_never_compresses_gap():
    d = design([('2020-12-01', '2021-01-10', 100.), ('2021-01-01', '2021-02-10', 110.),
                ('2021-03-01', '2021-04-10', 150.)], code=5)
    f = _features(d)
    assert f.loc['2021Q1', 'X__m1'] == pytest.approx(np.log(1.1))
    assert np.isnan(f.loc['2021Q1', 'X__m2']) and np.isnan(f.loc['2021Q1', 'X__m3'])
    assert f.loc['2021Q1', 'X__m3__missing'] == 1.
    # Quarterly inputs remain one quarterly channel; GDP is never interpolated.
    q = _features(design([('2020-10-01','2021-01-10',10.),('2021-01-01','2021-04-20',11.)], 5, 'Q'))
    assert list(q.columns) == ['X__q', 'X__q__missing']
    assert q.loc['2021Q1','X__q'] == pytest.approx(np.log(1.1))


@pytest.mark.parametrize('date,vintage', [('2021-05-01','2021-04-10'), ('2021-01-01','2021-05-01')])
def test_native_feature_defense_rejects_future_records(date, vintage):
    with pytest.raises(PITError, match='cutoff'):
        _features(design([(date, vintage, 1.)]))


def test_slots_reject_duplicate_native_period():
    with pytest.raises(PITError, match='Duplicate'):
        _features(design([('2021-01-01','2021-02-01',1.),('2021-01-15','2021-02-01',2.)]))


@pytest.mark.parametrize('model', ['tabpfn_bridge','chronos2_covariates'])
def test_real_release_calendar_ragged_slots_and_provenance(real_store, manifests, recordings, model):
    p = provider_for(real_store, covariate_mode='processed')
    try:
        f, a = run_pit_backtest(p, pd.DataFrame([dict(origin_date='2019-02-15',target_quarter='2019Q2')]),
            models=[model], covariates=['UNRATE'], foundation_features='monthly_slots', **options(manifests))
    finally:
        p.close()
    detail = a[0]['model_fits'][model]
    future = detail['forecast_features']
    assert f.foundation_features.tolist() == ['monthly_slots']
    assert len(detail['feature_columns']) == 6
    assert future['UNRATE__m1__missing'] == [0.,0.,1.]
    assert future['UNRATE__m2__missing'] == [0.,1.,1.]
    assert future['UNRATE__m3__missing'] == [0.,1.,1.]
    assert future['UNRATE__m2'][1:] == [None, None]
    if model == 'tabpfn_bridge':
        assert recordings['bridge_fit'][0][0].shape[1] == 10  # 4 GDP lags + 6 channels
    else:
        assert set(recordings['chronos'][0]['future_covariates']) == set(detail['feature_columns'])
    json.dumps(a, allow_nan=False)


def freshness(store, expected, *, bad_count=False):
    def fetch(params):
        assert params['realtime_start'] == params['realtime_end'] == '2021-04-30'
        assert params['observation_end'] == '2021-04-30' and params['output_type'] == 1
        rows = [obs(d, '2021-04-30', v, '2021-04-30') for d,v in expected.items()]
        return dict(realtime_start='2021-04-30',realtime_end='2021-04-30',output_type=1,units='lin',
                    count=len(rows)+int(bad_count),offset=0,observations=rows)
    return check_freshness(store, origin='2021-05-01', series_specs={'X': {'frequency':'M'}},
        observation_start='2021-01-01',fetch=fetch,retrieved_at='2021-05-01T12:00:00Z')


def test_freshness_checks_old_revisions_even_if_latest_observation_matches(store):
    ingest(store,'X',[obs('2021-01-01','2021-02-01',1.),obs('2021-02-01','2021-03-01',2.)])
    report,evidence = freshness(store, {'2021-01-01':9., '2021-02-01':2.})
    r = report['series'][0]
    assert not report['all_fresh'] and r['status'] == 'stale' and r['different_observations'] == 1
    assert r['local_latest_observation'] == r['alfred_latest_observation']
    assert evidence[0]['sha256']
    ingest(store,'X',[obs('2021-01-01','2021-04-01',9.)])
    assert freshness(store, {'2021-01-01':9., '2021-02-01':2.})[0]['all_fresh']


def test_freshness_withdrawal_future_revision_and_partial_response(store):
    ingest(store,'X',[obs('2021-01-01','2021-02-01',1.),obs('2021-02-01','2021-03-01',2.),
                      obs('2021-02-01','2021-05-01',999.)])
    assert freshness(store, {'2021-01-01':1., '2021-02-01':2.})[0]['all_fresh']
    assert not freshness(store, {'2021-01-01':None, '2021-02-01':2.})[0]['all_fresh']
    assert freshness(store, {'2021-01-01':1.}, bad_count=True)[0]['series'][0]['status'] == 'unverified'
    assert freshness(store, {})[0]['series'][0]['status'] == 'unsupported'


def test_explicit_sync_does_not_reuse_alias_or_ignore_frequency(store, tmp_path, monkeypatch):
    store.upsert_alias(variable_name='X', universe='qd', series_id='WRONG')
    specs = tmp_path/'specs.json'; specs.write_text(json.dumps({'X':{'frequency':'M'}}))
    report = tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['sync', '--db', str(store.db_path),'--series-specs',str(specs),'--report_json',str(report)])
    monkeypatch.setattr(sync,'resolve_api_key',lambda args:'synthetic')
    def metadata(**kwargs):
        assert kwargs['series_id'] == 'X'
        return dict(id='X',frequency_short='Q')
    monkeypatch.setattr(sync,'fred_series_meta',metadata)
    monkeypatch.setattr(sync,'sync_series_intervals',lambda **kw: pytest.fail('Cannot sync wrong frequency'))
    assert sync.main() == 2
    assert 'frequency mismatch' in json.loads(report.read_text())['failures']['qd:X']


@pytest.mark.parametrize('sid,observation,release,value', [
    ('BOPTEXP','2026-07-01','2026-09-03',310723.),
    ('TTLCONS','2026-07-01','2026-09-01',2157581.),
    ('AMDMTI','2026-07-01','2026-08-26',604357.),
    ('JTSJOL','2026-07-01','2026-09-01',7271.),
    ('ADPMNUSNERSA','2026-08-01','2026-09-02',132803000.),
    ('A261RX1Q020SBEA','2026-04-01','2026-08-26',24089.624),
    ('ULCNFB','2026-04-01','2026-08-06',124.022),
    ('AMINVTS','2026-07-01','2026-08-27',959084.),
    ('ARINVTS','2026-07-01','2026-08-27',838457.),
    ('TOTALSA','2026-08-01','2026-09-04',17.19),
])
def test_captured_nowcast_release_series_first_day_excluded(store, sid, observation, release, value):
    record = json.loads((FIXTURES/(sid.lower()+'_2026_intervals.json')).read_text())
    store.ingest_alfred_response(record['response'],record['params'],retrieved_at=record['retrieved_at'])
    kwargs = dict(series_ids=[sid],obs_start=observation,obs_end=observation)
    assert store.snapshot_long(asof_ts=release,**kwargs).empty
    assert store.snapshot_long(asof_ts=pd.Timestamp(release)+pd.Timedelta(days=1),**kwargs).value.iloc[0] == value
    if sid == 'AMDMTI':
        assert store.snapshot_long(asof_ts='2026-09-02',**kwargs).value.iloc[0] == 604357.
        assert store.snapshot_long(asof_ts='2026-09-03',**kwargs).value.iloc[0] == 604669.


@pytest.mark.parametrize('sid',['AMINVTS','ARINVTS'])
def test_advance_inventory_snapshot_is_not_a_full_training_history(store, sid):
    record = json.loads((FIXTURES/(sid.lower()+'_2026_snapshot.json')).read_text())
    store.ingest_alfred_response(record['response'],record['params'],retrieved_at=record['retrieved_at'])
    frame = store.snapshot_long(asof_ts='2026-09-06',series_ids=[sid])
    valid = frame.loc[frame.value.notna()]
    assert valid.obs_ts.dt.strftime('%Y-%m').tolist() == ['2026-06','2026-07']
    assert frame.loc[frame.obs_ts < pd.Timestamp('2026-06-01'),'value'].isna().all()
    # The one-day response is not proof these observations persist forever.
    assert store.snapshot_long(asof_ts='2026-09-07',series_ids=[sid]).empty
