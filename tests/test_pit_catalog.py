"""Every migrated family must obey the same origin contract, not just one adapter."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_pit_contract import ROOT, FIXTURES, store, real_store, ingest, obs, provider_for
from fev_macro.pit import PITError, content_hash
from fev_macro.pit_benchmark import MODELS, run_pit_backtest, provenance_ids
from fev_macro.pit_models import CATALOG_MODELS, ModelData, forecast_model, standardize, monthly_inputs
from fev_macro.pit_checkpoint import validate_checkpoint

OPTIONAL = {
    'auto_arima': 'statsforecast', 'auto_ets': 'statsforecast', 'theta': 'statsforecast',
    'local_trend_ssm': 'statsmodels', 'mixed_freq_dfm_md': 'statsmodels',
    'random_forest': 'sklearn', 'xgboost': 'xgboost', 'factor_pca_qd': 'sklearn',
    'lstm_univariate': 'torch', 'lstm_multivariate': 'torch',
}


def test_every_legacy_registry_entry_is_explicitly_accounted_for():
    tree = ast.parse((ROOT / 'src/fev_macro/models/__init__.py').read_text())
    registry = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign) and n.target.id == 'MODEL_REGISTRY')
    assert {key.value for key in registry.keys}.issubset(CATALOG_MODELS)
    assert len(MODELS) == len(set(MODELS))


@pytest.mark.parametrize('model', [m for m in MODELS if m not in {'chronos2', 'ensemble_avg_top3', 'ensemble_weighted_top5', 'mixed_freq_dfm_md'}])
def test_each_family_invariant_to_unavailable_revisions(real_store, model, monkeypatch):
    if model in OPTIONAL:
        pytest.importorskip(OPTIONAL[model])
    # No model is allowed to reach out to old monthly/quarterly panel files.
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: pytest.fail('Hidden panel access'))
    provider = provider_for(real_store, covariate_mode='processed')
    origins = pd.read_csv(FIXTURES / 'origins.csv').iloc[:1]
    try:
        before, audit = run_pit_backtest(provider, origins, models=[model], covariates=['UNRATE'])
        ingest(real_store, 'GDPC1', [obs('2018-10-01', '2021-01-01', 1e8), obs('2019-01-01', '2021-01-01', 1e9)])
        ingest(real_store, 'UNRATE', [obs('2019-01-01', '2021-01-01', 1e7), obs('2019-04-01', '2021-01-01', 1e6)])
        after, audit2 = run_pit_backtest(provider, origins, models=[model], covariates=['UNRATE'])
        pd.testing.assert_frame_equal(before, after)
        assert audit == audit2
        assert before.status.tolist() == ['ok']
        assert before.pit_validated.all()
        assert all(r['vintage_date'] < '2019-04-25' for r in audit[0]['inputs'])
    finally:
        provider.close()


def test_preprocessing_does_not_fit_on_forecast_rows():
    train = np.array([[1., np.nan, np.nan], [3., 8., np.nan]])
    a, future, details = standardize(train, np.array([[1e8, -1e9, 1e10]]))
    _, _, changed = standardize(train, np.array([[-1e30, 1e30, -1e30]]))
    assert details == changed
    assert details['imputation_mean'] == [2., 8., 0.]
    assert np.isfinite(a).all() and np.isfinite(future).all()


@pytest.mark.parametrize('model', ['random_forest', 'xgboost', 'factor_pca_qd', 'lstm_multivariate'])
def test_ragged_forecast_features_do_not_change_training_scaling(real_store, model):
    pytest.importorskip(OPTIONAL[model])
    provider = provider_for(real_store, covariate_mode='processed')
    origins = pd.DataFrame([dict(origin_date='2019-02-15', target_quarter='2019Q1')])
    try:
        _, a = run_pit_backtest(provider, origins, models=[model], covariates=['UNRATE'])
        # An extra current-quarter observation is known, but cannot fit preprocessing.
        ingest(real_store, 'UNRATE', [obs('2019-02-01', '2019-02-14', 1e4)])
        _, b = run_pit_backtest(provider, origins, models=[model], covariates=['UNRATE'])
        assert a[0]['model_fits'][model]['preprocessing'] == b[0]['model_fits'][model]['preprocessing']
        if model == 'factor_pca_qd':
            assert a[0]['model_fits'][model]['pca_loadings'] == b[0]['model_fits'][model]['pca_loadings']
    finally:
        provider.close()


@pytest.mark.parametrize('model', ['random_normal', 'random_forest', 'lstm_univariate'])
def test_seed_is_independent_of_requested_model_order(real_store, model):
    if model in OPTIONAL:
        pytest.importorskip(OPTIONAL[model])
    provider = provider_for(real_store)
    origins = pd.read_csv(FIXTURES / 'origins.csv').iloc[:1]
    try:
        alone, _ = run_pit_backtest(provider, origins, models=[model], covariates=['UNRATE'], seed=7)
        group, _ = run_pit_backtest(provider, origins, models=['naive_last', model], covariates=['UNRATE'], seed=7)
        assert alone.g_hat_saar.iloc[0] == group.loc[group.model == model, 'g_hat_saar'].iloc[0]
    finally:
        provider.close()


def test_no_model_substitution_and_every_failure_has_a_row(real_store):
    provider = provider_for(real_store)
    origins = pd.read_csv(FIXTURES / 'origins.csv').iloc[:1]
    try:
        with pytest.raises(PITError, match='requires.*checkpoint'):
            run_pit_backtest(provider, origins, models=['chronos2'])
        f, a = run_pit_backtest(provider, origins, models=['naive_last', 'chronos2', 'mixed_freq_dfm_md'], on_model_error='record')
        assert f.model.tolist() == ['naive_last', 'chronos2', 'mixed_freq_dfm_md']
        assert f.status.tolist() == ['ok', 'unsupported', 'unsupported']
        assert f.g_hat_saar.iloc[1:].isna().all()
        assert not f.pit_validated.iloc[1:].any()
        assert len(a[0]['model_errors']) == 2
        json.dumps(a, allow_nan=False)
    finally:
        provider.close()


def test_bvar_does_not_treat_partial_mean_as_complete_quarter(real_store):
    provider = provider_for(real_store)
    origin = pd.DataFrame([dict(origin_date='2019-02-15', target_quarter='2019Q2')])
    try:
        before, audit = run_pit_backtest(provider, origin, models=['bvar_minnesota_growth_8'], covariates=['UNRATE'])
        # Only January is available, so its partial-quarter value cannot be a full VAR state.
        ingest(real_store, 'UNRATE', [obs('2019-01-01', '2019-02-14', 1e8)])
        after, _ = run_pit_backtest(provider, origin, models=['bvar_minnesota_growth_8'], covariates=['UNRATE'])
        assert before.g_hat_saar.iloc[0] == after.g_hat_saar.iloc[0]
        assert audit[0]['model_fits']['bvar_minnesota_growth_8']['dimension'] == 2
    finally:
        provider.close()


def test_ensemble_replays_inner_origins_and_cannot_see_later_revisions(real_store):
    provider = provider_for(real_store)
    origin = pd.read_csv(FIXTURES / 'origins.csv').iloc[:1]
    candidates = ['naive_last', 'last_growth', 'mean_growth', 'ar4', 'drift']
    kwargs = dict(models=['ensemble_avg_top3', 'ensemble_weighted_top5'], covariates=['UNRATE'],
                  ensemble_windows=4, ensemble_candidates=candidates)
    try:
        before, audit = run_pit_backtest(provider, origin, **kwargs)
        selection = audit[0]['model_fits']['ensemble_avg_top3']['selection']
        assert len(selection['validation_audits']) == 4
        for inner in selection['validation_audits']:
            assert inner['origin_date'] < audit[0]['origin_date']
            assert all(r['vintage_date'] <= inner['information_cutoff'] for r in inner['inputs'])
        assert provenance_ids(selection).issubset(provenance_ids(audit))
        weights = audit[0]['model_fits']['ensemble_weighted_top5']['weights']
        assert len(weights) == 5 and sum(weights) == pytest.approx(1)
        ingest(real_store, 'GDPC1', [obs('2018-10-01', '2021-01-01', 1e9)])
        after, audit2 = run_pit_backtest(provider, origin, **kwargs)
        pd.testing.assert_frame_equal(before, after)
        assert audit == audit2
    finally:
        provider.close()


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / 'checkpoint'
    directory.mkdir()
    files = {}
    for name, body in [('config.json', b'{}'), ('model.safetensors', b'test-only-no-inference')]:
        (directory / name).write_bytes(body)
        sha = hashlib.sha256(body).hexdigest()
        blob = hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        files[name] = dict(sha256=sha, publisher_lfs_sha256=sha, publisher_git_blob=blob)
    revision = 'a'*40
    record = dict(repository='amazon/chronos-2', revision=revision, directory='checkpoint', files=files,
                  publication_evidence=dict(commit_id=revision, created_at='2025-10-01T12:00:00Z',
                                            source_url=f'https://huggingface.co/amazon/chronos-2/commit/{revision}'))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(record))
    return manifest


def test_checkpoint_publication_and_integrity_gates(checkpoint):
    for origin in ['2019-04-25', '2025-10-01']:
        with pytest.raises(PITError, match='not available'):
            validate_checkpoint(checkpoint, origin)
    directory, evidence = validate_checkpoint(checkpoint, '2025-10-02')
    assert evidence['manifest_sha256'] == content_hash(json.loads(checkpoint.read_text()))
    (directory / 'model.safetensors').write_bytes(b'tampered')
    with pytest.raises(PITError, match='hash mismatch'):
        validate_checkpoint(checkpoint, '2026-09-04')


def test_forecast_only_cli_has_no_truth_or_scores_and_reports_failures(tmp_path):
    out = tmp_path / 'out'
    command = [sys.executable, str(ROOT / 'scripts/run_pit_backtest.py'), '--db', str(tmp_path/'data.duckdb'),
               '--fixture-dir', str(FIXTURES), '--origins', str(FIXTURES/'origins.csv'), '--forecast-only',
               '--models', 'naive_last', 'chronos2', '--on-model-error', 'record', '--out', str(out)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (out/'truth.csv').exists() and not (out/'metrics.csv').exists()
    assert len(pd.read_csv(out/'failures.csv')) == 4
    assert len(pd.read_csv(out/'forecasts.csv')) == 8
    assert json.loads((out/'manifest.json').read_text())['config']['forecast_only']
    repeat = subprocess.run(command, capture_output=True, text=True)
    assert repeat.returncode != 0 and 'new output directory' in repeat.stderr


def test_nested_selection_does_not_refit_past_origins_on_outer_vintage(real_store):
    from fev_macro.pit_benchmark import _ensemble_validation
    p = provider_for(real_store)
    def validate():
        panel, _ = p.build_panel_asof(asof_ts='2019-04-25', target_col='GDPC1', covariate_columns=[])
        y = panel.set_index('quarter').GDPC1.dropna()
        return _ensemble_validation(p, pd.Timestamp('2019-04-25'), pd.Period('2019Q1'), y,
             candidates=['naive_last', 'last_growth', 'mean_growth', 'ar4', 'drift'], windows=4,
             covariates=[], min_train=24, rolling_size=None, seed=0)
    try:
        a = validate()
        # This revision is known for current model selection, but was not known at any inner origin.
        ingest(real_store, 'GDPC1', [obs('2018-04-01', '2019-04-20', 25000.)])
        b = validate()
        assert a['validation_audits'] == b['validation_audits']
        assert a['rmse_saar'] != b['rmse_saar']
    finally:
        p.close()


def test_real_mixed_frequency_filter_uses_only_released_months(store):
    pytest.importorskip('statsmodels')
    rng = np.random.default_rng(84)
    months = pd.period_range('2000-01', '2015-03', freq='M')
    f = np.zeros(len(months))
    for i in range(1, len(f)):
        f[i] = .65*f[i-1] + rng.normal(scale=.4)
    series = [f + rng.normal(scale=.3, size=len(f)), -.7*f + rng.normal(scale=.3, size=len(f)),
              .4*f + rng.normal(scale=.4, size=len(f))]
    for sid, values in zip(['X', 'Z', 'W'], series):
        ingest(store, sid, [obs(str(m.start_time.date()), str((m+1).start_time.date()), value)
                            for m, value in zip(months, values)])
    quarters = pd.period_range('2000Q1', '2014Q4', freq='Q')
    logy = [np.log(100.)]
    for q in quarters[1:]:
        pos = months.get_loc(q.asfreq('M', 'end'))
        g = .006 + .002 * (f[pos-4:pos+1] @ np.array([1.,2.,3.,2.,1.])) + rng.normal(scale=.001)
        logy.append(logy[-1]+g)
    ingest(store, 'GDPC1', [obs(str(q.start_time.date()), str((q.end_time.normalize()+pd.Timedelta(days=25)).date()), np.exp(v))
                           for q, v in zip(quarters, logy)])
    from fev_macro.asof_provider import AsofVintageProvider
    p = AsofVintageProvider(db_path=store.db_path, covariate_mode='processed', series_specs={
        'GDPC1': {'frequency':'Q'}, **{c:{'frequency':'M','tcode':1} for c in ['X','Z','W']}})
    origins = pd.DataFrame([dict(origin_date='2015-03-15', target_quarter='2015Q1')])
    try:
        before, a = run_pit_backtest(p, origins, models=['mixed_freq_dfm_md'], covariates=['X','Z','W'])
        detail = a[0]['model_fits']['mixed_freq_dfm_md']
        assert detail['training_months'][1] == '2014-12'
        assert a[0]['quarterly_observation_counts']['X']['2015Q1'] == 2
        ingest(store, 'X', [obs('2015-03-01', '2015-04-02', 1e9)])
        after, _ = run_pit_backtest(p, origins, models=['mixed_freq_dfm_md'], covariates=['X','Z','W'])
        pd.testing.assert_frame_equal(before, after)
        # Known current-quarter news changes filtering, never fitted parameters/scaling.
        ingest(store, 'X', [obs('2015-02-01', '2015-03-14', 2.)])
        news, b = run_pit_backtest(p, origins, models=['mixed_freq_dfm_md'], covariates=['X','Z','W'])
        assert detail['fit_params'] == b[0]['model_fits']['mixed_freq_dfm_md']['fit_params']
        assert abs(news.g_hat_saar.iloc[0] - before.g_hat_saar.iloc[0]) > 1e-5
    finally:
        p.close()


def test_nonconvergence_is_not_relabelled_as_an_ssm_forecast(real_store, monkeypatch):
    structural = pytest.importorskip('statsmodels.tsa.statespace.structural')
    from types import SimpleNamespace
    monkeypatch.setattr(structural.UnobservedComponents, 'fit', lambda *a, **k: SimpleNamespace(mle_retvals={'converged':False}))
    p = provider_for(real_store)
    try:
        f, _ = run_pit_backtest(p, pd.read_csv(FIXTURES/'origins.csv').iloc[:1], models=['local_trend_ssm'], on_model_error='record')
        assert f.status.iloc[0] == 'failed'
        assert np.isnan(f.g_hat_saar.iloc[0])
        assert not f.pit_validated.iloc[0]
    finally:
        p.close()


def test_random_forest_calls_tree_estimator(real_store, monkeypatch):
    ensemble = pytest.importorskip('sklearn.ensemble')
    actual_fit = ensemble.RandomForestRegressor.fit
    seen = []
    def observe(self, x, y, **kwargs):
        seen.append((x.shape, self.random_state))
        return actual_fit(self, x, y, **kwargs)
    monkeypatch.setattr(ensemble.RandomForestRegressor, 'fit', observe)
    p = provider_for(real_store)
    try:
        run_pit_backtest(p, pd.read_csv(FIXTURES/'origins.csv').iloc[:1], models=['random_forest'], covariates=['UNRATE'])
        assert len(seen) == 1 and seen[0][0][1] == 11  # 8 lags + value/count/missing
    finally:
        p.close()


@pytest.mark.parametrize('model', ['auto_arima', 'random_forest', 'bvar_minnesota_8', 'lstm_multivariate'])
def test_multistep_saar_matches_final_step_not_cumulative_growth(real_store, model):
    if model in OPTIONAL:
        pytest.importorskip(OPTIONAL[model])
    p = provider_for(real_store, covariate_mode='processed')
    try:
        f, a = run_pit_backtest(p, pd.DataFrame([dict(origin_date='2019-02-15', target_quarter='2019Q1')]),
                                models=[model], covariates=['UNRATE'], rolling_size=32)
        path = a[0]['model_fits'][model]['growth_path']
        assert len(path) == f.horizon.iloc[0] == 2
        assert f.n_train.iloc[0] == 32
        assert f.g_hat_saar.iloc[0] == pytest.approx(100*np.expm1(4*path[-1]))
        last = [r['value'] for r in a[0]['inputs'] if r['variable']=='GDPC1' and r['obs_date']=='2018-07-01'][0]
        assert f.y_hat_level.iloc[0] == pytest.approx(last*np.exp(sum(path)))
    finally:
        p.close()


def test_ensemble_calendar_preserves_day_across_unequal_quarters(real_store):
    from fev_macro.pit_benchmark import _ensemble_validation
    p = provider_for(real_store)
    try:
        panel, _ = p.build_panel_asof(asof_ts='2019-10-25', target_col='GDPC1', covariate_columns=[])
        selection = _ensemble_validation(p, pd.Timestamp('2019-10-25'), pd.Period('2019Q3'),
            panel.set_index('quarter').GDPC1.dropna(),
            candidates=['naive_last','last_growth','mean_growth','ar4','drift'], windows=4,
            covariates=[], min_train=24, rolling_size=None, seed=0)
        inner = {a['target_quarter']:a['origin_date'] for a in selection['validation_audits']}
        assert inner['2019Q1'] == '2019-04-25T00:00:00'  # not April 27, after the advance release
    finally:
        p.close()


def test_univariate_provenance_does_not_claim_unused_feature_vintage(real_store):
    ingest(real_store, 'UNRATE', [obs('2019-01-01', '2019-04-24', 4.)])
    p = provider_for(real_store)
    try:
        f, a = run_pit_backtest(p, pd.read_csv(FIXTURES/'origins.csv').iloc[:1], models=['ar4'], covariates=['UNRATE'])
        assert f.max_vintage_inspected.iloc[0] == '2019-04-24'
        gdp = [r['vintage_date'] for r in a[0]['inputs'] if r['variable']=='GDPC1']
        assert f.max_vintage_used.iloc[0] == max(gdp) < '2019-04-24'
        assert a[0]['model_input_columns']['ar4'] == ['GDPC1']
    finally:
        p.close()


def test_random_seed_uses_information_date_not_intraday_timestamp(real_store):
    p = provider_for(real_store)
    try:
        rows = pd.DataFrame([dict(origin_date=o, target_quarter='2019Q1') for o in
                             ['2019-04-25', '2019-04-25T18:00:00', '2019-04-26T02:00:00Z']])
        f, _ = run_pit_backtest(p, rows, models=['random_normal'])
        assert f.information_cutoff.nunique() == 1
        assert f.g_hat_saar.nunique() == 1
    finally:
        p.close()
