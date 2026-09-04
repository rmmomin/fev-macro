from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fev_macro.asof_provider import AsofVintageProvider
from fev_macro.asof_store import AsofStore
from fev_macro.pit import PITError, alfred_rows, information_date
from fev_macro.pit_benchmark import (build_release_truth, growth, paired_metrics,
                                     paired_dm, run_pit_backtest, score_forecasts, validate_release_calendar)
import sync_alfred_asof_store as sync

FIXTURES = ROOT / "tests/fixtures/alfred"


def ingest(store, sid, observations):
    params = dict(series_id=sid, realtime_start="1776-07-04", realtime_end="9999-12-31",
                  output_type=1, units="lin", offset=0, limit=100000)
    payload = dict(realtime_start=params["realtime_start"], realtime_end=params["realtime_end"],
                   output_type=1, units="lin", count=len(observations), offset=0, observations=observations)
    return store.ingest_alfred_response(payload, params)


def obs(d, a, value, b="9999-12-31"):
    return dict(date=d, realtime_start=a, realtime_end=b, value="." if value is None else str(value))


@pytest.fixture
def store(tmp_path):
    s = AsofStore(tmp_path / "pit.duckdb")
    yield s
    s.close()


@pytest.fixture
def real_store(store):
    for name in ("gdpc1_intervals", "unrate_intervals", "gdp_intervals", "cpiaucsl_intervals"):
        fixture = json.loads((FIXTURES / (name + ".json")).read_text())
        store.ingest_alfred_response(fixture["response"], fixture["params"], retrieved_at=fixture["retrieved_at"])
    return store


def provider_for(store, **kwargs):
    return AsofVintageProvider(db_path=store.db_path, series_specs={
        "GDPC1": {"frequency": "Q"}, "UNRATE": {"frequency": "M", "tcode": 1}}, **kwargs)


@pytest.mark.parametrize("origin,cutoff", [
    ("2019-04-26", "2019-04-25"), ("2019-04-26 23:59", "2019-04-25"),
    ("2019-04-27T02:00:00Z", "2019-04-25"), ("2019-04-27T04:00:00Z", "2019-04-26")])
def test_new_york_prior_day_contract(origin, cutoff):
    assert information_date(origin) == pd.Timestamp(cutoff)


def test_missing_revision_does_not_resurrect_value(store):
    ingest(store, "X", [obs("2019-01-01", "2019-02-01", 1), obs("2019-01-01", "2019-03-01", None)])
    assert store.snapshot_long(asof_ts="2019-03-01", series_ids=["X"]).value.iloc[0] == 1
    assert pd.isna(store.snapshot_long(asof_ts="2019-03-02", series_ids=["X"]).value.iloc[0])


def test_realtime_end_is_inclusive_and_no_resurrection_after_gap(store):
    ingest(store, "X", [obs("2019-01-01", "2019-02-01", 1),
                         obs("2019-01-01", "2019-03-01", 2, "2019-03-02")])
    assert store.snapshot_long(asof_ts="2019-03-03", series_ids=["X"]).value.iloc[0] == 2
    assert store.snapshot_long(asof_ts="2019-03-04", series_ids=["X"]).empty


@pytest.mark.parametrize("source", [None, "fred_observations_fallback", "alfred_output_type_1", "unit_test"])
def test_old_or_unproven_records_refused(store, source):
    store.ingest_versions(pd.DataFrame([dict(series_id="X", obs_ts="2019-01-01", asof_ts="2019-01-01", value=1)]), source=source)
    with pytest.raises(PITError, match="Unverified/legacy"):
        store.snapshot_long(asof_ts="2020-01-01", series_ids=["X"])


def test_duplicate_ingestion_counts_and_conflict(store):
    row = obs("2019-01-01", "2019-02-01", 1)
    assert ingest(store, "X", [row]) == 1
    assert ingest(store, "X", [row]) == 0
    with pytest.raises(PITError, match="Conflicting"):
        ingest(store, "X", [obs("2019-01-01", "2019-02-01", 99)])


def test_future_observation_date_excluded_even_if_source_reports_it(store):
    ingest(store, "X", [obs("2025-01-01", "2019-02-01", 999)])
    assert store.snapshot_long(asof_ts="2019-03-01", series_ids=["X"]).empty


@pytest.mark.parametrize("name,sid,date,origin,value", [
    ("GDP", "GDP", "2019-01-01", "2019-04-27", 21062.691),
    ("UNRATE", "UNRATE", "2019-01-01", "2019-02-02", 4.0),
    ("CPI", "CPIAUCSL", "2019-01-01", "2019-02-14", 252.673),
    ("GDPC1", "GDPC1", "2019-01-01", "2019-04-27", 18912.326)])
def test_actual_api_release_fixtures(real_store, name, sid, date, origin, value):
    frame = real_store.snapshot_long(asof_ts=origin, series_ids=[sid], obs_start=date, obs_end=date)
    assert frame.value.iloc[0] == pytest.approx(value)
    assert real_store.snapshot_long(asof_ts=pd.Timestamp(origin)-pd.Timedelta(days=1), series_ids=[sid],
                                    obs_start=date, obs_end=date).empty


def test_actual_snapshot_bounds_are_clipped_not_initial_release():
    f = json.loads((FIXTURES / "gdpc1_on_advance.json").read_text())
    rows = alfred_rows(f["response"], f["params"])
    assert rows.asof_ts.eq("2019-04-26").all()
    # q-1 was already published in February; the one-day query clips its interval.
    assert rows.realtime_end.eq("2019-04-26").all()
    before = json.loads((FIXTURES / "gdpc1_before_advance.json").read_text())
    assert before["response"]["observations"] == []


def test_actual_output3_and_vintage_date_semantics():
    dates = json.loads((FIXTURES / "gdpc1_vintage_dates.json").read_text())["response"]
    assert dates["vintage_dates"] == ["2019-04-26", "2019-05-30", "2019-06-27"]
    changes = json.loads((FIXTURES / "gdpc1_output3_one_vintage.json").read_text())["response"]
    assert changes["observations"] == [{"date": "2019-01-01", "GDPC1_20190530": "18907.517"}]
    # The unchanged 2018Q4 level is absent from the change response.


def test_native_transform_before_partial_quarter_aggregation(store):
    ingest(store, "GDPC1", [obs("2018-10-01", "2019-02-28", 100)])
    ingest(store, "UNRATE", [obs("2018-12-01", "2019-01-05", 1), obs("2019-01-01", "2019-02-01", 2),
                              obs("2019-02-01", "2019-03-01", 4), obs("2019-03-01", "2019-04-01", 100)])
    p = provider_for(store, covariate_mode="processed")
    p.series_specs["UNRATE"]["tcode"] = 2
    try:
        panel, meta = p.build_panel_asof(asof_ts="2019-03-15", target_col="GDPC1", covariate_columns=["UNRATE"])
        q1 = panel.loc[panel.quarter == pd.Period("2019Q1")].iloc[0]
        assert q1.UNRATE == 1.5  # mean of monthly differences (2-1, 4-2)
        assert pd.isna(q1.GDPC1)
        assert meta["quarterly_observation_counts"]["UNRATE"]["2019Q1"] == 2
        assert max(r["vintage_date"] for r in meta["inputs"]) <= "2019-03-14"
    finally:
        p.close()


def test_adapter_never_fills_pit_holes_from_latest_panel(real_store):
    p = provider_for(real_store)
    try:
        base = pd.DataFrame(dict(quarter=pd.period_range("2018Q4", "2019Q1", freq="Q"),
                                 GDPC1=[999999., 999999.], UNRATE=[999., 999.], UNKNOWN=[999., 999.]))
        result, _ = p.adapt_train_df(train_df=base, asof_ts="2019-04-25",
                                    cutoff_quarter=pd.Period("2019Q1"), target_col="GDPC1")
        assert result.GDPC1.dropna().max() < 20000
        assert result.loc[result.quarter == pd.Period("2019Q1"), "GDPC1"].isna().all()
        assert result.UNKNOWN.isna().all()
        with pytest.raises(PITError):
            p.adapt_train_df(train_df=base, asof_ts="1900-01-01", cutoff_quarter=pd.Period("2019Q1"), target_col="GDPC1")
    finally:
        p.close()


def test_gdp_release_truth_matches_all_twelve_bea_releases(real_store):
    calendar = pd.read_csv(FIXTURES / "release_calendar.csv")
    truth = build_release_truth(real_store, calendar)
    assert len(truth) == 12
    assert np.allclose(truth.g_true_saar, calendar.expected_saar, atol=.06)
    assert np.allclose(100 * ((1 + truth.g_true_qoq / 100)**4 - 1), truth.g_true_saar)


def test_unchanged_estimate_is_still_a_release_and_denominator_same_vintage(store):
    ingest(store, "GDPC1", [obs("2018-10-01", "2019-02-01", 100),
                            obs("2018-10-01", "2019-05-30", 200),
                            obs("2019-01-01", "2019-04-26", 101),
                            obs("2019-01-01", "2019-05-30", 202)])
    calendar = pd.read_csv(FIXTURES / "release_calendar.csv").iloc[:3].drop(columns="expected_saar")
    truth = build_release_truth(store, calendar)
    assert np.allclose(truth.g_true_saar, growth(101, 100))
    assert truth.truth_previous_level.tolist() == [100, 200, 200]
    assert truth.release_date.tolist() == ["2019-04-26", "2019-05-30", "2019-06-27"]


def test_release_calendar_duplicates_and_wrong_order_rejected():
    c = pd.read_csv(FIXTURES / "release_calendar.csv").iloc[:3].copy()
    with pytest.raises(PITError):
        validate_release_calendar(pd.concat([c, c.iloc[:1]]))
    c.loc[0, "release_date"] = "2019-07-01"
    with pytest.raises(PITError):
        validate_release_calendar(c)


def test_forecasts_invariant_to_future_revisions_and_targets(real_store):
    p = provider_for(real_store)
    origins = pd.read_csv(FIXTURES / "origins.csv").iloc[:1]
    try:
        before, audits = run_pit_backtest(p, origins, models=["ar4", "bridge_ridge"], covariates=["UNRATE"])
        ingest(real_store, "GDPC1", [obs("2018-10-01", "2021-01-01", 1e8), obs("2019-01-01", "2021-01-01", 1e9)])
        ingest(real_store, "UNRATE", [obs("2019-01-01", "2021-01-01", 1e6)])
        after, audits2 = run_pit_backtest(p, origins, models=["ar4", "bridge_ridge"], covariates=["UNRATE"])
        pd.testing.assert_frame_equal(before, after)
        assert audits == audits2
        assert before.pit_validated.all()
    finally:
        p.close()


def test_rolling_counts_observed_targets_and_horizon_uses_last_available(real_store):
    p = provider_for(real_store)
    try:
        origins = pd.DataFrame([dict(origin_date="2019-02-15", target_quarter="2019Q1")])
        forecasts, audit = run_pit_backtest(p, origins, rolling_size=24, covariates=["UNRATE"])
        assert forecasts.n_train.eq(24).all()
        assert forecasts.horizon.eq(2).all()  # 2018Q4 GDP was delayed until Feb 28.
        assert forecasts.training_max_quarter.eq("2018Q3").all()
        assert audit[0]["training_levels"] == 24
        with pytest.raises(PITError, match="already available"):
            run_pit_backtest(p, pd.DataFrame([dict(origin_date="2019-04-27", target_quarter="2019Q1")]))
    finally:
        p.close()


def test_internal_target_gap_fails_closed(real_store):
    ingest(real_store, "GDPC1", [obs("2018-01-01", "2019-01-01", None)])
    p = provider_for(real_store)
    try:
        with pytest.raises(PITError, match="Internal GDP gap"):
            run_pit_backtest(p, pd.read_csv(FIXTURES / "origins.csv").iloc[:1])
    finally:
        p.close()


def test_metrics_use_matched_finite_benchmark_rows():
    rows = []
    for model in ["naive_last", "test"]:
        for i in range(3):
            rows.append(dict(model=model, origin_date=str(i), target_quarter=str(i), horizon=1,
                             release_stage="first", g_true_saar=0,
                             g_hat_saar=([1, 1, 100][i] if model == "naive_last" else [2, 2, np.nan][i])))
    result = paired_metrics(pd.DataFrame(rows)).set_index("model")
    assert result.loc["test", "n"] == 2
    assert result.loc["test", "rel_rmse"] == 2
    with pytest.raises(PITError, match="Duplicate"):
        paired_metrics(pd.DataFrame(rows + rows[:1]))


def sync_kwargs(store):
    return dict(store=store, series_id="X", api_key="not-a-real-secret", rate_limiter=sync.RateLimiter(0),
                stats=sync.APIStats(), args=SimpleNamespace(page_limit=1, observation_start=None, observation_end=None,
                backfill_realtime_start="1776-07-04", backfill_realtime_end="9999-12-31", lookback_days=7))


def test_sync_http_failure_never_calls_current_fred_fallback(store, monkeypatch):
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise sync.FredAPIError("HTTP 400")
    monkeypatch.setattr(sync, "fred_series_observations", fail)
    with pytest.raises(sync.FredAPIError):
        sync.backfill_series_output_type_1(**sync_kwargs(store))
    assert len(calls) == 1
    assert store.available_series_ids() == set()


def test_sync_pagination_atomic_rollback_and_replay(store, monkeypatch):
    def response(**kwargs):
        offset = kwargs["params"]["offset"]
        if offset == 1:
            raise sync.FredAPIError("HTTP 500")
        return dict(output_type=1, units="lin", realtime_start="1776-07-04", realtime_end="9999-12-31",
                    offset=offset, count=2, observations=[obs("2019-01-01", "2019-02-01", 1)])
    monkeypatch.setattr(sync, "fred_series_observations", response)
    with pytest.raises(sync.FredAPIError):
        sync.backfill_series_output_type_1(**sync_kwargs(store))
    assert store.available_series_ids() == set()
    assert store._con.execute("SELECT count(*) FROM asof_api_responses").fetchone()[0] == 0
    def complete(**kwargs):
        p = kwargs["params"]
        return dict(output_type=1, units="lin", realtime_start=p["realtime_start"], realtime_end=p["realtime_end"],
                    offset=p["offset"], count=2, observations=[obs("2019-01-01", "2019-02-01", 1)
                     if p["offset"] == 0 else obs("2019-02-01", "2019-03-01", None)])
    monkeypatch.setattr(sync, "fred_series_observations", complete)
    sync.backfill_series_output_type_1(**sync_kwargs(store))
    assert len(store.snapshot_long(asof_ts="2019-04-01", series_ids=["X"])) == 2


def test_incremental_replay_does_not_skip_the_checkpoint_day(store, monkeypatch):
    ingest(store, "X", [obs("2019-01-01", "2019-02-01", 1)])
    seen = []
    def response(**kwargs):
        p = kwargs["params"]
        seen.append(p)
        return dict(output_type=1, units="lin", realtime_start=p["realtime_start"], realtime_end=p["realtime_end"],
                    offset=0, count=1, observations=[obs("2019-02-01", "2019-02-01", 2)])
    monkeypatch.setattr(sync, "fred_series_observations", response)
    assert sync.update_series_intervals(**sync_kwargs(store)) == 1
    assert seen[0]["realtime_start"] == "2019-01-25"
    assert len(store.snapshot_long(asof_ts="2019-02-02", series_ids=["X"])) == 2


@pytest.mark.parametrize("change", [dict(units="pch"), dict(output_type=3), dict(realtime_start=None)])
def test_questionable_response_configuration_is_rejected(change):
    fixture = json.loads((FIXTURES / "gdpc1_on_advance.json").read_text())
    fixture["params"].update(change)
    with pytest.raises(PITError):
        alfred_rows(fixture["response"], fixture["params"])


def test_http_errors_do_not_expose_api_key(monkeypatch):
    import urllib.error
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.com", 400, "Bad request", {}, None)
    monkeypatch.setattr(sync.urllib.request, "urlopen", fail)
    with pytest.raises(sync.FredAPIError) as err:
        sync._request_json(url=sync.FRED_BASE+"/series/observations", params={"api_key": "secret"},
                           timeout_seconds=1, max_retries=0, retry_backoff_seconds=0,
                           rate_limiter=sync.RateLimiter(0), stats=sync.APIStats())
    assert "secret" not in str(err.value)


def test_legacy_entrypoints_refuse_pit_claim_by_default():
    pytest.importorskip("fev", reason="Legacy guard test needs the optional full research stack")
    from fev_macro.realtime_oos import run_backtest
    from build_gdp_releases import build_release_dataset
    with pytest.raises(ValueError, match="Legacy run_backtest"):
        run_backtest([], pd.DataFrame(), pd.DataFrame())
    with pytest.raises(ValueError, match="release calendar"):
        build_release_dataset(pd.DataFrame(), "GDPC1")


def test_dm_rejects_small_and_repeated_target_samples():
    data = []
    for i, q in enumerate(pd.period_range("2000Q1", periods=24, freq="Q")):
        for model in ("m", "b"):
            data.append(dict(model=model, origin_date=str(q.start_time), target_quarter=str(q), horizon=2,
                             release_stage="first", g_true_saar=0,
                             g_hat_saar=1 + (i % 5) / 10 if model == "m" else 1))
    frame = pd.DataFrame(data)
    result = paired_dm(frame, "m", "b", horizon=2, stage="first")
    reverse = paired_dm(frame, "b", "m", horizon=2, stage="first")
    assert result["statistic"] == pytest.approx(-reverse["statistic"])
    assert result["p_value"] == reverse["p_value"]
    assert result["hac_lags"] >= 1
    with pytest.raises(PITError, match="at least 20"):
        paired_dm(frame.iloc[:8], "m", "b", horizon=2, stage="first")
    repeated = frame.copy()
    repeated.loc[repeated.target_quarter == "2000Q2", "target_quarter"] = "2000Q1"
    with pytest.raises(PITError, match="unique consecutive"):
        paired_dm(repeated, "m", "b", horizon=2, stage="first")


def test_strict_cli_roundtrip(tmp_path):
    import subprocess
    out = tmp_path / "results"
    command = [sys.executable, str(ROOT / "scripts/run_pit_backtest.py"), "--db", str(tmp_path / "fixture.duckdb"),
               "--fixture-dir", str(FIXTURES), "--origins", str(FIXTURES / "origins.csv"),
               "--release-calendar", str(FIXTURES / "release_calendar.csv"),
               "--series-specs", str(FIXTURES / "series_specs.json"), "--out", str(out)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    forecasts = pd.read_csv(out / "forecasts.csv")
    assert len(forecasts) == 12
    audits = json.loads((out / "audit.json").read_text())
    assert set(forecasts.audit_id) == {a["audit_id"] for a in audits}
    assert len(pd.read_csv(out / "scored.csv")) == 36
    assert (out / "manifest.json").exists()


def test_target_alias_cannot_be_a_feature(real_store):
    p = provider_for(real_store)
    p.series_specs["leaked_target"] = {"series_id": "GDPC1", "frequency": "Q"}
    try:
        with pytest.raises(PITError, match="alias the GDP target"):
            run_pit_backtest(p, pd.read_csv(FIXTURES / "origins.csv"), covariates=["leaked_target"])
    finally:
        p.close()


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("FEV_LIVE_ALFRED") != "1", reason="Live ALFRED is explicitly opt-in")
def test_live_alfred_matches_frozen_historical_response():
    fixture = json.loads((FIXTURES / "gdpc1_on_advance.json").read_text())
    args = SimpleNamespace(timeout_seconds=30, max_retries=2, retry_backoff_seconds=1,
                           api_key=None, env_file=str(ROOT / ".env"))
    payload = sync.fred_series_observations(series_id="GDPC1", api_key=sync.resolve_api_key(args), args=args,
              params=fixture["params"], rate_limiter=sync.RateLimiter(.55), stats=sync.APIStats())
    assert payload["observations"] == fixture["response"]["observations"]
