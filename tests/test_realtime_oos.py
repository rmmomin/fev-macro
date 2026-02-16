from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.realtime_oos import (  # noqa: E402
    build_vintage_calendar,
    compute_vintage_asof_date,
    levels_from_saar_growth,
    run_backtest,
    saar_growth_series_from_levels,
    select_training_vintage,
    to_quarter_period,
    to_saar_growth,
)
from fev_macro.realtime_feeds import train_df_to_datasets  # noqa: E402
from fev_macro.vintage_panels import build_panel_from_files  # noqa: E402


def test_saar_conversion_toy_series() -> None:
    current = 101.0
    previous = 100.0
    expected = 100.0 * ((current / previous) ** 4 - 1.0)
    actual = to_saar_growth(current, previous)
    assert np.isclose(actual, expected, rtol=0.0, atol=1e-12)


def test_saar_growth_series_round_trip_to_levels() -> None:
    levels = pd.Series([100.0, 101.5, 103.0, 104.1], dtype=float)
    growth = saar_growth_series_from_levels(levels)
    assert np.isnan(growth.iloc[0])

    reconstructed = levels_from_saar_growth(last_level=float(levels.iloc[0]), g_hat=growth.iloc[1:].to_numpy(dtype=float))
    assert np.allclose(reconstructed, levels.iloc[1:].to_numpy(dtype=float), rtol=0.0, atol=1e-10)


def test_select_training_vintage_last_business_day_bounds() -> None:
    panel = pd.DataFrame(
        {
            "vintage": ["2024-01", "2024-02", "2024-03"],
            "asof_date": [
                compute_vintage_asof_date("2024-01"),
                compute_vintage_asof_date("2024-02"),
                compute_vintage_asof_date("2024-03"),
            ],
            "quarter": pd.PeriodIndex(["2023Q4", "2023Q4", "2023Q4"], freq="Q-DEC"),
            "GDPC1": [1.0, 1.0, 1.0],
        }
    )

    calendar = build_vintage_calendar(panel)
    origin_date = pd.Timestamp("2024-03-15")

    selected = select_training_vintage(origin_date=origin_date, vintage_calendar=calendar)

    assert selected["vintage"] == "2024-02"
    assert selected["asof_date"] <= origin_date
    assert origin_date < selected["next_asof_date"]


def _make_synthetic_release_table() -> pd.DataFrame:
    quarters = pd.period_range("2020Q1", "2023Q4", freq="Q-DEC")
    obs_dates = [q.start_time for q in quarters]
    first_release_dates = [(q.end_time + pd.Timedelta(days=35)).normalize() for q in quarters]
    first_levels = 100.0 + 1.25 * np.arange(len(quarters), dtype=float)
    second_release_dates = [d + pd.Timedelta(days=30) for d in first_release_dates]
    third_release_dates = [d + pd.Timedelta(days=60) for d in first_release_dates]
    second_levels = first_levels + 0.10
    third_levels = first_levels + 0.20

    df = pd.DataFrame(
        {
            "observation_date": obs_dates,
            "quarter": quarters,
            "first_release_date": first_release_dates,
            "first_release": first_levels,
            "second_release_date": second_release_dates,
            "second_release": second_levels,
            "third_release_date": third_release_dates,
            "third_release": third_levels,
        }
    )
    df["first_release_lag_days"] = (df["first_release_date"] - df["observation_date"]).dt.days
    df["valid_origin"] = True

    y_prev = df["first_release"].shift(1)
    df["g_true_saar_first"] = [to_saar_growth(c, p) for c, p in zip(df["first_release"], y_prev)]
    df["qoq_saar_growth_realtime_first_pct"] = df["g_true_saar_first"]
    y_prev_second = df["second_release"].shift(1)
    y_prev_third = df["third_release"].shift(1)
    df["qoq_saar_growth_realtime_second_pct"] = [to_saar_growth(c, p) for c, p in zip(df["second_release"], y_prev_second)]
    df["qoq_saar_growth_realtime_third_pct"] = [to_saar_growth(c, p) for c, p in zip(df["third_release"], y_prev_third)]
    return df


def _make_synthetic_vintage_panel(release_table: pd.DataFrame) -> pd.DataFrame:
    quarters = pd.PeriodIndex(release_table["quarter"], freq="Q-DEC")
    base_levels = release_table["first_release"].to_numpy(dtype=float)

    vintages = pd.period_range("2020-04", "2024-12", freq="M")
    rows: list[dict[str, object]] = []
    for vintage in vintages:
        vintage_label = str(vintage)
        asof = compute_vintage_asof_date(vintage)
        observed_quarter = vintage.asfreq("Q-DEC") - 1

        for q, level in zip(quarters, base_levels):
            ts = q.asfreq("M", "end").to_timestamp()
            value = float(level) if q <= observed_quarter else np.nan
            rows.append(
                {
                    "vintage": vintage_label,
                    "asof_date": asof,
                    "timestamp": ts,
                    "quarter": q,
                    "GDPC1": value,
                }
            )

    panel = pd.DataFrame(rows)
    panel["quarter"] = to_quarter_period(panel["timestamp"])
    return panel


def test_run_backtest_training_cutoff_before_target() -> None:
    release = _make_synthetic_release_table()
    panel = _make_synthetic_vintage_panel(release)

    preds = run_backtest(
        models=["naive_last"],
        release_table=release,
        vintage_panel=panel,
        horizons=[1, 2, 3],
        target_col="GDPC1",
        max_origins=8,
        min_train_observations=4,
    )

    assert not preds.empty

    for row in preds.itertuples(index=False):
        train_max = pd.Period(row.training_max_quarter, freq="Q-DEC")
        target_q = pd.Period(row.target_quarter, freq="Q-DEC")
        assert train_max < target_q


def test_run_backtest_monthly_release_stages_and_buckets() -> None:
    release = _make_synthetic_release_table()
    panel = _make_synthetic_vintage_panel(release)

    preds = run_backtest(
        models=["naive_last"],
        release_table=release,
        vintage_panel=panel,
        horizons=[1],
        target_col="GDPC1",
        origin_schedule="monthly",
        release_stages=["first", "second", "third"],
        max_origins=12,
        min_train_observations=4,
    )

    assert not preds.empty
    assert set(preds["release_stage"].unique()) == {"first", "second", "third"}
    assert preds["months_to_first_release_bucket"].notna().all()


def test_run_backtest_respects_min_target_quarter_filter() -> None:
    release = _make_synthetic_release_table()
    panel = _make_synthetic_vintage_panel(release)

    preds = run_backtest(
        models=["naive_last"],
        release_table=release,
        vintage_panel=panel,
        horizons=[1, 2, 3, 4],
        target_col="GDPC1",
        origin_schedule="quarterly",
        max_origins=None,
        min_train_observations=4,
        min_target_quarter="2022Q1",
    )

    assert not preds.empty
    min_target = min(pd.Period(q, freq="Q-DEC") for q in preds["target_quarter"].astype(str))
    assert min_target >= pd.Period("2022Q1", freq="Q-DEC")


def test_run_backtest_prefers_realtime_saar_truth_columns() -> None:
    release = _make_synthetic_release_table()
    panel = _make_synthetic_vintage_panel(release)

    # Force a distinctive truth signal that differs from level-derived growth.
    release.loc[release["quarter"] >= pd.Period("2020Q2", freq="Q-DEC"), "qoq_saar_growth_realtime_first_pct"] = 9.99

    preds = run_backtest(
        models=["naive_last"],
        release_table=release,
        vintage_panel=panel,
        horizons=[1],
        target_col="GDPC1",
        origin_schedule="quarterly",
        release_stages=["first"],
        max_origins=6,
        min_train_observations=4,
    )

    assert not preds.empty
    finite_truth = preds["g_true_saar"].dropna().to_numpy(dtype=float)
    assert finite_truth.size > 0
    assert np.allclose(finite_truth, 9.99)


def test_run_backtest_supports_growth_target_models() -> None:
    release = _make_synthetic_release_table()
    panel = _make_synthetic_vintage_panel(release)

    preds = run_backtest(
        models=["naive_last_growth"],
        release_table=release,
        vintage_panel=panel,
        horizons=[1, 2],
        target_col="GDPC1",
        origin_schedule="quarterly",
        max_origins=6,
        min_train_observations=4,
    )

    assert not preds.empty
    assert set(preds["model"].unique()) == {"naive_last_growth"}
    assert preds["y_hat_level"].notna().any()
    assert preds["g_hat_saar"].notna().any()


def test_train_df_to_datasets_builds_ragged_edge_future_covariates() -> None:
    train_df = pd.DataFrame(
        {
            "quarter": pd.PeriodIndex(["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"], freq="Q-DEC"),
            "GDPC1": [100.0, 101.0, 102.0, np.nan, np.nan],
            "UNRATE": [4.0, 4.1, 4.2, 4.3, np.nan],
        }
    )
    past_data, future_data, task = train_df_to_datasets(
        train_df=train_df,
        target_col="GDPC1",
        horizon=2,
        covariate_cols=["UNRATE"],
    )

    assert task.horizon == 2
    assert len(past_data[0]["GDPC1"]) == 3
    assert future_data[0]["UNRATE"] == [4.3, 4.3]


def test_build_panel_applies_transforms_but_can_exclude_gdpc1(tmp_path: Path) -> None:
    csv_path = tmp_path / "fred_qd_2020m01.csv"
    csv_path.write_text(
        "sasdate,GDPC1,UNRATE\n"
        "transform,5,2\n"
        "01/01/2020,100,4.0\n"
        "04/01/2020,110,4.5\n",
        encoding="utf-8",
    )
    panel = build_panel_from_files(
        {pd.Period("2020-01", freq="M"): csv_path},
        apply_transforms=True,
        exclude_from_transforms=("timestamp", "GDPC1"),
    )

    assert np.isclose(panel.loc[0, "GDPC1"], 100.0)
    assert np.isclose(panel.loc[1, "GDPC1"], 110.0)
    assert np.isnan(panel.loc[0, "UNRATE"])
    assert np.isclose(panel.loc[1, "UNRATE"], 0.5)
