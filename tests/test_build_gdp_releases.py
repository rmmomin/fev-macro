from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_gdp_releases import (  # noqa: E402
    STAGE_ALFRED_PREV_LEVEL_COLS,
    STAGE_QOQ_SAAR_ALFRED_COLS,
    _format_release_output_for_alfred_qoq_saar,
    build_release_dataset,
    load_qd_vintage_series_panel,
    validate_release_table,
)


def _saar(curr: float, prev: float) -> float:
    return float(100.0 * ((curr / prev) ** 4 - 1.0))


def test_build_release_dataset_realtime_growth_uses_single_vintage_snapshot() -> None:
    wide = pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01"],
            "GDPC1_20240425": [100.0, np.nan],
            "GDPC1_20240530": [101.0, 110.0],
            "GDPC1_20240627": [102.0, 111.0],
            "GDPC1_20240725": [103.0, 112.0],
        }
    )

    qd_panel = pd.DataFrame(
        {
            "vintage": [
                "2024-06",
                "2024-06",
                "2024-07",
                "2024-07",
                "2024-08",
                "2024-08",
            ],
            "vintage_timestamp": pd.to_datetime(
                ["2024-06-01", "2024-06-01", "2024-07-01", "2024-07-01", "2024-08-01", "2024-08-01"]
            ),
            "quarter": pd.PeriodIndex(["2024Q1", "2024Q2", "2024Q1", "2024Q2", "2024Q1", "2024Q2"], freq="Q-DEC"),
            "quarter_ord": pd.PeriodIndex(["2024Q1", "2024Q2", "2024Q1", "2024Q2", "2024Q1", "2024Q2"], freq="Q-DEC").astype(
                "int64"
            ),
            "value": [101.0, 110.0, 102.0, 111.0, 103.0, 112.0],
        }
    )

    out = build_release_dataset(wide=wide, series="GDPC1", qd_panel=qd_panel, vintage_select="next")

    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_first_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_second_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_third_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_alfred_first_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_alfred_second_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_alfred_third_pct"])

    # For 2024Q2, each stage maps to the next available monthly panel vintage:
    # first: release 2024-05-30 -> panel 2024-06, second: 2024-06-27 -> 2024-07,
    # third: 2024-07-25 -> 2024-08.
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_first_pct"], _saar(110.0, 101.0))
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_second_pct"], _saar(111.0, 102.0))
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_third_pct"], _saar(112.0, 103.0))
    assert np.isclose(out.loc[1, "qoq_growth_realtime_first_pct"], (110.0 / 101.0 - 1.0) * 100.0)
    assert np.isclose(out.loc[1, "qoq_saar_growth_alfred_first_pct"], _saar(110.0, 101.0))
    assert np.isclose(out.loc[1, "qoq_saar_growth_alfred_second_pct"], _saar(111.0, 102.0))
    assert np.isclose(out.loc[1, "qoq_saar_growth_alfred_third_pct"], _saar(112.0, 103.0))
    assert np.isclose(out.loc[1, "qoq_growth_alfred_first_pct"], (110.0 / 101.0 - 1.0) * 100.0)


def test_reindex_break_regression_uses_same_vintage_for_numerator_and_denominator() -> None:
    # v0 and v1 describe identical real growth, but v1 is 10% higher in level due to reindexing.
    wide = pd.DataFrame(
        {
            "observation_date": ["2023-01-01", "2023-04-01"],
            "GDPC1_20230525": [100.0, np.nan],
            "GDPC1_20230830": [110.0, 110.0],
        }
    )

    qd_panel = pd.DataFrame(
        {
            "vintage": ["2023-06", "2023-09", "2023-09"],
            "vintage_timestamp": pd.to_datetime(["2023-06-01", "2023-09-01", "2023-09-01"]),
            "quarter": pd.PeriodIndex(["2023Q1", "2023Q1", "2023Q2"], freq="Q-DEC"),
            "quarter_ord": pd.PeriodIndex(["2023Q1", "2023Q1", "2023Q2"], freq="Q-DEC").astype("int64"),
            "value": [100.0, 110.0, 110.0],
        }
    )

    out = build_release_dataset(wide=wide, series="GDPC1", qd_panel=qd_panel, vintage_select="next")

    # Old stitched-level approach mixes vintages and creates a fake jump.
    old_stitched = _saar(float(out.loc[1, "first_release"]), float(out.loc[0, "first_release"]))
    assert old_stitched > 40.0

    # ALFRED same-vintage growth uses the stage release vintage for both q and q-1.
    assert np.isclose(float(out.loc[1, "qoq_saar_growth_alfred_first_pct"]), 0.0)

    # New method uses one panel snapshot (v1) for both q and q-1, avoiding the break.
    assert np.isclose(float(out.loc[1, "qoq_saar_growth_realtime_first_pct"]), 0.0)


def test_format_release_output_keeps_only_alfred_saar_growth_and_alfred_levels() -> None:
    wide = pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01"],
            "GDPC1_20240425": [100.0, np.nan],
            "GDPC1_20240530": [101.0, 110.0],
            "GDPC1_20240627": [102.0, 111.0],
            "GDPC1_20240725": [103.0, 112.0],
        }
    )
    qd_panel = pd.DataFrame(
        {
            "vintage": ["2024-06", "2024-06", "2024-07", "2024-07", "2024-08", "2024-08"],
            "vintage_timestamp": pd.to_datetime(
                ["2024-06-01", "2024-06-01", "2024-07-01", "2024-07-01", "2024-08-01", "2024-08-01"]
            ),
            "quarter": pd.PeriodIndex(["2024Q1", "2024Q2", "2024Q1", "2024Q2", "2024Q1", "2024Q2"], freq="Q-DEC"),
            "quarter_ord": pd.PeriodIndex(["2024Q1", "2024Q2", "2024Q1", "2024Q2", "2024Q1", "2024Q2"], freq="Q-DEC").astype(
                "int64"
            ),
            "value": [101.0, 110.0, 102.0, 111.0, 103.0, 112.0],
        }
    )

    full = build_release_dataset(wide=wide, series="GDPC1", qd_panel=qd_panel, vintage_select="next")
    out = _format_release_output_for_alfred_qoq_saar(full)

    growth_cols = [c for c in out.columns if c.startswith("qoq_")]
    assert set(growth_cols) == set(STAGE_QOQ_SAAR_ALFRED_COLS.values())
    assert "qoq_saar_growth_realtime_first_pct" not in out.columns
    assert "qoq_growth_alfred_first_pct" not in out.columns
    assert "qoq_saar_growth_latest_pct" not in out.columns

    for col in STAGE_ALFRED_PREV_LEVEL_COLS.values():
        assert col in out.columns

    prev_col = STAGE_ALFRED_PREV_LEVEL_COLS["first"]
    saar_col = STAGE_QOQ_SAAR_ALFRED_COLS["first"]
    y_q = float(out.loc[1, "first_release"])
    y_prev = float(out.loc[1, prev_col])
    expected = _saar(y_q, y_prev)
    assert np.isclose(float(out.loc[1, saar_col]), expected)


def test_validation_smoke_no_2023_q2_q3_spike_flags_with_panel_growth(tmp_path: Path) -> None:
    release_path = ROOT / "data" / "panels" / "gdpc1_releases_first_second_third.csv"
    panel_path = ROOT / "data" / "panels" / "fred_qd_vintage_panel.parquet"
    if not release_path.exists() or not panel_path.exists():
        return

    try:
        panel_df = load_qd_vintage_series_panel(panel_path=panel_path, series="GDPC1")
    except RuntimeError as exc:
        if "parquet engine" in str(exc):
            return
        raise

    releases_df = pd.read_csv(release_path)
    releases_df["observation_date"] = pd.to_datetime(releases_df["observation_date"], errors="coerce")
    releases_df = releases_df.loc[releases_df["observation_date"] >= pd.Timestamp("2018-01-01")].copy()

    report_path = tmp_path / "gdpc1_release_validation_report.csv"
    report_df, _summary = validate_release_table(
        releases_df=releases_df,
        panel_df=panel_df,
        vintage_select="next",
        report_path=report_path,
    )

    assert report_path.exists()
    spike_rows = report_df.loc[
        report_df["flag_type"].isin({"spike_abs_gt_15", "spike_delta_gt_12"})
        & report_df["quarter"].isin({"2023Q2", "2023Q3"})
    ]
    assert spike_rows.empty
