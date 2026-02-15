from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_gdp_releases import build_release_dataset


def _saar(curr: float, prev: float) -> float:
    return float(100.0 * ((curr / prev) ** 4 - 1.0))


def test_build_release_dataset_realtime_growth_uses_previous_level_asof_current_release() -> None:
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
                "2024-05",
                "2024-05",
                "2024-06",
                "2024-06",
                "2024-07",
                "2024-07",
            ],
            "vintage_timestamp": pd.to_datetime(
                ["2024-05-01", "2024-05-01", "2024-06-01", "2024-06-01", "2024-07-01", "2024-07-01"]
            ),
            "quarter": pd.PeriodIndex(["2024Q1", "2024Q2", "2024Q1", "2024Q2", "2024Q1", "2024Q2"], freq="Q-DEC"),
            "quarter_ord": pd.PeriodIndex(["2024Q1", "2024Q2", "2024Q1", "2024Q2", "2024Q1", "2024Q2"], freq="Q-DEC").astype(
                "int64"
            ),
            "value": [101.0, 110.0, 102.0, 111.0, 103.0, 112.0],
        }
    )

    out = build_release_dataset(wide=wide, series="GDPC1", qd_panel=qd_panel)

    # First row has no previous quarter for q/q growth.
    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_first_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_second_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_third_pct"])

    # For 2024Q2, previous-quarter levels are taken as-of each release vintage:
    # first uses 2024-05-30 vintage previous level=101
    # second uses 2024-06-27 vintage previous level=102
    # third uses 2024-07-25 vintage previous level=103
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_first_pct"], _saar(110.0, 101.0))
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_second_pct"], _saar(111.0, 102.0))
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_third_pct"], _saar(112.0, 103.0))
    assert np.isclose(out.loc[1, "qoq_growth_realtime_first_pct"], (110.0 / 101.0 - 1.0) * 100.0)


def test_realtime_growth_regression_quarters_2023_no_series_break_spikes() -> None:
    path = ROOT / "data" / "panels" / "gdpc1_releases_first_second_third.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    required = {
        "observation_date",
        "qoq_saar_growth_realtime_first_pct",
        "qoq_saar_growth_realtime_third_pct",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise AssertionError(f"Missing realtime growth columns in release CSV: {missing}")

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    row_2023q2 = df.loc[df["observation_date"] == pd.Timestamp("2023-04-01")]
    row_2023q3 = df.loc[df["observation_date"] == pd.Timestamp("2023-07-01")]
    if row_2023q2.empty or row_2023q3.empty:
        return

    v_third_2023q2 = float(row_2023q2["qoq_saar_growth_realtime_third_pct"].iloc[0])
    v_first_2023q3 = float(row_2023q3["qoq_saar_growth_realtime_first_pct"].iloc[0])

    assert np.isfinite(v_third_2023q2)
    assert np.isfinite(v_first_2023q3)
    assert abs(v_third_2023q2) < 15.0
    assert abs(v_first_2023q3) < 15.0
