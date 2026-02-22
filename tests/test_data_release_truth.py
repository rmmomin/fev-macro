from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import apply_gdpc1_release_truth_target


def test_apply_gdpc1_release_truth_target_replaces_target_by_quarter(tmp_path: Path) -> None:
    dataset_df = pd.DataFrame(
        {
            "item_id": ["LOG_REAL_GDP"] * 3,
            "timestamp": pd.to_datetime(["2024-03-01", "2024-06-01", "2024-09-01"]),
            "target": [0.0, 0.0, 0.0],
            "UNRATE": [3.8, 3.9, 4.0],
        }
    )

    release_csv = tmp_path / "gdpc1_releases.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "first_release": [100.0, 110.0, 121.0],
            "second_release": [100.5, 110.5, 121.5],
            "third_release": [101.0, 111.0, 122.0],
            "latest_release": [102.0, 112.0, 123.0],
        }
    ).to_csv(release_csv, index=False)

    out, meta = apply_gdpc1_release_truth_target(
        dataset_df=dataset_df,
        release_csv_path=release_csv,
        release_stage="first",
        target_transform="log_level",
    )

    expected = np.log(np.array([100.0, 110.0, 121.0], dtype=float))
    assert np.allclose(out["target"].to_numpy(dtype=float), expected)
    assert out["UNRATE"].tolist() == [3.8, 3.9, 4.0]
    assert meta["source"] == "gdpc1_release_csv"
    assert meta["release_stage"] == "first"
    assert meta["release_column"] == "first_release"


def test_apply_gdpc1_release_truth_target_drops_missing_release_stage(tmp_path: Path) -> None:
    dataset_df = pd.DataFrame(
        {
            "item_id": ["LOG_REAL_GDP"] * 3,
            "timestamp": pd.to_datetime(["2024-03-01", "2024-06-01", "2024-09-01"]),
            "target": [1.0, 1.0, 1.0],
            "UNRATE": [3.8, 3.9, 4.0],
        }
    )

    release_csv = tmp_path / "gdpc1_releases.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "first_release": [100.0, 110.0, 121.0],
            "second_release": [100.5, 110.5, 121.5],
            "third_release": [101.0, np.nan, 122.0],
            "latest_release": [102.0, 112.0, 123.0],
        }
    ).to_csv(release_csv, index=False)

    out, meta = apply_gdpc1_release_truth_target(
        dataset_df=dataset_df,
        release_csv_path=release_csv,
        release_stage="third_release",
        target_transform="level",
    )

    assert out["timestamp"].tolist() == [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-09-01")]
    assert out["target"].tolist() == [101.0, 122.0]
    assert meta["release_stage"] == "third"
    assert meta["release_column"] == "third_release"


def test_apply_gdpc1_release_truth_target_realtime_qoq_saar(tmp_path: Path) -> None:
    dataset_df = pd.DataFrame(
        {
            "item_id": ["LOG_REAL_GDP"] * 3,
            "timestamp": pd.to_datetime(["2024-03-01", "2024-06-01", "2024-09-01"]),
            "target": [0.0, 0.0, 0.0],
            "UNRATE": [3.8, 3.9, 4.0],
        }
    )

    release_csv = tmp_path / "gdpc1_releases.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "first_release": [100.0, 110.0, 121.0],
            "second_release": [100.5, 110.5, 121.5],
            "third_release": [101.0, 111.0, 122.0],
            "latest_release": [102.0, 112.0, 123.0],
            "qoq_saar_growth_realtime_first_pct": [np.nan, 7.5, 8.1],
            "qoq_saar_growth_realtime_second_pct": [np.nan, 7.2, 7.9],
            "qoq_saar_growth_realtime_third_pct": [np.nan, 6.9, 7.7],
        }
    ).to_csv(release_csv, index=False)

    out, meta = apply_gdpc1_release_truth_target(
        dataset_df=dataset_df,
        release_csv_path=release_csv,
        release_stage="second",
        release_metric="realtime_qoq_saar",
        target_transform="saar_growth",
    )

    assert out["timestamp"].tolist() == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-01")]
    assert out["target"].tolist() == [7.2, 7.9]
    assert meta["release_stage"] == "second"
    assert meta["release_metric"] == "realtime_qoq_saar"
    assert meta["release_column"] == "qoq_saar_growth_realtime_second_pct"


def test_apply_gdpc1_release_truth_target_alfred_qoq(tmp_path: Path) -> None:
    dataset_df = pd.DataFrame(
        {
            "item_id": ["LOG_REAL_GDP"] * 3,
            "timestamp": pd.to_datetime(["2024-03-01", "2024-06-01", "2024-09-01"]),
            "target": [0.0, 0.0, 0.0],
            "UNRATE": [3.8, 3.9, 4.0],
        }
    )

    release_csv = tmp_path / "gdpc1_releases.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "first_release": [100.0, 110.0, 121.0],
            "second_release": [100.5, 110.5, 121.5],
            "third_release": [101.0, 111.0, 122.0],
            "latest_release": [102.0, 112.0, 123.0],
            "qoq_growth_alfred_first_pct": [np.nan, 2.0, 2.2],
            "qoq_growth_alfred_second_pct": [np.nan, 2.1, 2.3],
            "qoq_growth_alfred_third_pct": [np.nan, 1.9, 2.1],
        }
    ).to_csv(release_csv, index=False)

    out, meta = apply_gdpc1_release_truth_target(
        dataset_df=dataset_df,
        release_csv_path=release_csv,
        release_stage="second",
        release_metric="alfred_qoq",
        target_transform="qoq_growth",
    )

    assert out["timestamp"].tolist() == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-01")]
    assert out["target"].tolist() == [2.1, 2.3]
    assert meta["release_stage"] == "second"
    assert meta["release_metric"] == "alfred_qoq"
    assert meta["release_column"] == "qoq_growth_alfred_second_pct"


def test_apply_gdpc1_release_truth_target_alfred_qoq_saar(tmp_path: Path) -> None:
    dataset_df = pd.DataFrame(
        {
            "item_id": ["LOG_REAL_GDP"] * 3,
            "timestamp": pd.to_datetime(["2024-03-01", "2024-06-01", "2024-09-01"]),
            "target": [0.0, 0.0, 0.0],
            "UNRATE": [3.8, 3.9, 4.0],
        }
    )

    release_csv = tmp_path / "gdpc1_releases.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "first_release": [100.0, 110.0, 121.0],
            "second_release": [100.5, 110.5, 121.5],
            "third_release": [101.0, 111.0, 122.0],
            "latest_release": [102.0, 112.0, 123.0],
            "qoq_saar_growth_alfred_first_pct": [np.nan, 8.3, 8.5],
            "qoq_saar_growth_alfred_second_pct": [np.nan, 8.1, 8.4],
            "qoq_saar_growth_alfred_third_pct": [np.nan, 7.9, 8.2],
        }
    ).to_csv(release_csv, index=False)

    out, meta = apply_gdpc1_release_truth_target(
        dataset_df=dataset_df,
        release_csv_path=release_csv,
        release_stage="second",
        release_metric="alfred_qoq_saar",
        target_transform="saar_growth",
    )

    assert out["timestamp"].tolist() == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-01")]
    assert out["target"].tolist() == [8.1, 8.4]
    assert meta["release_stage"] == "second"
    assert meta["release_metric"] == "alfred_qoq_saar"
    assert meta["release_column"] == "qoq_saar_growth_alfred_second_pct"
