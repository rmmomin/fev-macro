from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import apply_gdpc1_release_truth_target, build_release_target_scaffold
from fev_macro.eval_runner import validate_y_true_matches_release_table


def test_truth_item_id_matches_realtime_saar_first_release(tmp_path: Path) -> None:
    release_csv = tmp_path / "gdpc1_releases.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "first_release": [100.0, 110.0, 121.0],
            "second_release": [100.5, 110.5, 121.5],
            "third_release": [101.0, 111.0, 122.0],
            "latest_release": [102.0, 112.0, 123.0],
            "qoq_saar_growth_realtime_first_pct": [1.1, 2.2, 3.3],
            "qoq_saar_growth_realtime_second_pct": [1.0, 2.0, 3.0],
            "qoq_saar_growth_realtime_third_pct": [0.9, 1.9, 2.9],
        }
    ).to_csv(release_csv, index=False)

    base_df, _ = build_release_target_scaffold(
        release_csv_path=release_csv,
        target_series_name="LOG_REAL_GDP",
    )
    out, meta = apply_gdpc1_release_truth_target(
        dataset_df=base_df,
        release_csv_path=release_csv,
        release_stage="first",
        release_metric="realtime_qoq_saar",
        target_transform="saar_growth",
    )

    assert out["item_id"].nunique() == 1
    assert str(out["item_id"].iloc[0]) == "gdpc1_qoq_saar_first_pct"
    assert meta["item_id"] == "gdpc1_qoq_saar_first_pct"
    assert meta["target_units"] == "pct_qoq_saar"
    assert np.allclose(
        out["target"].to_numpy(dtype=float),
        np.array([1.1, 2.2, 3.3], dtype=float),
    )


def test_validate_y_true_matches_release_table_for_realtime_stages(tmp_path: Path) -> None:
    release_csv = tmp_path / "gdpc1_releases.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "first_release": [100.0, 110.0, 121.0],
            "second_release": [100.5, 110.5, 121.5],
            "third_release": [101.0, 111.0, 122.0],
            "latest_release": [102.0, 112.0, 123.0],
            "qoq_saar_growth_realtime_first_pct": [1.1, 2.2, 3.3],
            "qoq_saar_growth_realtime_second_pct": [1.0, 2.0, 3.0],
            "qoq_saar_growth_realtime_third_pct": [0.9, 1.9, 2.9],
        }
    ).to_csv(release_csv, index=False)

    base_df, _ = build_release_target_scaffold(
        release_csv_path=release_csv,
        target_series_name="LOG_REAL_GDP",
    )

    rows: list[dict[str, object]] = []
    for stage in ("first", "second", "third"):
        stage_df, meta = apply_gdpc1_release_truth_target(
            dataset_df=base_df,
            release_csv_path=release_csv,
            release_stage=stage,
            release_metric="realtime_qoq_saar",
            target_transform="saar_growth",
        )
        for _, rec in stage_df.iterrows():
            rows.append(
                {
                    "task_name": f"gdp_{stage}_h1",
                    "timestamp": rec["timestamp"],
                    "y_true": float(rec["target"]),
                    "release_stage": stage,
                    "release_metric": meta["release_metric"],
                }
            )

    records_df = pd.DataFrame(rows)
    stats = validate_y_true_matches_release_table(
        records_df=records_df,
        release_csv_path=release_csv,
        tolerance=1e-12,
        strict=True,
    )

    assert set(stats.keys()) == {"first", "second", "third"}
    for stage in ("first", "second", "third"):
        assert stats[stage]["num_bad"] == 0
        assert float(stats[stage]["max_abs_diff"]) == 0.0


def test_validate_y_true_matches_release_table_raises_on_mismatch(tmp_path: Path) -> None:
    release_csv = tmp_path / "gdpc1_releases.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "first_release": [100.0, 110.0, 121.0],
            "second_release": [100.5, 110.5, 121.5],
            "third_release": [101.0, 111.0, 122.0],
            "latest_release": [102.0, 112.0, 123.0],
            "qoq_saar_growth_realtime_first_pct": [1.1, 2.2, 3.3],
            "qoq_saar_growth_realtime_second_pct": [1.0, 2.0, 3.0],
            "qoq_saar_growth_realtime_third_pct": [0.9, 1.9, 2.9],
        }
    ).to_csv(release_csv, index=False)

    records_df = pd.DataFrame(
        {
            "task_name": ["gdp_first_h1", "gdp_first_h1", "gdp_first_h1"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01"]),
            "y_true": [1.1, 2.2, 3.8],
            "release_stage": ["first", "first", "first"],
            "release_metric": ["realtime_qoq_saar", "realtime_qoq_saar", "realtime_qoq_saar"],
        }
    )

    with pytest.raises(ValueError, match="Release-truth validation failed"):
        validate_y_true_matches_release_table(
            records_df=records_df,
            release_csv_path=release_csv,
            tolerance=1e-12,
            strict=True,
        )
