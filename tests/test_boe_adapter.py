from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.boe_adapter import (  # noqa: E402
    _quarter_end_from_period,
    export_to_boe_schema,
)


def test_quarter_end_from_period() -> None:
    actual = _quarter_end_from_period("2020Q1")
    assert actual == pd.Timestamp("2020-03-31")


def test_export_shifts_fev_horizon_to_boe_horizon(tmp_path: Path) -> None:
    pred_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "model": ["naive_last"],
            "origin_quarter": ["2020Q1"],
            "target_quarter": ["2020Q1"],
            "horizon": [1],
            "release_stage": ["first"],
            "y_hat_level": [101.0],
            "y_true_level_first": [100.0],
        }
    ).to_csv(pred_csv, index=False)

    forecasts_path, _ = export_to_boe_schema(
        predictions_csv=pred_csv,
        release_table_csv=None,
        out_dir=tmp_path / "boe",
        truth="first",
        variable="GDPC1",
        metric="levels",
        forecast_value_col="y_hat_level",
    )

    forecasts = pd.read_csv(forecasts_path)
    assert forecasts["forecast_horizon"].tolist() == [0]


def test_first_release_outturn_alignment_matches_k0(tmp_path: Path) -> None:
    fe = pytest.importorskip("forecast_evaluation")

    pred_csv = tmp_path / "predictions.csv"
    rel_csv = tmp_path / "release.csv"

    pd.DataFrame(
        {
            "model": ["naive_last"],
            "origin_quarter": ["2020Q1"],
            "target_quarter": ["2020Q1"],
            "horizon": [1],
            "release_stage": ["first"],
            "y_hat_level": [101.0],
            "y_true_level_first": [100.0],
        }
    ).to_csv(pred_csv, index=False)

    pd.DataFrame(
        {
            "quarter": ["2020Q1"],
            "first_release_date": ["2020-04-29"],
            "second_release_date": ["2020-05-28"],
            "third_release_date": ["2020-06-25"],
        }
    ).to_csv(rel_csv, index=False)

    forecasts_path, outturns_path = export_to_boe_schema(
        predictions_csv=pred_csv,
        release_table_csv=rel_csv,
        out_dir=tmp_path / "boe",
        truth="first",
        variable="GDPC1",
        metric="levels",
        forecast_value_col="y_hat_level",
    )

    forecasts = pd.read_csv(forecasts_path, parse_dates=["date", "vintage_date"])
    outturns = pd.read_csv(outturns_path, parse_dates=["date", "vintage_date"])
    data = fe.ForecastData(forecasts_data=forecasts, outturns_data=outturns)

    main_table = data._main_table.copy()
    assert "k" in main_table.columns
    assert 0 in set(pd.to_numeric(main_table["k"], errors="coerce").dropna().astype(int).tolist())
