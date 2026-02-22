from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.eval_runner import _augment_leaderboard_with_point_error_metrics


def test_leaderboard_includes_ex_covid_error_columns() -> None:
    leaderboard_df = pd.DataFrame({"model_name": ["m1", "m2"], "win_rate": [0.5, 0.5]})
    records_df = pd.DataFrame(
        {
            "model_name": ["m1", "m2", "m1", "m2"],
            "timestamp": pd.to_datetime(["2019-10-01", "2019-10-01", "2020-04-01", "2020-04-01"]),
            "y_true": [1.0, 1.0, -32.0, -32.0],
            "y_pred": [1.2, 0.7, -10.0, -30.0],
        }
    )

    out = _augment_leaderboard_with_point_error_metrics(leaderboard_df=leaderboard_df, records_df=records_df)
    for col in ("MAE", "MSE", "RMSE", "MAE_ex_covid", "MSE_ex_covid", "RMSE_ex_covid"):
        assert col in out.columns
        assert out[col].notna().all()


def test_leaderboard_ranks_by_rmse_ex_covid() -> None:
    leaderboard_df = pd.DataFrame({"model_name": ["m1", "m2"], "win_rate": [0.5, 0.5]})
    records_df = pd.DataFrame(
        {
            "model_name": ["m1", "m2", "m1", "m2", "m1", "m2"],
            "timestamp": pd.to_datetime(
                ["2019-10-01", "2019-10-01", "2020-04-01", "2020-04-01", "2020-07-01", "2020-07-01"]
            ),
            "y_true": [0.0, 0.0, -32.0, -32.0, 33.0, 33.0],
            # m1 is better ex-COVID, m2 is much better in COVID outliers.
            "y_pred": [0.0, 1.0, 28.0, -32.0, -27.0, 33.0],
        }
    )

    out = _augment_leaderboard_with_point_error_metrics(leaderboard_df=leaderboard_df, records_df=records_df)
    assert list(out["model_name"]) == ["m1", "m2"]
    assert float(out.loc[out["model_name"] == "m1", "RMSE_ex_covid"].iloc[0]) < float(
        out.loc[out["model_name"] == "m2", "RMSE_ex_covid"].iloc[0]
    )
