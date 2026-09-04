from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import COVID_DUMMY_COLUMNS, HistoricalQuarterlyVintageProvider


def test_historical_provider_falls_back_to_qd_panel(tmp_path: Path) -> None:
    missing_csv_dir = tmp_path / "missing_csv_vintages"
    missing_csv_dir.mkdir(parents=True, exist_ok=True)

    panel_path = tmp_path / "panel.parquet"
    panel = pd.DataFrame(
        {
            "vintage": ["2020-01", "2020-01", "2020-02", "2020-02"],
            "vintage_timestamp": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]),
            "timestamp": pd.to_datetime(["2019-10-01", "2020-01-01", "2019-10-01", "2020-01-01"]),
            "GDPC1": [100.0, 101.0, 100.0, 102.0],
            "UNRATE": [3.5, 3.6, 3.5, 3.7],
        }
    )
    panel.to_parquet(panel_path, index=False)

    provider = HistoricalQuarterlyVintageProvider(
        historical_qd_dir=missing_csv_dir,
        target_series_name="LOG_REAL_GDP",
        target_transform="log_level",
        include_covariates=True,
        covariate_columns=["UNRATE"],
        apply_fred_transforms=True,
        qd_panel_path=panel_path,
    )

    assert provider.available_range_str() == "2020-01..2020-02"
    selected = provider.select_vintage_period(pd.Timestamp("2020-02-15"))
    assert str(selected) == "2020-01"  # month labels cannot justify same-month availability

    frame = provider._load_vintage_frame(pd.Period("2020-02", freq="M"))
    assert frame.shape[0] == 2
    assert np.isfinite(frame["target"]).all()
    assert "UNRATE" in frame.columns
    for col in COVID_DUMMY_COLUMNS:
        assert col in frame.columns
    assert frame.loc[0, "covid_dummy_2020q2"] == 0.0
    assert frame.loc[1, "covid_dummy_2020q2"] == 0.0
    assert frame.loc[:, "covid_dummy_2020q3"].eq(0.0).all()


def test_historical_provider_does_not_fill_unreleased_target_from_scaffold(tmp_path):
    from datasets import Dataset
    from types import SimpleNamespace
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"vintage": ["2019-06"] * 3,
                  "vintage_timestamp": pd.to_datetime(["2019-06-01"] * 3),
                  "timestamp": pd.to_datetime(["2019-03-01", "2019-06-01", "2019-09-01"]),
                  "GDPC1": [100., np.nan, np.nan], "UNRATE": [4., np.nan, np.nan]}).to_parquet(panel_path)
    provider = HistoricalQuarterlyVintageProvider(historical_qd_dir=tmp_path / "missing",
        target_series_name="GDPC1", target_transform="level", include_covariates=True,
        covariate_columns=["UNRATE"], qd_panel_path=panel_path, strict=True)
    task = SimpleNamespace(id_column="id", timestamp_column="timestamp", target="target",
                           known_dynamic_columns=[], past_dynamic_columns=["UNRATE"])
    past = Dataset.from_list([dict(id="GDP", timestamp=pd.to_datetime(["2019-01-01", "2019-04-01", "2019-07-01"]).tolist(),
                                  target=[999., 999., 999.], UNRATE=[999., 999., 999.])])
    result = provider.adapt_past_data(past, task)[0]
    assert result["target"][0] == 100.
    assert all(v is None or np.isnan(v) for v in result["target"][1:])
    assert 999. not in result["UNRATE"]
