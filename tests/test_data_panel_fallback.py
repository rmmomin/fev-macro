from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import HistoricalQuarterlyVintageProvider


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
    assert str(selected) == "2020-02"

    frame = provider._load_vintage_frame(pd.Period("2020-02", freq="M"))
    assert frame.shape[0] == 2
    assert np.isfinite(frame["target"]).all()
    assert "UNRATE" in frame.columns
