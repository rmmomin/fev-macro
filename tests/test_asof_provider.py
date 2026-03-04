from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.asof_provider import AsofVintageProvider
from fev_macro.asof_store import AsofStore


def test_asof_provider_adapts_train_df_from_snapshot(tmp_path) -> None:
    import pytest

    pytest.importorskip("duckdb")
    db_path = tmp_path / "asof.duckdb"
    store = AsofStore(db_path=db_path)
    try:
        versions = pd.DataFrame(
            [
                {"series_id": "GDPC1", "obs_ts": "2020-01-01", "asof_ts": "2020-02-15", "value": 100.0},
                {"series_id": "GDPC1", "obs_ts": "2020-04-01", "asof_ts": "2020-05-15", "value": 101.0},
                {"series_id": "UNRATE", "obs_ts": "2020-01-01", "asof_ts": "2020-02-15", "value": 4.0},
                {"series_id": "UNRATE", "obs_ts": "2020-04-01", "asof_ts": "2020-05-15", "value": 5.0},
            ]
        )
        store.ingest_versions(versions, source="unit_test")
    finally:
        store.close()

    provider = AsofVintageProvider(
        db_path=db_path,
        covariate_mode="unprocessed",
        universe="both",
    )
    try:
        train_df = pd.DataFrame(
            {
                "quarter": pd.PeriodIndex(["2020Q1", "2020Q2"], freq="Q-DEC"),
                "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")],
                "GDPC1": [90.0, np.nan],
                "UNRATE": [9.0, 9.0],
            }
        )
        adapted, meta = provider.adapt_train_df(
            train_df=train_df,
            asof_ts="2020-06-01",
            cutoff_quarter=pd.Period("2020Q2", freq="Q-DEC"),
            target_col="GDPC1",
        )
        assert bool(meta.get("used_snapshot", False))
        assert np.isclose(float(adapted.loc[0, "GDPC1"]), 100.0)
        assert np.isclose(float(adapted.loc[1, "GDPC1"]), 101.0)
        assert np.isclose(float(adapted.loc[0, "UNRATE"]), 4.0)
        assert np.isclose(float(adapted.loc[1, "UNRATE"]), 5.0)
    finally:
        provider.close()
