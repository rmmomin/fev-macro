from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.models.gdpnow import AtlantaFedGDPNowModel


class _DummyTask:
    id_column = "id"
    timestamp_column = "timestamp"
    target = "target"
    horizon = 1
    known_dynamic_columns: list[str] = []
    past_dynamic_columns: list[str] = []


def test_gdpnow_model_uses_latest_nowcast_before_first_release(tmp_path: Path) -> None:
    gdpnow_csv = tmp_path / "atlantafedgdpnow.csv"
    gdpnow_csv.write_text(
        (
            "Forecast Date,Quarter being forecasted,GDP Nowcast\n"
            "7/15/20,6/30/20,1.0\n"
            "7/30/20,6/30/20,2.0\n"
            "8/1/20,6/30/20,3.0\n"
        ),
        encoding="utf-8",
    )

    release_csv = tmp_path / "gdpc1_release.csv"
    release_csv.write_text(
        "observation_date,first_release_date\n2020-04-01,2020-07-31\n",
        encoding="utf-8",
    )

    model = AtlantaFedGDPNowModel(gdpnow_csv_path=gdpnow_csv, release_csv_path=release_csv)
    task = _DummyTask()
    past_data = Dataset.from_dict({"id": ["gdp"], "timestamp": [["2020-01-01"]], "target": [[0.8]]})
    future_data = Dataset.from_dict({"id": ["gdp"], "timestamp": [["2020-04-01"]]})

    out = model.predict(past_data=past_data, future_data=future_data, task=task)
    pred = float(np.asarray(out["predictions"], dtype=float)[0][0])
    assert pred == 2.0


def test_gdpnow_model_falls_back_to_last_history_when_missing_quarter(tmp_path: Path) -> None:
    gdpnow_csv = tmp_path / "atlantafedgdpnow.csv"
    gdpnow_csv.write_text(
        "Forecast Date,Quarter being forecasted,GDP Nowcast\n7/15/20,6/30/20,1.0\n",
        encoding="utf-8",
    )

    release_csv = tmp_path / "gdpc1_release.csv"
    release_csv.write_text(
        "observation_date,first_release_date\n2020-04-01,2020-07-31\n",
        encoding="utf-8",
    )

    model = AtlantaFedGDPNowModel(gdpnow_csv_path=gdpnow_csv, release_csv_path=release_csv)
    task = _DummyTask()
    past_data = Dataset.from_dict({"id": ["gdp"], "timestamp": [["2020-01-01"]], "target": [[1.25]]})
    future_data = Dataset.from_dict({"id": ["gdp"], "timestamp": [["2020-07-01"]]})

    out = model.predict(past_data=past_data, future_data=future_data, task=task)
    pred = float(np.asarray(out["predictions"], dtype=float)[0][0])
    assert pred == 1.25
