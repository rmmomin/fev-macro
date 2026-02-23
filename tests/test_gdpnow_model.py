from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.models.gdpnow import (  # noqa: E402
    AtlantaFedGDPNowAsOfWindowCutoffModel,
    AtlantaFedGDPNowFinalPreReleaseModel,
)

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "gdpnow"
FIXTURE_GDPNOW = FIXTURE_DIR / "atlantafedgdpnow.csv"
FIXTURE_RELEASE = FIXTURE_DIR / "gdpc1_releases_first_second_third.csv"


class _DummyTask:
    id_column = "id"
    timestamp_column = "timestamp"
    target = "target"
    horizon = 1
    known_dynamic_columns: list[str] = []
    past_dynamic_columns: list[str] = []
    task_name = "gdp_h1"
    eval_metric = "RMSE"


def _datasets(*, cutoff_ts: str, future_ts: str, last_target: float = 0.8) -> tuple[Dataset, Dataset]:
    past_data = Dataset.from_dict({"id": ["gdp"], "timestamp": [[cutoff_ts]], "target": [[last_target]]})
    future_data = Dataset.from_dict({"id": ["gdp"], "timestamp": [[future_ts]]})
    return past_data, future_data


def test_final_pre_release_picks_latest_strictly_before_first_release() -> None:
    model = AtlantaFedGDPNowFinalPreReleaseModel(
        gdpnow_csv_path=FIXTURE_GDPNOW,
        release_csv_path=FIXTURE_RELEASE,
    )
    task = _DummyTask()
    past_data, future_data = _datasets(cutoff_ts="2020-07-20", future_ts="2020-04-01")

    out = model.predict(past_data=past_data, future_data=future_data, task=task)
    pred = float(np.asarray(out["predictions"], dtype=float)[0][0])
    assert pred == 2.0

    debug = pd.DataFrame(model.get_selection_debug_rows())
    row = debug.iloc[-1]
    assert pd.Timestamp(row["selected_forecast_date"]) == pd.Timestamp("2020-07-30")
    assert pd.Timestamp(row["bea_first_release_date"]) == pd.Timestamp("2020-07-31")
    assert bool(row["is_on_or_after_first_release"]) is False


def test_final_pre_release_excludes_same_day_first_release_update(tmp_path: Path) -> None:
    gdpnow_csv = tmp_path / "same_day_only.csv"
    gdpnow_csv.write_text(
        (
            "Forecast Date,Quarter being forecasted,GDP Nowcast\n"
            "7/31/20,6/30/20,2.5\n"
            "8/1/20,6/30/20,3.0\n"
        ),
        encoding="utf-8",
    )

    model = AtlantaFedGDPNowFinalPreReleaseModel(
        gdpnow_csv_path=gdpnow_csv,
        release_csv_path=FIXTURE_RELEASE,
    )
    task = _DummyTask()
    past_data, future_data = _datasets(cutoff_ts="2020-07-20", future_ts="2020-04-01", last_target=1.25)

    out = model.predict(past_data=past_data, future_data=future_data, task=task)
    pred = float(np.asarray(out["predictions"], dtype=float)[0][0])
    assert pred == 1.25

    debug = pd.DataFrame(model.get_selection_debug_rows())
    row = debug.iloc[-1]
    assert pd.isna(row["selected_forecast_date"])


def test_asof_window_cutoff_picks_latest_update_on_or_before_cutoff() -> None:
    model = AtlantaFedGDPNowAsOfWindowCutoffModel(
        gdpnow_csv_path=FIXTURE_GDPNOW,
        release_csv_path=FIXTURE_RELEASE,
    )
    task = _DummyTask()
    past_data, future_data = _datasets(cutoff_ts="2020-07-20", future_ts="2020-04-01")

    out = model.predict(past_data=past_data, future_data=future_data, task=task)
    pred = float(np.asarray(out["predictions"], dtype=float)[0][0])
    assert pred == 1.0

    debug = pd.DataFrame(model.get_selection_debug_rows())
    row = debug.iloc[-1]
    assert pd.Timestamp(row["selected_forecast_date"]) == pd.Timestamp("2020-07-15")
    assert bool(row["is_after_window_cutoff"]) is False


def test_asof_window_cutoff_returns_nan_when_no_update_available_by_cutoff() -> None:
    model = AtlantaFedGDPNowAsOfWindowCutoffModel(
        gdpnow_csv_path=FIXTURE_GDPNOW,
        release_csv_path=FIXTURE_RELEASE,
    )
    task = _DummyTask()
    past_data, future_data = _datasets(cutoff_ts="2020-08-01", future_ts="2020-07-01")

    out = model.predict(past_data=past_data, future_data=future_data, task=task)
    pred = float(np.asarray(out["predictions"], dtype=float)[0][0])
    assert np.isnan(pred)

    debug = pd.DataFrame(model.get_selection_debug_rows())
    row = debug.iloc[-1]
    assert pd.isna(row["selected_forecast_date"])
