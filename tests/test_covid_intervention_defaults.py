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

from fev_macro.models.base import apply_default_covid_intervention
from fev_macro.models.baselines import NaiveLast


class _DummyTask:
    id_column = "id"
    timestamp_column = "timestamp"
    target = "target"
    horizon = 1
    known_dynamic_columns: list[str] = []
    past_dynamic_columns: list[str] = []


def test_apply_default_covid_intervention_no_pandemic_quarters_is_noop() -> None:
    past_q = pd.period_range("2017Q1", "2018Q4", freq="Q-DEC")
    future_q = pd.period_range("2019Q1", "2019Q2", freq="Q-DEC")
    y = np.linspace(1.0, 8.0, len(past_q))

    adjusted, future_effect = apply_default_covid_intervention(
        y=y,
        past_timestamps=past_q.to_timestamp(how="end").to_numpy(),
        future_timestamps=future_q.to_timestamp(how="end").to_numpy(),
        horizon=2,
    )

    assert np.allclose(adjusted, y)
    assert np.allclose(future_effect, np.zeros(2, dtype=float))


def test_naive_last_uses_default_covid_adjustment() -> None:
    task = _DummyTask()
    past_q = pd.period_range("2018Q3", "2020Q2", freq="Q-DEC")
    future_q = pd.period_range("2020Q3", "2020Q3", freq="Q-DEC")

    baseline = np.linspace(100.0, 107.0, len(past_q))
    y = baseline.copy()
    y[-1] = y[-1] + 25.0

    past_ts = past_q.to_timestamp(how="end").strftime("%Y-%m-%d").tolist()
    future_ts = future_q.to_timestamp(how="end").strftime("%Y-%m-%d").tolist()

    past_data = Dataset.from_dict({"id": ["s1"], "timestamp": [past_ts], "target": [y.tolist()]})
    future_data = Dataset.from_dict({"id": ["s1"], "timestamp": [future_ts]})

    adjusted, _ = apply_default_covid_intervention(
        y=y,
        past_timestamps=past_q.to_timestamp(how="end").to_numpy(),
        future_timestamps=future_q.to_timestamp(how="end").to_numpy(),
        horizon=1,
    )
    assert adjusted[-1] < y[-1]

    model = NaiveLast()
    out = model.predict(past_data=past_data, future_data=future_data, task=task)
    pred = float(np.asarray(out["predictions"], dtype=float)[0][0])
    assert np.isfinite(pred)
    assert pred < y[-1]
