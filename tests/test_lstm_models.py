from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("torch")

from fev_macro.models.lstm_models import LSTMMultivariateModel, LSTMUnivariateModel


class _DummyTask:
    id_column = "id"
    timestamp_column = "timestamp"
    target = "target"
    horizon = 4
    past_dynamic_columns = ["x1"]
    known_dynamic_columns = ["x2"]


def _make_sequence_dataset() -> tuple[Dataset, Dataset]:
    n = 36
    horizon = 4
    t = np.arange(n, dtype=float)

    y1 = 0.25 * t + np.sin(t / 4.0)
    y2 = 0.15 * t + np.cos(t / 5.0)

    x1_1 = np.cos(t / 7.0)
    x1_2 = np.sin(t / 8.0)
    x2_1 = 0.1 * t + np.sin(t / 6.0)
    x2_2 = -0.05 * t + np.cos(t / 9.0)

    future_t = np.arange(n, n + horizon, dtype=float)
    future_x2_1 = 0.1 * future_t + np.sin(future_t / 6.0)
    future_x2_2 = -0.05 * future_t + np.cos(future_t / 9.0)

    past_data = Dataset.from_dict(
        {
            "id": ["s1", "s2"],
            "timestamp": [
                [f"2000-{(i % 12) + 1:02d}-01" for i in range(n)],
                [f"2000-{(i % 12) + 1:02d}-01" for i in range(n)],
            ],
            "target": [y1.tolist(), y2.tolist()],
            "x1": [x1_1.tolist(), x1_2.tolist()],
            "x2": [x2_1.tolist(), x2_2.tolist()],
        }
    )
    future_data = Dataset.from_dict(
        {
            "id": ["s1", "s2"],
            "timestamp": [
                [f"2003-{(i % 12) + 1:02d}-01" for i in range(horizon)],
                [f"2003-{(i % 12) + 1:02d}-01" for i in range(horizon)],
            ],
            "x2": [future_x2_1.tolist(), future_x2_2.tolist()],
        }
    )

    return past_data, future_data


def test_lstm_variants_emit_finite_predictions() -> None:
    past_data, future_data = _make_sequence_dataset()
    task = _DummyTask()

    models = [
        LSTMUnivariateModel(
            seed=11,
            lookback=8,
            hidden_size=16,
            max_epochs=3,
            patience=2,
            min_train_samples=6,
            batch_size=8,
        ),
        LSTMMultivariateModel(
            seed=11,
            lookback=8,
            hidden_size=16,
            max_epochs=3,
            patience=2,
            min_train_samples=6,
            batch_size=8,
        ),
    ]

    for model in models:
        out = model.predict(past_data=past_data, future_data=future_data, task=task)
        preds = np.asarray(out["predictions"], dtype=float)
        assert out.num_rows == 2
        assert preds.shape == (2, task.horizon)
        assert np.isfinite(preds).all()


def test_lstm_univariate_short_history_fallback_shape() -> None:
    task = _DummyTask()
    past_data = Dataset.from_dict(
        {
            "id": ["s1"],
            "timestamp": [["2000-01-01", "2000-02-01", "2000-03-01"]],
            "target": [[1.0, 1.2, 1.4]],
        }
    )
    future_data = Dataset.from_dict(
        {
            "id": ["s1"],
            "timestamp": [["2000-04-01", "2000-05-01", "2000-06-01", "2000-07-01"]],
        }
    )

    model = LSTMUnivariateModel(
        seed=3,
        lookback=12,
        max_epochs=2,
        patience=1,
        min_train_samples=20,
        batch_size=4,
    )
    out = model.predict(past_data=past_data, future_data=future_data, task=task)
    preds = np.asarray(out["predictions"], dtype=float)
    assert preds.shape == (1, task.horizon)
    assert np.isfinite(preds).all()
