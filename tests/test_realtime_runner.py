from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.realtime_runner import (  # noqa: E402
    normalize_mode,
    resolve_baseline_model,
    resolve_md_panel_path,
    resolve_models,
    resolve_qd_panel_path,
)


def test_resolve_models_defaults_by_mode() -> None:
    unprocessed = resolve_models(mode="unprocessed", requested_models=None)
    processed = resolve_models(mode="processed", requested_models=None)

    assert "rw_drift_log" in unprocessed
    assert "ensemble_avg_top3" in unprocessed

    assert "naive_last_growth" in processed
    assert "auto_arima_growth" in processed
    assert "ensemble_avg_top3_processed_g" in processed


def test_resolve_baseline_model_defaults_by_mode() -> None:
    assert resolve_baseline_model(mode="unprocessed", explicit_baseline=None) == "rw_drift_log"
    assert resolve_baseline_model(mode="processed", explicit_baseline=None) == "naive_last_growth"


def test_resolve_panel_paths_follow_mode() -> None:
    assert resolve_qd_panel_path(mode="unprocessed").as_posix().endswith("data/panels/fred_qd_vintage_panel.parquet")
    assert resolve_qd_panel_path(mode="processed").as_posix().endswith("data/panels/fred_qd_vintage_panel_processed.parquet")
    assert resolve_md_panel_path(mode="unprocessed").as_posix().endswith("data/panels/fred_md_vintage_panel.parquet")
    assert resolve_md_panel_path(mode="processed").as_posix().endswith("data/panels/fred_md_vintage_panel_processed.parquet")


def test_normalize_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        normalize_mode("unknown")
