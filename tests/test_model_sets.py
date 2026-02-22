from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.model_sets import resolve_model_names


def test_auto_model_set_unprocessed_uses_ll_defaults() -> None:
    names = resolve_model_names(None, covariate_mode="unprocessed", model_set="auto")
    assert "drift" in names
    assert "auto_ets" in names
    assert "atlantafed_gdpnow" in names
    assert "bvar_minnesota_8" in names


def test_auto_model_set_processed_uses_growth_defaults() -> None:
    names = resolve_model_names(None, covariate_mode="processed", model_set="auto")
    assert "bvar_minnesota_growth_8" in names
    assert "bvar_minnesota_growth_20" in names
    assert "atlantafed_gdpnow" in names
    assert "auto_arima" in names
