from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import exclude_years
from fev_macro.eval_runner import (
    _apply_profile_defaults,
    _resolve_eval_release_metric,
    _resolve_eval_release_stages,
    _select_covariate_columns,
    build_eval_arg_parser,
    parse_args_with_provenance,
)


def test_eval_parser_does_not_exclude_2020_by_default() -> None:
    parser = build_eval_arg_parser(
        description="test",
        default_target_transform="log_level",
        default_results_dir="results/test",
    )
    args = parser.parse_args([])
    assert args.exclude_years == []


def test_vintage_fallback_defaults_to_strict_mode() -> None:
    parser = build_eval_arg_parser(
        description="test",
        default_target_transform="log_level",
        default_results_dir="results/test",
    )
    args = parser.parse_args([])
    assert bool(args.vintage_fallback_to_earliest) is False


def test_exclude_years_none_keeps_2020_rows() -> None:
    df = pd.DataFrame(
        {
            "item_id": ["gdp", "gdp", "gdp"],
            "timestamp": pd.to_datetime(["2019-10-01", "2020-01-01", "2021-01-01"]),
            "target": [1.0, 2.0, 3.0],
        }
    )

    out = exclude_years(df, years=None)
    years = pd.to_datetime(out["timestamp"]).dt.year.tolist()
    assert 2020 in years


def test_smoke_profile_defaults_match_contract() -> None:
    parser = build_eval_arg_parser(
        description="test",
        default_target_transform="saar_growth",
        default_results_dir="results/test",
    )
    args = parse_args_with_provenance(parser, ["--profile", "smoke"])
    _apply_profile_defaults(args, covariate_mode="processed")

    assert args.horizons == [1, 4]
    assert args.num_windows == 10
    assert args.models == ["naive_last", "drift", "auto_arima"]


def test_standard_profile_defaults_match_contract() -> None:
    parser = build_eval_arg_parser(
        description="test",
        default_target_transform="saar_growth",
        default_results_dir="results/test",
    )

    args_unprocessed = parse_args_with_provenance(parser, ["--profile", "standard"])
    _apply_profile_defaults(args_unprocessed, covariate_mode="unprocessed")
    assert args_unprocessed.horizons == [1, 2, 4]
    assert args_unprocessed.num_windows == 60
    assert "chronos2" not in args_unprocessed.models
    assert "ensemble_avg_top3" not in args_unprocessed.models
    assert "ensemble_weighted_top5" not in args_unprocessed.models

    args_processed = parse_args_with_provenance(parser, ["--profile", "standard"])
    _apply_profile_defaults(args_processed, covariate_mode="processed")
    assert args_processed.horizons == [1, 2, 4]
    assert args_processed.num_windows == 40
    assert args_processed.target_transform == "saar_growth"
    assert "chronos2" not in args_processed.models
    assert "ensemble_avg_top3" not in args_processed.models
    assert "ensemble_weighted_top5" not in args_processed.models


def test_auto_release_metric_defaults_to_alfred_qoq_saar_for_saar_growth() -> None:
    assert _resolve_eval_release_metric("auto", "saar_growth") == "alfred_qoq_saar"


def test_alfred_qoq_saar_default_stage_is_first() -> None:
    stages = _resolve_eval_release_stages(
        eval_release_metric="alfred_qoq_saar",
        eval_release_stage=None,
        eval_release_stages=None,
    )
    assert stages == ["first"]


def test_select_covariates_keeps_required_columns_under_cap() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["gdp"] * 10,
            "timestamp": pd.date_range("2018-01-01", periods=10, freq="QS-DEC"),
            "target": [1.0] * 10,
            "AAA": range(10),
            "BBB": range(10),
            "covid_dummy_2020q2": [0.0] * 10,
            "covid_dummy_2020q3": [0.0] * 10,
        }
    )
    cols = _select_covariate_columns(
        frame=frame,
        target_col="target",
        max_covariates=2,
        required_columns=["covid_dummy_2020q2", "covid_dummy_2020q3"],
    )
    assert cols == ["covid_dummy_2020q2", "covid_dummy_2020q3"]
