from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.data import exclude_years
from fev_macro.eval_runner import build_eval_arg_parser


def test_eval_parser_does_not_exclude_2020_by_default() -> None:
    parser = build_eval_arg_parser(
        description="test",
        default_target_transform="log_level",
        default_results_dir="results/test",
    )
    args = parser.parse_args([])
    assert args.exclude_years == []


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
