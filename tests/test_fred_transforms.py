from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.fred_transforms import extract_fred_transform_codes


def test_extract_transform_codes_handles_trailing_colon() -> None:
    raw = pd.DataFrame(
        {
            "sasdate": ["Transform:", "1/31/2020", "2/29/2020"],
            "A": ["5", "100.0", "101.0"],
            "B": ["2", "3.0", "4.0"],
            "C": ["999", "1.0", "2.0"],
        }
    )

    codes = extract_fred_transform_codes(raw_df=raw, first_col_name="sasdate")
    assert codes == {"A": 5, "B": 2}
