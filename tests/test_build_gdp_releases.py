from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_gdp_releases import build_release_dataset


def _saar(curr: float, prev: float) -> float:
    return float(100.0 * ((curr / prev) ** 4 - 1.0))


def test_build_release_dataset_realtime_growth_uses_previous_level_asof_current_release() -> None:
    wide = pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-04-01"],
            "GDPC1_20240425": [100.0, np.nan],
            "GDPC1_20240530": [101.0, 110.0],
            "GDPC1_20240627": [102.0, 111.0],
            "GDPC1_20240725": [103.0, 112.0],
        }
    )

    out = build_release_dataset(wide=wide, series="GDPC1")

    # First row has no previous quarter for q/q growth.
    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_first_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_second_pct"])
    assert np.isnan(out.loc[0, "qoq_saar_growth_realtime_third_pct"])

    # For 2024Q2, previous-quarter levels are taken as-of each release vintage:
    # first uses 2024-05-30 vintage previous level=101
    # second uses 2024-06-27 vintage previous level=102
    # third uses 2024-07-25 vintage previous level=103
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_first_pct"], _saar(110.0, 101.0))
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_second_pct"], _saar(111.0, 102.0))
    assert np.isclose(out.loc[1, "qoq_saar_growth_realtime_third_pct"], _saar(112.0, 103.0))
