from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.asof_store import AsofStore


def test_snapshot_long_uses_latest_version_at_or_before_cutoff(tmp_path) -> None:
    import pytest

    pytest.importorskip("duckdb")
    db_path = tmp_path / "asof.duckdb"
    store = AsofStore(db_path=db_path)
    try:
        versions = pd.DataFrame(
            [
                {"series_id": "GDPC1", "obs_ts": "2020-01-01", "asof_ts": "2020-02-01", "value": 100.0},
                {"series_id": "GDPC1", "obs_ts": "2020-01-01", "asof_ts": "2020-03-01", "value": 101.0},
                {"series_id": "GDPC1", "obs_ts": "2020-04-01", "asof_ts": "2020-05-01", "value": 102.0},
            ]
        )
        store.ingest_versions(versions, source="unit_test")

        early = store.snapshot_long(
            asof_ts="2020-02-15",
            series_ids=["GDPC1"],
            include_asof_used=True,
        )
        assert len(early) == 1
        assert float(early.loc[0, "value"]) == 100.0
        assert pd.Timestamp(early.loc[0, "asof_used"]) == pd.Timestamp("2020-02-01")

        late = store.snapshot_long(
            asof_ts="2020-03-15",
            series_ids=["GDPC1"],
            include_asof_used=True,
        )
        assert len(late) == 1
        assert float(late.loc[0, "value"]) == 101.0
        assert pd.Timestamp(late.loc[0, "asof_used"]) == pd.Timestamp("2020-03-01")
    finally:
        store.close()


def test_ingest_versions_ignores_duplicate_primary_keys(tmp_path) -> None:
    import pytest

    pytest.importorskip("duckdb")
    db_path = tmp_path / "asof.duckdb"
    store = AsofStore(db_path=db_path)
    try:
        batch = pd.DataFrame(
            [
                {"series_id": "UNRATE", "obs_ts": "2021-01-01", "asof_ts": "2021-02-01", "value": 6.0},
                {"series_id": "UNRATE", "obs_ts": "2021-01-01", "asof_ts": "2021-02-01", "value": 6.0},
            ]
        )
        attempted = store.ingest_versions(batch, source="unit_test")
        assert attempted == 1  # pre-deduped within batch

        attempted_again = store.ingest_versions(batch, source="unit_test")
        assert attempted_again == 1  # attempted row count still reflects pre-deduped input

        snap = store.snapshot_long(asof_ts="2021-03-01", series_ids=["UNRATE"])
        assert len(snap) == 1
        assert float(snap.loc[0, "value"]) == 6.0
        assert store.max_asof_ts("UNRATE") == pd.Timestamp("2021-02-01")
    finally:
        store.close()
