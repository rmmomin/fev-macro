"""As-of (bitemporal) storage for realtime macro forecasting evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json

import pandas as pd

from .pit import PITError, alfred_rows, content_hash, information_date


def _naive_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _normalize_datetime_series(values: pd.Series) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    try:
        tz = out.dt.tz
    except Exception:
        tz = None
    if tz is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


@dataclass
class AsofStore:
    """Versioned DuckDB observations with retained API responses and PIT checks."""

    db_path: str | Path
    strict_pit: bool = True

    def __post_init__(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "duckdb is required for AsofStore. Install with: pip install duckdb"
            ) from exc

        self.db_path = Path(self.db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.db_path))
        self._ensure_schema()

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS asof_observations (
              series_id   VARCHAR NOT NULL,
              obs_ts      TIMESTAMP NOT NULL,
              asof_ts     TIMESTAMP NOT NULL,
              value       DOUBLE,
              ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              source      VARCHAR,
              PRIMARY KEY (series_id, obs_ts, asof_ts)
            );
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS asof_series_state (
              series_id    VARCHAR PRIMARY KEY,
              max_asof_ts  TIMESTAMP,
              updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS asof_series_aliases (
              variable_name VARCHAR NOT NULL,
              universe      VARCHAR NOT NULL,
              series_id     VARCHAR NOT NULL,
              updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (variable_name, universe)
            );
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS asof_series_meta (
              series_id                 VARCHAR PRIMARY KEY,
              title                     VARCHAR,
              frequency_short           VARCHAR,
              units                     VARCHAR,
              seasonal_adjustment_short VARCHAR,
              observation_start         DATE,
              observation_end           DATE,
              last_updated              VARCHAR,
              updated_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_asof_lookup
            ON asof_observations(series_id, obs_ts, asof_ts);
            """
        )

        self._con.execute("ALTER TABLE asof_observations ADD COLUMN IF NOT EXISTS realtime_end VARCHAR")
        self._con.execute("ALTER TABLE asof_observations ADD COLUMN IF NOT EXISTS provenance_id VARCHAR")
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS asof_api_responses (
                provenance_id VARCHAR PRIMARY KEY, request_json VARCHAR NOT NULL,
                response_json VARCHAR NOT NULL, retrieved_at VARCHAR NOT NULL, endpoint VARCHAR NOT NULL
            )
        """)

    def ingest_alfred_response(self, payload: dict, params: dict, *, retrieved_at: str | None = None) -> int:
        """Keep sanitized request and exact JSON content together with its versions.

        Callers syncing multiple pages must wrap all pages in one transaction.
        Credentials are never retained. Only validated API interval rows receive
        the source tag accepted by strict queries.
        """
        safe = {k: v for k, v in params.items() if k != "api_key"}
        rows = alfred_rows(payload, safe)
        provenance_id = content_hash({"params": safe, "response": payload})
        self._con.execute(
            "INSERT INTO asof_api_responses VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [provenance_id, json.dumps(safe, sort_keys=True), json.dumps(payload, sort_keys=True),
             retrieved_at or pd.Timestamp.now(tz="UTC").isoformat(),
             "https://api.stlouisfed.org/fred/series/observations"],
        )
        rows["provenance_id"] = provenance_id
        return self.ingest_versions(rows, source="alfred_api_v2")

    def ingest_versions(self, df_versions: pd.DataFrame, *, source: str | None = None) -> int:
        """Insert versions; replay interval bounds, reject conflicting values, count inserts."""
        if df_versions is None or df_versions.empty:
            return 0

        required = {"series_id", "obs_ts", "asof_ts", "value"}
        missing = sorted(required.difference(df_versions.columns))
        if missing:
            raise ValueError(f"df_versions missing required columns: {missing}")

        work = df_versions.copy()
        if work["series_id"].isna().any():
            raise PITError("Missing series identifier")
        work["series_id"] = work["series_id"].astype(str)
        work["obs_ts"] = _normalize_datetime_series(work["obs_ts"])
        work["asof_ts"] = _normalize_datetime_series(work["asof_ts"])
        work["value"] = pd.to_numeric(work["value"], errors="raise")
        if work[["obs_ts", "asof_ts"]].isna().any().any():
            raise PITError("Invalid observation or availability timestamp")
        if work["value"].isin([float("inf"), -float("inf")]).any():
            raise PITError("Infinite observations are invalid")
        work["source"] = source
        for col in ("realtime_end", "provenance_id"):
            if col not in work:
                work[col] = None
        keys = ["series_id", "obs_ts", "asof_ts"]
        if (work.groupby(keys, dropna=False)["value"].nunique(dropna=False) > 1).any():
            raise PITError("Conflicting values for the same version")
        work = work.drop_duplicates(keys)
        cols = [*keys, "value", "source", "realtime_end", "provenance_id"]
        self._con.register("incoming_versions", work[cols])
        try:
            conflict = self._con.execute("""
                SELECT COUNT(*) FROM incoming_versions i JOIN asof_observations o
                USING(series_id, obs_ts, asof_ts)
                WHERE i.value IS DISTINCT FROM o.value OR i.source IS DISTINCT FROM o.source
            """).fetchone()[0]
            if conflict:
                raise PITError("Conflicting stored version; rebuild legacy databases into a new file")
            before = self._con.execute("SELECT COUNT(*) FROM asof_observations").fetchone()[0]
            self._con.execute("""
                INSERT INTO asof_observations(series_id, obs_ts, asof_ts, value, source, realtime_end, provenance_id)
                SELECT series_id, obs_ts, asof_ts, value, source, realtime_end, provenance_id FROM incoming_versions
                ON CONFLICT(series_id, obs_ts, asof_ts) DO UPDATE SET
                    realtime_end = excluded.realtime_end, provenance_id = excluded.provenance_id
            """)
            self._con.execute("""
                INSERT INTO asof_series_state(series_id, max_asof_ts, updated_at)
                SELECT series_id, MAX(asof_ts), now() FROM incoming_versions GROUP BY series_id
                ON CONFLICT(series_id) DO UPDATE SET
                    max_asof_ts = GREATEST(asof_series_state.max_asof_ts, excluded.max_asof_ts),
                    updated_at = now()
            """)
            return int(self._con.execute("SELECT COUNT(*) FROM asof_observations").fetchone()[0] - before)
        finally:
            self._con.unregister("incoming_versions")

    def upsert_alias(self, *, variable_name: str, universe: str, series_id: str) -> None:
        self._con.execute(
            """
            INSERT INTO asof_series_aliases(variable_name, universe, series_id, updated_at)
            VALUES (?, ?, ?, now())
            ON CONFLICT(variable_name, universe)
            DO UPDATE SET series_id = excluded.series_id,
                          updated_at = now();
            """,
            [str(variable_name), str(universe), str(series_id)],
        )

    def alias_map(self, *, universe: str | None = None) -> dict[str, str]:
        if universe is None:
            rows = self._con.execute(
                """
                SELECT variable_name, series_id
                FROM asof_series_aliases
                ORDER BY updated_at DESC;
                """
            ).fetchall()
            out: dict[str, str] = {}
            for variable_name, series_id in rows:
                if variable_name not in out:
                    out[str(variable_name)] = str(series_id)
            return out

        rows = self._con.execute(
            """
            SELECT variable_name, series_id
            FROM asof_series_aliases
            WHERE universe = ?;
            """,
            [str(universe)],
        ).fetchall()
        return {str(variable_name): str(series_id) for variable_name, series_id in rows}

    def upsert_series_meta(self, *, series_id: str, meta: dict) -> None:
        self._con.execute(
            """
            INSERT INTO asof_series_meta(
              series_id, title, frequency_short, units, seasonal_adjustment_short,
              observation_start, observation_end, last_updated, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, now())
            ON CONFLICT(series_id)
            DO UPDATE SET
              title = excluded.title,
              frequency_short = excluded.frequency_short,
              units = excluded.units,
              seasonal_adjustment_short = excluded.seasonal_adjustment_short,
              observation_start = excluded.observation_start,
              observation_end = excluded.observation_end,
              last_updated = excluded.last_updated,
              updated_at = now();
            """,
            [
                str(series_id),
                meta.get("title"),
                meta.get("frequency_short"),
                meta.get("units"),
                meta.get("seasonal_adjustment_short"),
                meta.get("observation_start"),
                meta.get("observation_end"),
                meta.get("last_updated"),
            ],
        )

    def max_asof_ts(self, series_id: str) -> pd.Timestamp | None:
        row = self._con.execute(
            "SELECT max_asof_ts FROM asof_series_state WHERE series_id = ?;",
            [str(series_id)],
        ).fetchone()
        if not row or row[0] is None:
            return None
        return pd.Timestamp(row[0])

    def available_series_ids(self) -> set[str]:
        rows = self._con.execute(
            """
            SELECT series_id
            FROM asof_series_state
            WHERE max_asof_ts IS NOT NULL;
            """
        ).fetchall()
        return {str(row[0]) for row in rows}

    def snapshot_long(
        self,
        *,
        asof_ts: object,
        series_ids: Iterable[str],
        obs_start: object | None = None,
        obs_end: object | None = None,
        include_asof_used: bool = False,
    ) -> pd.DataFrame:
        series_list = [str(s) for s in series_ids]
        if not series_list:
            cols = ["series_id", "obs_ts", "value"]
            if include_asof_used:
                cols.append("asof_used")
            return pd.DataFrame(columns=cols)

        cutoff = information_date(asof_ts) if self.strict_pit else _naive_timestamp(asof_ts)
        where = ["o.asof_ts <= ?", "o.obs_ts <= ?"]
        params: list[object] = [cutoff, cutoff]
        if obs_start is not None:
            where.append("o.obs_ts >= ?")
            params.append(_naive_timestamp(obs_start))
        if obs_end is not None:
            where.append("o.obs_ts <= ?")
            params.append(_naive_timestamp(obs_end))
        self._con.register("series_filter", pd.DataFrame({"series_id": sorted(set(series_list))}))
        try:
            if self.strict_pit:
                bad = self._con.execute("""
                    SELECT DISTINCT o.series_id FROM asof_observations o
                    JOIN series_filter f USING(series_id)
                    LEFT JOIN asof_api_responses p USING(provenance_id)
                    WHERE o.source IS DISTINCT FROM 'alfred_api_v2'
                       OR o.realtime_end IS NULL OR p.provenance_id IS NULL
                """).fetchall()
                if bad:
                    raise PITError(f"Unverified/legacy series in strict PIT store: {bad}; re-sync into a new database")
            # Select the newest ROW, including NULL withdrawals. ARG_MAX(value, date)
            # skips NULL and can silently resurrect a superseded observation.
            extra = ", asof_ts AS asof_used, realtime_end, source, provenance_id" if include_asof_used else ""
            end_filter = "WHERE realtime_end >= ?" if self.strict_pit else ""
            if self.strict_pit:
                params.append(cutoff.date().isoformat())
            out = self._con.execute(f"""
                WITH ranked AS (
                    SELECT o.*, ROW_NUMBER() OVER (
                        PARTITION BY o.series_id, o.obs_ts ORDER BY o.asof_ts DESC
                    ) AS rn
                    FROM asof_observations o JOIN series_filter f USING(series_id)
                    WHERE {' AND '.join(where)}
                ), latest AS (SELECT * FROM ranked WHERE rn = 1)
                SELECT series_id, obs_ts, value {extra} FROM latest
                {end_filter} ORDER BY obs_ts, series_id
            """, params).df()
        finally:
            self._con.unregister("series_filter")
        return out

    def snapshot_wide(
        self,
        *,
        asof_ts: object,
        series_ids: Iterable[str],
        obs_start: object | None = None,
        obs_end: object | None = None,
        timestamp_name: str = "timestamp",
    ) -> pd.DataFrame:
        long = self.snapshot_long(
            asof_ts=asof_ts,
            series_ids=series_ids,
            obs_start=obs_start,
            obs_end=obs_end,
            include_asof_used=False,
        )
        if long.empty:
            return pd.DataFrame(columns=[timestamp_name])

        wide = long.pivot(index="obs_ts", columns="series_id", values="value").sort_index()
        wide = wide.reset_index().rename(columns={"obs_ts": timestamp_name})
        wide.columns.name = None
        return wide
