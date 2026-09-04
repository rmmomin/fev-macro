#!/usr/bin/env python3
"""Refresh small live API fixtures explicitly; ordinary tests never use the network."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sync_alfred_asof_store import APIStats, RateLimiter, fred_series_observations, resolve_api_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/fixtures/alfred")
    parser.add_argument("--env_file", default=".env")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()
    args.timeout_seconds, args.max_retries, args.retry_backoff_seconds = 30, 2, 1
    key = resolve_api_key(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queries = {
        "gdpc1_intervals": dict(series_id="GDPC1", observation_start="2005-01-01", observation_end="2019-10-01",
                                realtime_start="2018-01-01", realtime_end="2020-04-30"),
        "unrate_intervals": dict(series_id="UNRATE", observation_start="2005-01-01", observation_end="2019-12-01",
                                 realtime_start="2018-01-01", realtime_end="2020-04-30"),
        "gdp_intervals": dict(series_id="GDP", observation_start="2018-10-01", observation_end="2019-01-01",
                              realtime_start="1776-07-04", realtime_end="2019-06-30"),
        "cpiaucsl_intervals": dict(series_id="CPIAUCSL", observation_start="2019-01-01", observation_end="2019-01-01",
                                   realtime_start="1776-07-04", realtime_end="2019-03-31"),
        "gdpc1_before_advance": dict(series_id="GDPC1", observation_start="2019-01-01", observation_end="2019-01-01",
                                    realtime_start="2019-04-25", realtime_end="2019-04-25"),
        "gdpc1_on_advance": dict(series_id="GDPC1", observation_start="2018-10-01", observation_end="2019-01-01",
                                realtime_start="2019-04-26", realtime_end="2019-04-26"),
    }
    for name, query in queries.items():
        params = dict(query, output_type=1, units="lin", limit=100000, offset=0, sort_order="asc")
        payload = fred_series_observations(series_id=query["series_id"], api_key=key, args=args,
                                           params=params, rate_limiter=RateLimiter(.55), stats=APIStats())
        if payload["count"] != len(payload["observations"]):
            raise ValueError("Fixture query exceeded one page")
        record = dict(endpoint="https://api.stlouisfed.org/fred/series/observations", params=params,
                      retrieved_at=datetime.now(timezone.utc).isoformat(), response=payload)
        (out / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        print(name, len(payload["observations"]))


if __name__ == "__main__":
    main()
