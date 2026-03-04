#!/usr/bin/env python3
"""Sync ALFRED/FRED vintages into a local as-of DuckDB store."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.asof_store import AsofStore  # noqa: E402
from fev_macro.fred_aliases import candidate_series_ids  # noqa: E402

FRED_BASE = "https://api.stlouisfed.org/fred"
USER_AGENT = "fev-macro-alfred-sync/1.0"

FRED_EARLIEST_DATE = "1776-07-04"
FRED_FAR_FUTURE_DATE = "9999-12-31"
FRED_PAGE_LIMIT_MAX = 100_000

DEFAULT_MD_TEMPLATE = "data/historical/md/vintages_1999_2026/2026-01.csv"
DEFAULT_QD_TEMPLATE = "data/historical/qd/vintages_2018_2026/FRED-QD_2026m1.csv"


class FredAPIError(RuntimeError):
    pass


@dataclass
class APIStats:
    total_requests: int = 0
    endpoint_counts: dict[str, int] = field(default_factory=dict)
    retries: int = 0
    http_429: int = 0
    http_5xx: int = 0

    def add(self, endpoint: str) -> None:
        self.total_requests += 1
        self.endpoint_counts[endpoint] = int(self.endpoint_counts.get(endpoint, 0)) + 1


@dataclass
class RateLimiter:
    """Global min-interval limiter to reduce 429 rate limits."""

    min_interval_seconds: float
    _lock: Lock = field(default_factory=Lock)
    _next_ok: float = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            if now < self._next_ok:
                time.sleep(self._next_ok - now)
            self._next_ok = time.time() + self.min_interval_seconds


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key and str(args.api_key).strip():
        return str(args.api_key).strip()

    from_env = str(os.getenv("FRED_API_KEY", "")).strip()
    if from_env:
        return from_env

    env_file = Path(args.env_file).expanduser().resolve()
    parsed = _load_dotenv(env_file)
    from_file = str(parsed.get("FRED_API_KEY", "")).strip()
    if from_file:
        os.environ["FRED_API_KEY"] = from_file
        return from_file

    raise ValueError("Missing FRED_API_KEY (env var or .env file).")


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None and v != ""}


def _request_json(
    *,
    url: str,
    params: dict[str, Any],
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    rate_limiter: RateLimiter,
    stats: APIStats,
) -> dict[str, Any]:
    params = _clean_params(params)
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    headers = {"User-Agent": USER_AGENT}
    endpoint = str(url).replace(FRED_BASE, "")

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        rate_limiter.wait()
        stats.add(endpoint)
        try:
            req = urllib.request.Request(full_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            last_err = e
            status = getattr(e, "code", None)
            if status == 429:
                stats.http_429 += 1
            if status in (500, 502, 503, 504):
                stats.http_5xx += 1
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                stats.retries += 1
                time.sleep(retry_backoff_seconds * (2**attempt))
                continue
            raise FredAPIError(f"HTTP {status} for {full_url}") from e
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                stats.retries += 1
                time.sleep(retry_backoff_seconds * (2**attempt))
                continue
            raise FredAPIError(f"Failed request {full_url}") from e

    raise FredAPIError(f"Request failed: {full_url}") from last_err


def fred_series_meta(
    *,
    series_id: str,
    api_key: str,
    args: argparse.Namespace,
    rate_limiter: RateLimiter,
    stats: APIStats,
) -> dict[str, Any] | None:
    payload = _request_json(
        url=f"{FRED_BASE}/series",
        params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        rate_limiter=rate_limiter,
        stats=stats,
    )
    items = payload.get("seriess", [])
    return items[0] if items else None


def fred_series_vintagedates(
    *,
    series_id: str,
    realtime_start: str,
    realtime_end: str,
    api_key: str,
    args: argparse.Namespace,
    rate_limiter: RateLimiter,
    stats: APIStats,
) -> list[str]:
    payload = _request_json(
        url=f"{FRED_BASE}/series/vintagedates",
        params={
            "series_id": series_id,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "asc",
            "limit": 10000,
            "offset": 0,
        },
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        rate_limiter=rate_limiter,
        stats=stats,
    )
    return [str(x) for x in payload.get("vintage_dates", [])]


def fred_series_observations(
    *,
    series_id: str,
    api_key: str,
    args: argparse.Namespace,
    params: dict[str, Any],
    rate_limiter: RateLimiter,
    stats: APIStats,
) -> dict[str, Any]:
    return _request_json(
        url=f"{FRED_BASE}/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            **params,
        },
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        rate_limiter=rate_limiter,
        stats=stats,
    )


def resolve_variable_to_series_id(
    *,
    variable_name: str,
    api_key: str,
    args: argparse.Namespace,
    rate_limiter: RateLimiter,
    stats: APIStats,
) -> str | None:
    for cand in candidate_series_ids(variable_name):
        try:
            meta = fred_series_meta(
                series_id=cand,
                api_key=api_key,
                args=args,
                rate_limiter=rate_limiter,
                stats=stats,
            )
        except FredAPIError:
            continue
        if meta is not None:
            return cand
    return None


def read_template_variables(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    if len(cols) <= 1:
        return []
    return [c for c in cols[1:] if isinstance(c, str) and c.strip() and not c.startswith("Unnamed")]


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s == "." or s.upper() in {"NA", "NAN"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def backfill_series_output_type_1(
    *,
    store: AsofStore,
    series_id: str,
    api_key: str,
    args: argparse.Namespace,
    rate_limiter: RateLimiter,
    stats: APIStats,
) -> None:
    page_limit = min(int(args.page_limit), FRED_PAGE_LIMIT_MAX)
    try:
        offset = 0
        while True:
            payload = fred_series_observations(
                series_id=series_id,
                api_key=api_key,
                args=args,
                rate_limiter=rate_limiter,
                stats=stats,
                params={
                    "output_type": 1,
                    "realtime_start": args.backfill_realtime_start,
                    "realtime_end": args.backfill_realtime_end,
                    "observation_start": args.observation_start,
                    "observation_end": args.observation_end,
                    "sort_order": "asc",
                    "limit": page_limit,
                    "offset": offset,
                },
            )
            obs = payload.get("observations", [])
            if not obs:
                break

            rows: list[dict[str, Any]] = []
            for o in obs:
                v = _parse_float(o.get("value"))
                if v is None:
                    continue
                rows.append(
                    {
                        "series_id": series_id,
                        "obs_ts": o.get("date"),
                        "asof_ts": o.get("realtime_start"),
                        "value": v,
                    }
                )
            if rows:
                store.ingest_versions(pd.DataFrame(rows), source="alfred_output_type_1")

            offset += page_limit
            if len(obs) < page_limit:
                break
    except FredAPIError as exc:
        # Some series are available in FRED but not as full ALFRED vintage histories.
        # Fall back to point-in-time availability using obs date as as-of timestamp.
        if "HTTP 400" not in str(exc):
            raise
        offset = 0
        while True:
            payload = fred_series_observations(
                series_id=series_id,
                api_key=api_key,
                args=args,
                rate_limiter=rate_limiter,
                stats=stats,
                params={
                    "observation_start": args.observation_start,
                    "observation_end": args.observation_end,
                    "sort_order": "asc",
                    "limit": page_limit,
                    "offset": offset,
                },
            )
            obs = payload.get("observations", [])
            if not obs:
                break
            rows: list[dict[str, Any]] = []
            for o in obs:
                v = _parse_float(o.get("value"))
                if v is None:
                    continue
                obs_date = o.get("date")
                rows.append(
                    {
                        "series_id": series_id,
                        "obs_ts": obs_date,
                        "asof_ts": obs_date,
                        "value": v,
                    }
                )
            if rows:
                store.ingest_versions(pd.DataFrame(rows), source="fred_observations_fallback")

            offset += page_limit
            if len(obs) < page_limit:
                break


def _infer_vintage_value_key(series_id: str, vintage_date: str, observations: list[dict[str, Any]]) -> str | None:
    if not observations:
        return None
    suffix = pd.to_datetime(vintage_date).strftime("%Y%m%d")
    sample = observations[0]
    direct = f"{series_id}_{suffix}"
    if direct in sample:
        return direct
    keys = [k for k in sample.keys() if k != "date" and isinstance(k, str)]
    candidates = [k for k in keys if k.endswith(suffix)]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        for k in candidates:
            if series_id in k:
                return k
        return candidates[0]
    return None


def update_series_from_vintages_output_type_3(
    *,
    store: AsofStore,
    series_id: str,
    api_key: str,
    args: argparse.Namespace,
    rate_limiter: RateLimiter,
    stats: APIStats,
) -> int:
    last = store.max_asof_ts(series_id)
    if last is None:
        return 0

    start_dt = (last - pd.Timedelta(days=int(args.lookback_days))).date()
    try:
        vintages = fred_series_vintagedates(
            series_id=series_id,
            realtime_start=start_dt.isoformat(),
            realtime_end=FRED_FAR_FUTURE_DATE,
            api_key=api_key,
            args=args,
            rate_limiter=rate_limiter,
            stats=stats,
        )
    except FredAPIError as exc:
        if "HTTP 400" in str(exc):
            return 0
        raise

    cutoff_date = last.date()
    new_vintages = [d for d in vintages if pd.to_datetime(d).date() > cutoff_date]
    if not new_vintages:
        return 0

    inserted_attempts = 0
    for vd in new_vintages:
        payload = fred_series_observations(
            series_id=series_id,
            api_key=api_key,
            args=args,
            rate_limiter=rate_limiter,
            stats=stats,
            params={
                "output_type": 3,
                "vintage_dates": vd,
                "observation_start": args.observation_start,
                "observation_end": args.observation_end,
                "sort_order": "asc",
                "limit": FRED_PAGE_LIMIT_MAX,
            },
        )
        obs = payload.get("observations", [])
        if not obs:
            continue
        val_key = _infer_vintage_value_key(series_id, vd, obs)
        if not val_key:
            continue

        rows: list[dict[str, Any]] = []
        for o in obs:
            v = _parse_float(o.get(val_key))
            if v is None:
                continue
            rows.append(
                {
                    "series_id": series_id,
                    "obs_ts": o.get("date"),
                    "asof_ts": vd,
                    "value": v,
                }
            )
        if rows:
            inserted_attempts += store.ingest_versions(
                pd.DataFrame(rows),
                source=f"alfred_output_type_3:{vd}",
            )
    return inserted_attempts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync ALFRED vintages into an as-of DuckDB store for time-travel queries."
    )
    p.add_argument("--env_file", type=str, default=".env")
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--db", type=str, default="data/realtime/asof.duckdb")

    p.add_argument("--md_template", type=str, default=DEFAULT_MD_TEMPLATE)
    p.add_argument("--qd_template", type=str, default=DEFAULT_QD_TEMPLATE)
    p.add_argument("--universe", type=str, choices=["md", "qd", "both"], default="both")

    p.add_argument("--series_limit", type=int, default=None, help="Debug cap on number of template variables.")
    p.add_argument("--backfill_missing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--observation_start", type=str, default=None)
    p.add_argument("--observation_end", type=str, default=None)
    p.add_argument("--backfill_realtime_start", type=str, default=FRED_EARLIEST_DATE)
    p.add_argument("--backfill_realtime_end", type=str, default=FRED_FAR_FUTURE_DATE)
    p.add_argument("--lookback_days", type=int, default=7)
    p.add_argument("--page_limit", type=int, default=100_000)

    p.add_argument("--timeout_seconds", type=float, default=30.0)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--retry_backoff_seconds", type=float, default=1.5)
    p.add_argument("--min_request_interval_seconds", type=float, default=0.55)

    p.add_argument("--report_json", type=str, default="data/realtime/asof_sync_report.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = resolve_api_key(args)
    stats = APIStats()
    started = time.time()

    store = AsofStore(Path(args.db).expanduser().resolve())
    rate_limiter = RateLimiter(min_interval_seconds=float(args.min_request_interval_seconds))

    variables: list[tuple[str, str]] = []
    if args.universe in ("md", "both"):
        variables.extend([("md", v) for v in read_template_variables(Path(args.md_template))])
    if args.universe in ("qd", "both"):
        variables.extend([("qd", v) for v in read_template_variables(Path(args.qd_template))])
    if args.series_limit is not None:
        variables = variables[: int(args.series_limit)]

    failures: dict[str, str] = {}
    resolved_vars = 0
    series_ids: list[str] = []
    seen_series: set[str] = set()
    existing_aliases: dict[str, dict[str, str]] = {
        "md": store.alias_map(universe="md"),
        "qd": store.alias_map(universe="qd"),
    }
    checkpoint_cache: dict[str, pd.Timestamp | None] = {}
    alias_cache_hits = 0
    alias_api_resolves = 0

    for uni, var in variables:
        try:
            sid = existing_aliases.get(uni, {}).get(var)
            if sid:
                alias_cache_hits += 1
            else:
                sid = resolve_variable_to_series_id(
                    variable_name=var,
                    api_key=api_key,
                    args=args,
                    rate_limiter=rate_limiter,
                    stats=stats,
                )
                if sid:
                    alias_api_resolves += 1
            if not sid:
                failures[f"{uni}:{var}"] = "unresolved"
                continue

            resolved_vars += 1
            store.upsert_alias(variable_name=var, universe=uni, series_id=sid)

            if sid not in seen_series:
                seen_series.add(sid)
                series_ids.append(sid)
                checkpoint = store.max_asof_ts(sid)
                checkpoint_cache[sid] = checkpoint
                if checkpoint is None:
                    meta = fred_series_meta(
                        series_id=sid,
                        api_key=api_key,
                        args=args,
                        rate_limiter=rate_limiter,
                        stats=stats,
                    )
                    if meta:
                        store.upsert_series_meta(series_id=sid, meta=meta)
        except Exception as exc:  # noqa: BLE001
            failures[f"{uni}:{var}"] = f"{type(exc).__name__}: {exc}"

    for sid in series_ids:
        try:
            checkpoint = checkpoint_cache.get(sid)
            if sid not in checkpoint_cache:
                checkpoint = store.max_asof_ts(sid)
                checkpoint_cache[sid] = checkpoint
            if checkpoint is None and args.backfill_missing:
                print(f"[backfill] {sid}")
                backfill_series_output_type_1(
                    store=store,
                    series_id=sid,
                    api_key=api_key,
                    args=args,
                    rate_limiter=rate_limiter,
                    stats=stats,
                )
                checkpoint = store.max_asof_ts(sid)
                checkpoint_cache[sid] = checkpoint

            if checkpoint is not None:
                print(f"[update] {sid} (checkpoint={checkpoint.date().isoformat()})")
                update_series_from_vintages_output_type_3(
                    store=store,
                    series_id=sid,
                    api_key=api_key,
                    args=args,
                    rate_limiter=rate_limiter,
                    stats=stats,
                )
        except Exception as exc:  # noqa: BLE001
            failures[f"series:{sid}"] = f"sync_failed: {type(exc).__name__}: {exc}"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db": str(Path(args.db).expanduser().resolve()),
        "universe": args.universe,
        "resolved_variables": resolved_vars,
        "resolved_unique_series": len(series_ids),
        "failures_count": len(failures),
        "failures": failures,
        "alias_resolution": {
            "alias_cache_hits": int(alias_cache_hits),
            "alias_api_resolves": int(alias_api_resolves),
        },
        "api_stats": {
            "total_requests": int(stats.total_requests),
            "endpoint_counts": dict(sorted(stats.endpoint_counts.items())),
            "retries": int(stats.retries),
            "http_429": int(stats.http_429),
            "http_5xx": int(stats.http_5xx),
            "min_request_interval_seconds": float(args.min_request_interval_seconds),
            "effective_min_elapsed_seconds": float(stats.total_requests) * float(args.min_request_interval_seconds),
        },
        "runtime_seconds": float(time.time() - started),
    }
    report_path = Path(args.report_json).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote report: {report_path}")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
