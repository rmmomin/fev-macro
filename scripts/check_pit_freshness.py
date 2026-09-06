#!/usr/bin/env python3
"""Verify ALFRED snapshot equality without mutating the store or inventing releases."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fev_macro.asof_store import AsofStore
from fev_macro.pit_freshness import check_freshness
from sync_alfred_asof_store import resolve_api_key, RateLimiter, APIStats, fred_series_observations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', required=True)
    p.add_argument('--series-specs', required=True)
    p.add_argument('--origin', required=True)
    p.add_argument('--observation-start', required=True)
    p.add_argument('--env-file', default='.env')
    p.add_argument('--out', required=True, help='New output directory for report and raw API evidence')
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        p.error('Use a new output directory')
    if not Path(args.db).is_file():
        p.error('Verified ALFRED database does not exist')
    args.api_key = None
    args.timeout_seconds, args.max_retries, args.retry_backoff_seconds = 30, 4, 1.5
    key = resolve_api_key(args)
    limiter, stats = RateLimiter(.55), APIStats()
    def fetch(params):
        return fred_series_observations(series_id=params['series_id'], api_key=key, args=args,
            params=params, rate_limiter=limiter, stats=stats)
    store = AsofStore(args.db)
    try:
        report, evidence = check_freshness(store, origin=args.origin,
            series_specs=json.loads(Path(args.series_specs).read_text()),
            observation_start=args.observation_start, fetch=fetch,
            retrieved_at=datetime.now(timezone.utc).isoformat())
    finally:
        store.close()
    report['api_requests'] = stats.total_requests
    out.mkdir(parents=True)
    for name, value in [('freshness.json', report), ('api_responses.json', evidence)]:
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for r in report['series']:
        print(f"{r['series_id']:20} {r['status']:10} latest={r.get('alfred_latest_observation')} differences={r.get('different_observations')} {r.get('error', '')}")
    print(f"All fresh: {report['all_fresh']}; cutoff: {report['information_cutoff']}; evidence: {out}")
    return 0 if report['all_fresh'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
