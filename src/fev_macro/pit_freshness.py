"""Compare a verified store with independently requested ALFRED cutoff snapshots.

Fresh means equal to the API at this cutoff and within this observation range.
It does not mean every agency release has reached ALFRED or every economic
indicator is in the declared universe. Snapshot bounds are NOT release dates.
"""
from __future__ import annotations

import pandas as pd

from .pit import PITError, alfred_rows, content_hash, information_date


def check_freshness(store, *, origin, series_specs, observation_start, fetch, retrieved_at):
    if not store.strict_pit:
        raise PITError('Freshness requires a verified PIT store')
    cutoff = information_date(origin).date().isoformat()
    if pd.Timestamp(observation_start) > pd.Timestamp(cutoff):
        raise PITError('Observation start exceeds information cutoff')
    rows, evidence = [], []
    for variable, spec in series_specs.items():
        sid = spec.get('series_id', variable)
        report = dict(variable=variable, series_id=sid, information_cutoff=cutoff)
        try:
            remote, offset, count = [], 0, None
            while True:
                params = dict(series_id=sid, output_type=1, units='lin',
                    realtime_start=cutoff, realtime_end=cutoff,
                    observation_start=observation_start, observation_end=cutoff,
                    limit=100000, offset=offset, sort_order='asc')
                response = fetch(params)
                page = alfred_rows(response, params)
                total = int(response['count'])
                count = total if count is None else count
                if (total != count or int(response.get('offset', -1)) != offset
                        or len(page) != min(params['limit'], count-offset)):
                    raise PITError('Incomplete or inconsistent freshness pagination')
                record = dict(params=params, response=response, retrieved_at=retrieved_at)
                record['sha256'] = content_hash(record)
                evidence.append(record)
                remote.append(page)
                offset += len(page)
                if offset == count:
                    break
            remote = pd.concat(remote, ignore_index=True)
            if remote.obs_ts.duplicated().any():
                raise PITError('Multiple values at a single cutoff')
            local = store.snapshot_long(asof_ts=origin, series_ids=[sid],
                obs_start=observation_start, obs_end=cutoff, include_asof_used=True)
            def values(frame):
                # Absent and explicit missing both mean no usable observation.
                return {str(pd.Timestamp(r.obs_ts).date()): float(r.value)
                        for r in frame.itertuples() if pd.notna(r.value)}
            actual, expected = values(local), values(remote)
            differences = [dict(obs_date=d, local=actual.get(d), alfred=expected.get(d))
                           for d in sorted(actual.keys() | expected.keys()) if actual.get(d) != expected.get(d)]
            report.update(status=('unsupported' if not expected else 'stale' if differences else 'fresh'),
                local_latest_observation=max(actual, default=None),
                alfred_latest_observation=max(expected, default=None),
                local_usable_observations=len(actual), alfred_usable_observations=len(expected),
                max_local_vintage=(str(local.asof_used.max().date()) if not local.empty else None),
                differences=differences, different_observations=len(differences),
                local_snapshot_sha256=content_hash(actual), alfred_snapshot_sha256=content_hash(expected))
        except (PITError, RuntimeError, OSError, KeyError, TypeError) as exc:
            report.update(status='unverified', error=f'{type(exc).__name__}: {exc}')
        rows.append(report)
    return dict(origin=str(origin), information_cutoff=cutoff, observation_start=observation_start,
        checked_at=retrieved_at, scope='declared series and observation range; ALFRED cutoff snapshot equality',
        all_fresh=bool(rows) and all(r['status'] == 'fresh' for r in rows), series=rows), evidence
