"""Shared point-in-time contract. ALFRED dates are not intraday timestamps."""
from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd


class PITError(ValueError):
    """The requested information set cannot be established safely."""


def information_date(origin: object) -> pd.Timestamp:
    """Last completed New York calendar date before the forecast origin.

    Naive inputs are New York local time. Even 23:59 origins exclude that day's
    vintages: ALFRED supplies no time of day at which a record was available.
    """
    ts = pd.Timestamp(origin)
    if pd.isna(ts):
        raise PITError("Forecast origin must be a valid date/time")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    return ts.normalize() - pd.Timedelta(days=1)


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def alfred_rows(payload: dict, params: dict) -> pd.DataFrame:
    """Validate an untransformed output_type=1 response, preserving withdrawals.

    Real-time bounds can be clipped to the requested interval. They are evidence
    only within that interval, never evidence of an earlier initial release.
    """
    if int(params.get("output_type", 0)) != 1 or int(payload.get("output_type", 0)) != 1:
        raise PITError("Only ALFRED output_type=1 interval responses are supported")
    if params.get("units", "lin") != "lin" or payload.get("units") != "lin" or params.get("frequency"):
        raise PITError("Store requires native-frequency untransformed ALFRED levels")
    if not params.get("realtime_start") or not params.get("realtime_end"):
        raise PITError("Explicit real-time bounds are required")
    start, end = date.fromisoformat(params["realtime_start"]), date.fromisoformat(params["realtime_end"])
    if start > end or payload.get("realtime_start") != str(start) or payload.get("realtime_end") != str(end):
        raise PITError("ALFRED response real-time bounds do not match the request")
    if not isinstance(payload.get("observations"), list):
        raise PITError("Missing observations in ALFRED response")
    rows = []
    for obs in payload["observations"]:
        try:
            a, b, d = (date.fromisoformat(obs[k]) for k in ("realtime_start", "realtime_end", "date"))
            if not start <= a <= b <= end:
                raise ValueError("interval outside request")
            value = None if obs["value"] == "." else float(obs["value"])
            if value is not None and not np.isfinite(value):
                raise ValueError("nonfinite observation")
        except (KeyError, TypeError, ValueError) as exc:
            raise PITError("Malformed ALFRED observation") from exc
        rows.append(dict(series_id=params["series_id"], obs_ts=str(d), asof_ts=str(a),
                         realtime_end=str(b), value=value))
    return pd.DataFrame(rows, columns=["series_id", "obs_ts", "asof_ts", "realtime_end", "value"])
