#!/usr/bin/env python3
"""Run the strict benchmark from a verified store and explicit origin/calendar CSVs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from fev_macro.asof_provider import AsofVintageProvider
from fev_macro.asof_store import AsofStore
from fev_macro.pit import content_hash
from fev_macro.pit_benchmark import MODELS, build_release_truth, paired_metrics, run_pit_backtest, score_forecasts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--origins", required=True)
    p.add_argument("--release-calendar", required=True)
    p.add_argument("--series-specs", help="JSON mapping of explicit variable frequency and tcode")
    p.add_argument("--covariates", nargs="*", default=[])
    p.add_argument("--models", nargs="+", choices=MODELS, default=["naive_last", "mean_growth", "ar4"])
    p.add_argument("--rolling-size", type=int)
    p.add_argument("--min-train", type=int, default=24)
    p.add_argument("--out", default="results/pit_backtest")
    p.add_argument("--fixture-dir", help="Replay gdpc1/unrate captured API fixtures into a NEW database for smoke testing")
    args = p.parse_args()
    if args.fixture_dir:
        if Path(args.db).exists():
            p.error("Fixture replay requires a new database path")
        store = AsofStore(args.db)
        try:
            for name in ("gdpc1_intervals", "unrate_intervals"):
                record = json.loads((Path(args.fixture_dir) / (name + ".json")).read_text())
                store.ingest_alfred_response(record["response"], record["params"], retrieved_at=record["retrieved_at"])
        finally:
            store.close()
    elif not Path(args.db).is_file():
        p.error("Verified ALFRED database does not exist")
    specs = json.loads(Path(args.series_specs).read_text()) if args.series_specs else {"GDPC1": {"frequency": "Q"}}
    provider = AsofVintageProvider(db_path=args.db, covariate_mode="processed", series_specs=specs)
    try:
        forecasts, audit = run_pit_backtest(provider, pd.read_csv(args.origins), models=args.models,
                                           covariates=args.covariates, rolling_size=args.rolling_size,
                                           min_train=args.min_train)
        # Truth cannot affect fitting: it is opened only after forecasting.
        truth = build_release_truth(provider.store, pd.read_csv(args.release_calendar))
        scored = score_forecasts(forecasts, truth)
        metrics = paired_metrics(scored)
        # Export the response ledger used in this run so the result is portable.
        ids = {row["provenance_id"] for origin in audit for row in origin["inputs"]}
        for group in truth.truth_provenance_ids:
            ids.update(json.loads(group))
        api_responses = []
        for pid in sorted(ids):
            record = provider.store._con.execute(
                "SELECT request_json, response_json, retrieved_at, endpoint FROM asof_api_responses WHERE provenance_id=?",
                [pid]).fetchone()
            if record is None:
                raise ValueError("Referenced API evidence is missing")
            api_responses.append(dict(provenance_id=pid, params=json.loads(record[0]),
                                      response=json.loads(record[1]), retrieved_at=record[2], endpoint=record[3]))
    finally:
        provider.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out / "forecasts.csv", index=False)
    truth.to_csv(out / "truth.csv", index=False)
    scored.to_csv(out / "scored.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    (out / "api_responses.json").write_text(json.dumps(api_responses, indent=2) + "\n")
    manifest = dict(config=vars(args), input_sha256={Path(f).name: content_hash(Path(f).read_text())
                    for f in (args.origins, args.release_calendar) + ((args.series_specs,) if args.series_specs else ())},
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    working_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                    versions={m: importlib.metadata.version(m) for m in ("numpy", "pandas", "duckdb")},
                    python_version=platform.python_version(),
                    audit_sha256=content_hash(audit), api_responses_sha256=content_hash(api_responses),
                    source_sha256={str(f.relative_to(ROOT)): content_hash(f.read_text())
                                   for f in [*sorted((ROOT / "src" / "fev_macro").glob("*.py")), Path(__file__).resolve()]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print(f"Wrote forecasts, truth, metrics, audit and manifest to {out.resolve()}")


if __name__ == "__main__":
    main()
