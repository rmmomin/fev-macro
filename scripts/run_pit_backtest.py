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
from fev_macro.pit_benchmark import MODELS, build_release_truth, paired_metrics, run_pit_backtest, score_forecasts, provenance_ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--origins", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--release-calendar", help="Sourced release CSV for scoring after fitting")
    mode.add_argument("--forecast-only", action="store_true", help="Forecast unreleased targets without truth/scoring")
    p.add_argument("--series-specs", help="JSON mapping of explicit variable frequency and tcode")
    cov = p.add_mutually_exclusive_group()
    cov.add_argument("--covariates", nargs="*", default=[])
    cov.add_argument("--covariates-from-specs", action="store_true", help="Use all declared variables except GDPC1 in JSON order")
    p.add_argument("--models", nargs="+", choices=[*MODELS, "all"], default=["naive_last", "mean_growth", "ar4"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--on-model-error", choices=["raise", "record"], default="raise")
    p.add_argument("--ensemble-windows", type=int, default=8)
    p.add_argument("--chronos-checkpoint", help="Pinned local checkpoint evidence manifest")
    p.add_argument("--rolling-size", type=int)
    p.add_argument("--min-train", type=int, default=24)
    p.add_argument("--out", default="results/pit_backtest")
    p.add_argument("--fixture-dir", help="Replay gdpc1/unrate captured API fixtures into a NEW database for smoke testing")
    args = p.parse_args()
    if "all" in args.models:
        if len(args.models) != 1:
            p.error("Use --models all on its own")
        args.models = list(MODELS)
    if not args.forecast_only and "naive_last" not in args.models:
        p.error("Scored runs require naive_last for paired benchmark metrics")
    out = Path(args.out)
    if any((out / name).exists() for name in ("forecasts.csv", "truth.csv", "scored.csv", "metrics.csv", "audit.json", "manifest.json")):
        p.error("Use a new output directory to preserve earlier results and avoid stale score files")
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
    if args.covariates_from_specs:
        args.covariates = [name for name in specs if name != "GDPC1"]
    provider = AsofVintageProvider(db_path=args.db, covariate_mode="processed", series_specs=specs)
    try:
        forecasts, audit = run_pit_backtest(provider, pd.read_csv(args.origins), models=args.models,
                                           covariates=args.covariates, rolling_size=args.rolling_size,
                                           min_train=args.min_train, seed=args.seed,
                                           on_model_error=args.on_model_error, ensemble_windows=args.ensemble_windows,
                                           chronos_checkpoint=args.chronos_checkpoint)
        # Truth cannot affect fitting: it is opened only after forecasting.
        truth = scored = metrics = None
        if not args.forecast_only:
            truth = build_release_truth(provider.store, pd.read_csv(args.release_calendar))
            scored = score_forecasts(forecasts, truth)
            metrics = paired_metrics(scored)
        # Export the response ledger used in this run so the result is portable.
        ids = provenance_ids(audit)
        if truth is not None:
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
    if truth is not None:
        truth.to_csv(out / "truth.csv", index=False)
        scored.to_csv(out / "scored.csv", index=False)
        metrics.to_csv(out / "metrics.csv", index=False)
    forecasts.loc[forecasts.status != "ok"].to_csv(out / "failures.csv", index=False)
    (out / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    (out / "api_responses.json").write_text(json.dumps(api_responses, indent=2) + "\n")
    manifest = dict(config=vars(args), input_sha256={Path(f).name: content_hash(Path(f).read_text())
                    for f in (args.origins, args.release_calendar, args.series_specs, args.chronos_checkpoint) if f},
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    working_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                    versions={m: importlib.metadata.version(m) for m in ("numpy", "pandas", "duckdb")},
                    python_version=platform.python_version(),
                    audit_sha256=content_hash(audit), api_responses_sha256=content_hash(api_responses),
                    source_sha256={str(f.relative_to(ROOT)): content_hash(f.read_text())
                                   for f in [*sorted((ROOT / "src" / "fev_macro").glob("*.py")), Path(__file__).resolve()]})
    for package in ("statsforecast", "statsmodels", "scikit-learn", "xgboost", "torch", "chronos-forecasting", "transformers", "scipy"):
        try:
            manifest["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["versions"][package] = None
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(forecasts[["model", "target_quarter", "g_hat_saar", "status", "error"]].to_string(index=False))
    if metrics is not None:
        print(metrics.to_string(index=False))
    print(f"Wrote forecasts, failure report, audit, API evidence and manifest to {out.resolve()}")


if __name__ == "__main__":
    main()
