# fev-macro

US real-GDP forecasting research, with a small strict point-in-time benchmark and a broader exploratory model collection.

**The previous “vintage-correct by default” claim was not supported.** The audit found current-FRED history backdated to observation dates, missing historical values filled from later panels, ambiguous archive dates, and ex-post model selection. These outputs are not validated historical forecasts. See [AUDIT.md](AUDIT.md) for findings, fixes, tests, and remaining limits.

## Strict benchmark

A forecast dated `D` uses ALFRED vintages dated **before D**, interpreting dates in America/New_York. Same-day releases are excluded even for an afternoon forecast because ALFRED has no intraday availability timestamps. Observation dates identify the period measured, not when its value became known.

The strict path reads verified ALFRED API intervals directly. It never reads latest-data CSVs, archive panels, or leaderboards. GDP release truth is loaded after forecasts are made, using a sourced BEA release calendar and same-release-vintage numerator and denominator.

```bash
# Python 3.11; minimal pinned dependencies, no neural-model stack needed
python3 -m venv .venv-pit
.venv-pit/bin/python -m pip install -r requirements-pit.txt

# Offline historical smoke backtest from captured real API responses.
# Use a new database path. On repeat runs, omit --fixture-dir.
.venv-pit/bin/python scripts/run_pit_backtest.py \
  --db data/realtime/pit_smoke.duckdb \
  --fixture-dir tests/fixtures/alfred \
  --origins tests/fixtures/alfred/origins.csv \
  --release-calendar tests/fixtures/alfred/release_calendar.csv \
  --series-specs tests/fixtures/alfred/series_specs.json \
  --models naive_last last_growth mean_growth ar4 bridge_ridge \
  --covariates UNRATE --out results/pit_smoke

.venv-pit/bin/python -m pytest tests/test_pit_contract.py -q
```

Outputs include `forecasts.csv`, `truth.csv`, `scored.csv`, paired `metrics.csv`, per-origin `audit.json`, portable `api_responses.json`, and `manifest.json`. Forecast rows link to the audit using a content hash. Stored API responses retain request bounds, values, interval endpoints, retrieval time, and provenance IDs without API keys.

Supported strict models are a constant-level baseline, last/mean log-growth baselines, fixed AR(4) on log growth, and a fixed ridge bridge with four GDP growth lags and declared monthly/quarterly features. Features are transformed at their native frequency, then averaged over released observations; counts and missing indicators expose partial quarters. Imputation/scaling are fitted on training rows. This is a simple bridge, not a dynamic mixed-frequency state-space model.

```bash
# Verified live backfill. Existing legacy databases must be rebuilt into a NEW file.
export FRED_API_KEY="..."  # alternatively use the ignored .env file
.venv-pit/bin/python scripts/sync_alfred_asof_store.py \
  --db data/realtime/pit.duckdb --series GDPC1 UNRATE \
  --observation_start 2005-01-01

# Refresh/replay the overlap; failed series return nonzero and appear in the report.
.venv-pit/bin/python scripts/sync_alfred_asof_store.py \
  --db data/realtime/pit.duckdb --series GDPC1 UNRATE --no-backfill_missing

# Deliberate network validation / fixture refresh
FEV_LIVE_ALFRED=1 .venv-pit/bin/python -m pytest tests/test_pit_contract.py -m integration -q
.venv-pit/bin/python scripts/capture_alfred_fixtures.py
```

Supply your own fixed origin CSV (`origin_date,target_quarter`), series specification JSON, and release calendar (`quarter,stage,release_date,source_url`, optional `expected_saar`) for other periods. The checked calendar covers only 2019. Keep design/tuning periods separate from evaluation periods; this code cannot prove that a human chose a model before seeing its test results.

Definitions and limitations: [point-in-time protocol](docs/realtime_protocol.md). Data transformations: [processing notes](docs/data_processing.md).

## Exploratory workflows

The historical MD/QD panel builders, `fev` model catalog, BoE exports, and latest-vintage scripts remain available for research. Legacy historical evaluation now refuses to run by default; `--no-strict-pit` explicitly opts into uncertified results. Selecting models or ensemble members from a leaderboard evaluated over the same periods is ex-post selection, not independent out-of-sample evidence.

```bash
# Broader optional model dependencies
python -m pip install -r requirements.txt
python -m pytest -q

# Explicitly exploratory; see AUDIT.md before interpreting scores
python scripts/run_eval_processed.py --profile smoke --no-strict-pit
python scripts/run_realtime_oos.py --mode processed --no-strict-pit
```

Latest-vintage 2025Q4 artifacts are retrospective if generated after its releases. They cannot be relabelled PIT forecasts. Existing generated results were left intact and are superseded as evidence by the audit.

- `src/fev_macro/pit*.py`, `asof_store.py`, `asof_provider.py`: strict contract and benchmark
- `tests/fixtures/alfred/`: captured official responses, explicit origins and BEA release calendar
- `scripts/`: ingestion, strict backtest, exploratory evaluation, and data utilities
- `data/`, `results/`: local data and generated artifacts (mostly ignored by Git)
- [Research model catalog](docs/models.md), [benchmark notes](docs/benchmarks.md), [BoE notes](docs/boe_evaluation.md)
