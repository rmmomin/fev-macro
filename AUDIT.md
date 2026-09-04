# Research and implementation audit — 2026-09-04

## Judgment

**The repository as a whole cannot defensibly be called a true point-in-time out-of-sample benchmark.** Its previously generated historical results are uncertified. Selecting a file labelled with a vintage month was not sufficient to make those results “vintage-correct.”

**The new strict path can support retrospective point-in-time, temporally out-of-sample forecasts**, conditional on ALFRED's historical archive, a correct sourced release calendar, and an explicitly fixed specification. It isolates fitting from scoring, refuses unverifiable history, and records its information set. It does not prove that the researcher chose that specification without hindsight or actually issued forecasts historically. Those stronger claims require an untouched evaluation period or archived/preregistered forecasting protocol.

The work was performed directly in the existing working tree, initially clean at `f78aaa3`. No clone/replacement, destructive database migration, or rewriting of existing result artifacts was performed. The strict implementation intentionally supports fewer models and frequencies than the exploratory collection.

## Ranked findings and disposition

| Severity | Finding and methodological consequence | Implemented response / boundary |
|---|---|---|
| **Critical C1** | `sync_alfred_asof_store.backfill_series_output_type_1` caught HTTP 400, downloaded current FRED values, and assigned `asof_ts=obs_date`. Revised historical values then appeared known at the beginning of their measurement period, often before the period had even occurred. | Removed this fallback entirely. Unsupported/failed series return nonzero and are reported. Strict queries reject all legacy/unproven rows. New verified response-ledger ingestion is required. |
| **Critical C2** | `AsofVintageProvider.adapt_train_df` retained base-panel values whenever a PIT value was missing, and returned the entire base on empty snapshots. `HistoricalQuarterlyVintageProvider.adapt_past_data` similarly filled missing target/covariate values from the later scaffold. This directly contaminated ragged edges, including the target. | PIT panels now replace the data completely; missing values remain unknown, empty snapshots fail, and no fallback to base values exists. Removed scaffold fallback in the historical provider too. Synthetic-hole and future-revision tests exercise the failure. |
| **Critical C3** | The historical provider aligned timestamps literally. Actual FRED-QD files can label a quarter by its last month (March 1), while GDP scaffolds use January 1. Missing matches could therefore fall through to later truth for whole histories. | Normalize economic quarters before alignment and reject duplicate quarterly targets. Added a last-month-versus-quarter-start fixture with unreleased targets. |
| **Critical C4** | `scripts/run_realtime_oos.py` selects models and top-three/top-five ensembles using a leaderboard without establishing that its losses precede each forecast origin. Results selected on overlapping test periods are ex-post model selection, even with valid input vintages. | Strict CLI rejects that route by default. The strict benchmark admits only fixed predeclared models/features/penalties and never reads a leaderboard. Nested origin-aware selection for the broad model set remains unimplemented. |
| **High H1** | Store queries used `asof_ts <= origin` with dates treated as midnight, implicitly admitting same-day releases before their actual publication. Neither archive month-start timestamps nor `BMonthEnd` availability conventions establish a real release instant. | A single strict New York prior-calendar-day rule applies to every query. Archive month labels are excluded from the strict path. Legacy quarterly selection avoids same-month labels, but remains an approximation and is not certified. |
| **High H2** | Store `ARG_MAX(value, asof_ts)` ignores NULL values. Sync skipped `.` observations and discarded `realtime_end`. A withdrawal or expired interval could resurrect a superseded value. | Preserve withdrawals/endpoints; select the newest row including NULL, then require interval validity. Never fall back to an older interval after a gap. Tests cover inclusive endpoints, NULL revisions and expired rows. |
| **High H3** | GDP first/second/third truth was based on the first three nonmissing vintage columns. A series-change calendar is not a release-stage calendar: extra updates, unchanged releases and early archive gaps break this identification. Legacy panel “next vintage” truth can use later revisions. | Strict truth requires explicit BEA stage/date/source records and samples the exact day's GDP pair. The old wide builder now requires a calendar by default and does not promote extra vintages to release stages; missing exact dates remain unsupported. No panel approximation is admitted to strict scoring. |
| **High H4** | `realtime_oos` sought `qoq_saar_growth_realtime_*` while the current GDP writer emits `qoq_saar_growth_alfred_*`. The fallback divided adjacent quarters' independently dated release levels, potentially mixing chain-dollar scales. | Legacy preference now uses the writer's same-vintage ALFRED columns. Strict truth has no cross-vintage fallback and verifies optional published BEA rounded growth. Level errors across rebasing are not the strict KPI. |
| **High H5** | The as-of provider quarterly-averaged before transforming, used the latest QD transformation metadata, then called a target builder that drops missing-target rows and fills covariates. These operations change mixed-frequency meaning and can erase the nowcast ragged edge. | Strict provider uses explicit IDs/frequencies/codes, transforms at native frequency, preserves native gaps, then averages only released observations. Counts and missing indicators expose partial quarters. No implicit present-day template aliases or COVID interventions. |
| **High H6** | Legacy ragged rows could consume a rolling window's target budget. Forecast paths begin after the last observed GDP quarter, but labels used a requested observed-quarter convention, even when that GDP release was delayed. Target gaps could compress calendar time. | Strict rolling windows count contiguous observed GDP levels. Horizon is the actual calendar distance to the target; unavailable internal quarters fail. Legacy forecast labels now follow the last observed target too. Test: February 15, 2019 has 2018Q3 as last GDP and a two-step horizon to 2019Q1. |
| **High H7** | Broad factor/MD models can load panels independently; model adapters backfill, use silent model fallbacks, and some workflows remove/reindex COVID years. Pretrained Chronos weights have no established historical training cutoff here. The provider alone cannot certify these models. | The strict runner has its own small deterministic model implementation and no hidden data access. Those research models remain available only on explicitly uncertified paths. No claim that every model-specific methodological problem is fixed. |
| **Medium M1** | Incremental sync fetched a lookback but excluded every vintage at/before the checkpoint. A partial run or later completion on the checkpoint date could never be repaired. Pagination/parse failures could silently truncate history. | Unified backfill/update interval queries, replay the overlap, validate count/offset/page length, and commit one complete series transaction. Failed multi-page runs roll back observations and provenance together. Conflicting same-version values raise rather than silently overwriting. |
| **Medium M2** | Metadata did not retain raw API responses or actual used observations. Error messages included credential-bearing URLs. Missing configured archive paths could silently autodiscover unrelated local data. | Sanitized content-addressed request/response ledger, forecast audit IDs, portable API-response export, source/config/dependency manifest; redacted request errors. Missing configured archive directories now fail instead of searching other data roots. |
| **Medium M3** | Legacy relative RMSE divides aggregate model/benchmark scores that may use different finite samples. BoE automatic baselines and pooled repeated-target DM inference are not proven PIT-correct. Small smoke samples cannot support significance claims. | Strict metrics pair by origin/target/horizon/stage and report counts. A guarded squared-loss DM helper requires at least 20 consecutive unique quarterly targets with identical truth and uses Bartlett HAC. No DM p-values are reported for the smoke sample. Legacy BoE remains uncertified. |
| **Medium M4** | README/docs asserted an authoritative vintage-correct workflow and full MD trimming/outlier semantics. The archive panel builder actually applies transform codes; broader scripts have differing processing. Latest 2025Q4 runs could be misread as historical predictions. | Rewrote scope/protocol/processing/benchmark docs. Legacy historical CLIs default to refusal; explicit `--no-strict-pit` opts into exploration. Legacy/latest outputs are labelled uncertified. Existing stored results are not relabelled. |
| **Low L1** | Duplicate ingestion returned attempted row counts; API-format branching and duplicated ingestion paths obscured what was stored. Broad dependencies were required to reach the as-of provider. | Return actual insertion counts; remove obsolete output-type-3/key-inference/vintage-date ingestion branches; lazy-load legacy data helpers. Added a minimal pinned strict runtime, clean-environment test, and CI job. |

## API and release evidence

Official specifications consulted: [real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html), [observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), and [vintage dates](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html). The precise contract is in [docs/realtime_protocol.md](docs/realtime_protocol.md).

Actual authenticated API responses were captured locally without printing/saving the key. Fixtures retain sanitized parameters, retrieval time and decoded JSON:

- GDPC1 2019Q1: first value 18912.326 on April 26; second 18907.517 on May 30; third 18910.332 on June 27. Exact-day April 25 returns no 2019Q1 observation; April 26 returns GDP(q) and GDP(q-1), both with bounds clipped to that query day.
- Nominal GDP: a second quarterly series validates the same release boundaries and different levels/units.
- UNRATE January 2019: 4.0 first available February 1, consistent with the [BLS release](https://www.bls.gov/news.release/archives/empsit_02012019.htm).
- CPIAUCSL January 2019: 252.673 first available February 13. The [BLS release](https://www.bls.gov/news.release/archives/cpi_02132019.htm) notes a same-day reissue, reinforcing the intraday limitation.
- A live GDPC1 `vintagedates` request returned April 26, May 30 and June 27. An output-type-3 May 30 query returned only the changed 2019Q1 record under `GDPC1_20190530`; unchanged 2018Q4 was absent. The old key format was therefore valid for this representative response; the proven update bugs were skipping missing values/checkpoint replay and inadequate failure handling, not a claim that every type-3 response was misparsed.
- All 12 first/second/third 2019 quarterly growth truths agree with the rounded BEA rates within 0.06 percentage points. Individual official sources and expected rates are in [release_calendar.csv](tests/fixtures/alfred/release_calendar.csv). This check uses same-release q and q-1 levels, including annual-revision changes to the denominator.

Date-level evidence is sufficient for the conservative next-calendar-day convention. It is not sufficient for an 08:00 versus 08:30 same-day comparison. ALFRED historical timestamps also do not prove institution-specific dissemination or ingestion latency.

## Implementation map

- `src/fev_macro/pit.py`: date convention, provenance hashes, validated ALFRED interval parsing.
- `src/fev_macro/asof_store.py`: schema migration that leaves old records untrusted, raw-response ledger, NULL-safe interval snapshots, conflict handling and real insertion counts.
- `scripts/sync_alfred_asof_store.py`: no FRED fallback, complete transactional pagination, overlapping update replay, explicit series lists, sanitized errors and nonzero failures.
- `src/fev_macro/asof_provider.py`: full PIT reconstruction, explicit IDs/native transformations, ragged quarters and per-observation metadata.
- `src/fev_macro/pit_benchmark.py`, `scripts/run_pit_backtest.py`: independent strict fitting/scoring, five fixed models, exact release truth, matched metrics, restricted DM and auditable outputs.
- `src/fev_macro/data.py`, `realtime_feeds.py`, `realtime_oos.py`: remove later-scaffold fallback, align quarters, avoid backward fill in common input adapters, reject compressed target gaps, correct legacy horizon/ALFRED truth preference, explicitly label/refuse uncertified paths.
- `scripts/build_gdp_releases.py`: sourced stage-calendar requirement and exact-date lookup for wide snapshots. Its older panel-comparison validation remains exploratory; the strict runner's truth builder is authoritative.
- Legacy evaluation/latest/one-off as-of scripts and `eval_runner.py`: explicit certification boundary and safer defaults.
- `scripts/capture_alfred_fixtures.py`, `scripts/compare_pit_smoke.py`: reproducible API capture and deliberately labelled revision-contamination comparison.
- `requirements-pit.txt`, `pytest.ini`, `.github/workflows/pit.yml`, Makefile: minimal strict runtime, discoverable tests and CI.

## Tests and execution

Baseline in the original local environment: **53 passed, 4 skipped**. DuckDB was absent, so three core as-of tests had not been exercising the store. Installed DuckDB, which is already a declared repository dependency, and ran all tests again after the fixes.

New tests in `tests/test_pit_contract.py` cover:

- Same-day/UTC-to-New-York boundaries; future observation dates; inclusive interval ends; NULL withdrawals; no resurrection; legacy/unproven rows; duplicate/conflicting ingestion.
- Frozen live GDP, GDPC1, UNRATE and CPI responses; clipped bounds; type-3 and vintage-date behavior; twelve BEA stage truths; unchanged release stages and same-vintage denominators.
- Native transformations before quarterly aggregation, partial-quarter counts, missing PIT values versus later scaffolds, forecasts invariant to extreme post-origin revisions/targets, target aliases, internal target gaps, delayed releases and rolling-window budgets.
- HTTP errors with no current-FRED fallback, multi-page transaction rollback, checkpoint-day replay, invalid API configurations, credential redaction.
- Matched benchmark samples, duplicate scores, DM sample/dependence restrictions, portable CLI round-trip, explicit live API integration, and refusal by uncertified legacy defaults.

Updated existing tests explicitly opt into the legacy behavior they test. They do not pretend those paths are certified. Added regression tests for archive-quarter alignment/no target fallback and explicit GDP stage dates despite extra intermediate snapshots.

Validation commands and observed results:

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | **96 passed, 2 skipped** (optional BoE dependency; explicitly opt-in live API) |
| `FEV_LIVE_ALFRED=1 .venv/bin/python -m pytest -q` | **97 passed, 1 skipped** (optional `forecast_evaluation` dependency unavailable) |
| Fresh Python 3.11 venv; install only `requirements-pit.txt`; run `tests/test_pit_contract.py` | **37 passed, 2 skipped** (optional full-stack legacy guard; live API) |
| Live paginated sync of GDPC1/UNRATE, followed by overlap update | **12 API requests, 8 observation pages, zero failures/retries**, 9.51 seconds |
| Full strict historical backtest using that live-synced database | **20 forecasts, 60 scored rows**, identical numeric forecasts to offline fixture replay |
| `git diff --check`, Python compile check and CLI help checks | Passed |

The optional BoE package is not needed by the strict path. Its test skip is retained rather than misreported as validation of BoE inference. A CI job runs the strict contract in the minimal environment; it deliberately has no secrets/network requirement.

## Behavioral and numerical changes

The strict smoke run fixes four dates (2019-04-25, 2019-07-25, 2019-10-25, 2020-01-25), forecasts 2019Q1–Q4, trains from 2005 where available, and scores all three release stages. There was no hyperparameter search or selection by these outcomes.

First-release SAAR RMSE, in percentage points:

| Fixed model | Strict PIT | Deliberately contaminated with 2020-04-30 revised history |
|---|---:|---:|
| Constant GDP level (`naive_last`) | 2.361271 | 2.361271 |
| Last log growth | 0.736677 | 1.164808 |
| Mean log growth | 0.738219 | 0.734156 |
| AR(4) log growth | 0.569793 | 0.831904 |
| Fixed ridge bridge with UNRATE | 0.539895 | Not compared |

The largest change in an individual forecast was **1.076742 SAAR percentage points**. The contaminated comparison holds origins, target masking, estimation ranges and model formulas fixed while substituting a later GDP history. It illustrates the removed current-FRED fallback's revised-history effect; it is **not** a reproduction of the entire old leaderboard or a claim that leakage always improves scores. Four quarters are insufficient to establish relative model quality.

Reproduce from a new DB with the README command, then run:

```bash
python scripts/compare_pit_smoke.py --db data/realtime/pit_smoke.duckdb --results results/pit_smoke
```

Full smoke metrics and comparison tables are retained in `docs/audit_smoke_metrics.csv` and `docs/audit_revision_comparison.csv`. Detailed local outputs are under `results/pit_backtest/` and `results/pit_live_backtest/`, including forecast-level audit IDs, `audit.json`, `api_responses.json` and manifests. Old result files were left untouched.

## Remaining limitations and tradeoffs

1. **Archive fidelity and timing:** ALFRED can correct historical archives; date-level releases cannot resolve intraday order. Excluding all same-day values sacrifices timeliness but avoids inventing it. Refresh/replay lookbacks do not discover arbitrary corrections years outside the lookback; periodic full revalidation is needed. Conflicting same-key corrections intentionally require investigation/rebuilding a new store.
2. **Research selection:** Explicit configuration is recorded, not proof of preregistration. Broad leaderboards, hand-selected features, survivor universes, historical aliases and pretrained-model training cutoffs remain outside certification. No nested origin-aware selection has been implemented for them.
3. **Coverage:** The verified release-calendar fixture covers 2019 only. Other periods require separately sourced calendars. Early GDP archive observations must not be called first releases simply because they are the oldest available. Missing truth fails or is left unsupported, never replaced by a later vintage.
4. **Model scope:** The bridge is quarterly regression over partially observed native-frequency features, not a daily event-driven DFM/Kalman filter. Training historical quarters can have fuller data than the forecast quarter; counts/missing indicators make that mismatch explicit, but do not solve it statistically. Daily/weekly/flow/end-of-period aggregation and all broad model variants need separate validation.
5. **Inference:** Relative RMSE uses pairwise matched samples; different model pairs can still cover different periods. DM is asymptotic, has no multiple-testing correction, and is refused for repeated-target nowcasts. The legacy BoE generated benchmarks/tests remain unverified.
6. **Operational scope:** This is a single-process local DuckDB benchmark, not a distributed ingestion service. Raw-response JSON retention grows storage; no compaction/retention service is implemented. Direct dependencies are pinned and manifests record runtime versions, but complete platform/transitive dependency identity is not guaranteed.
7. **Legacy behavior:** The archive pipelines retain exploratory approximations and some model-level fallbacks. Their default refusal is intentional, not a claim they were fully repaired. Latest-data scenarios and pre-existing CSV/plot results are not converted into PIT evidence by these changes.

**Final disposition:** use the strict path for auditable retrospective information sets and temporal holdout forecasts. Do not make an unqualified “true point-in-time out-of-sample” claim about the old results, the full model catalog, intraday forecasts, or research specifications selected after examining their evaluation outcomes.
