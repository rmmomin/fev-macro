# Point-in-time protocol

## Meaning of as-of

Strict `asof_ts=D` means the information set at the beginning of the New York calendar date containing D. Timezone-aware inputs are converted to America/New_York; naive inputs mean New York local time. Every same-day vintage is excluded, including at 23:59. For a release dated April 26, April 26 forecasts exclude it and April 27 forecasts can use it. Weekend dates are ordinary calendar dates; no invented holiday release schedule is applied.

ALFRED real-time periods have inclusive start/end dates. An output-type-1 request can clip observation intervals to the requested bounds. A one-day query labels all returned values with that day; it does not establish their initial release date. FRED defaults both bounds to today. Native observation dates refer to measured months/quarters, not release dates. The store queries the latest row, including NULL revisions, then checks its interval against the information date. An expired latest row never resurrects an older value. See the official [real-time-period specification](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) and [observations specification](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).

`vintagedates` reports dates of new/revised series values and omits releases with no changes. It is not a BEA stage calendar. A date-suffixed output-type-3 value reports a change, not a replacement for release-stage metadata. These behaviors were checked against live GDPC1 responses and retained as fixtures. [Official vintage-date specification](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html)

## Data admission and ingestion

Strict reads require `alfred_api_v2` records with real-time endpoints and linked sanitized request/response provenance. Older databases are rejected, including legacy records tagged `alfred_output_type_1`, because they lack endpoints/response evidence and may have lost withdrawals. Re-sync into a new file; merely changing a source label does not validate old data.

Backfills and updates use the same output-type-1 parser, explicit bounds, original units and frequency, complete pagination, and one transaction per series. API errors, malformed/incomplete pages, and conflicting values fail closed. Updates replay the overlap. `.` remains missing. Never substitute current FRED observations. Response JSON is stored in DuckDB and content-hashed; retrieval time is distinct from the historical vintage date. A bounded download supports snapshots only inside its interval coverage.

ALFRED is retrospective archival evidence, not a timestamped log of what a particular trader actually received. Source corrections to historical archives, undocumented intraday changes, and actual dissemination latency remain outside this guarantee. The strict mode requires honest externally sourced API records; it is not a defense against someone deliberately fabricating a database or fixture.

## Forecast inputs and horizons

Use `scripts/run_pit_backtest.py`. Explicit origin/target requests determine the schedule; GDP release truth never determines forecast dates or training values. Do not use an ex-post “day before the eventual release” schedule without proving that release date was announced by then. The included smoke schedule uses the 25th of April, July, October and January.

The last available GDP quarter determines horizon: `h = target_quarter - last_available_GDP_quarter`. For February 15, 2019, 2018Q4 GDP was not available yet; 2019Q1 is two steps from 2018Q3. Missing internal GDP quarters raise an error instead of compressing time. Target quarters already available at origin are refused.

Expanding windows use all contiguous admitted GDP levels. A rolling size N means N observed GDP levels; growth differencing and four lags reduce usable regression rows. Ragged feature rows do not consume that budget. Native-frequency transforms use only backward lags. GDP growth is always formed from adjacent levels in one origin snapshot, so revisions known at the origin are legitimate training data.

Strict variable IDs and transform codes are explicit configuration, not inferred from current template membership/aliases. Quarterly GDP stays a quarterly level. Quarterly bridge/tree/PCA/LSTM features use a mean of released transformed months, with observation count and missing indicator. The migrated DFM consumes native monthly series; VARs distinguish complete from partial quarterly aggregates. Unreleased months remain unknown. Daily, weekly, summed-flow and end-of-period aggregation are not implemented; specify and test their economic meaning before adding them. No outlier deletion or COVID-specific intervention is applied in the strict benchmark.

The bridge fits imputation means, scales and ridge coefficients on its training rows. For an unavailable forecast feature it uses the training mean with a missing indicator. It does not fetch future covariates. The core AR lag order, feature list and ridge penalty are fixed. The expanded catalog supports training-only automatic order selection and nested PIT ensemble selection. No saved leaderboard is consulted. Chronos requires dated immutable checkpoint evidence; a later checkpoint cannot be used for an earlier origin. [Model-specific rules](models.md) specify the forecast targets, preprocessing, monthly DFM conditioning, validation calendar and checkpoint assumptions.

## Truth and evaluation

A BEA-sourced calendar identifies advance/first, second and third releases. At each exact release date, read both GDP(q) and GDP(q-1) from the same ALFRED snapshot. Repeated unchanged estimates still count as releases. Do not divide a quarter's first-release level by the preceding quarter's own first-release level, and do not substitute a later monthly panel vintage.

- q/q percent: `100 * (GDP(q) / GDP(q-1) - 1)`.
- q/q SAAR percent: `100 * ((GDP(q) / GDP(q-1))**4 - 1)`.
- GDP levels already expressed at annual rates do not change these ratio formulas.
- Multi-step SAAR forecasts compare consecutive forecast levels; the first step uses the final observed level. It is not cumulative annualized growth over h quarters.

The strict KPI is SAAR percentage-point error. Level errors across chain-dollar rebasing are not ranked. RMSE/MAE and relative RMSE use matched finite model/baseline observations and expose matched/requested counts. Samples can still differ between pairwise comparisons, so do not interpret their rankings as a common-sample tournament without matching all models.

`paired_dm` supports one observation per consecutive target quarter at a fixed horizon/stage, at least 20 observations, matched truth, and Bartlett HAC with bandwidth at least h-1. Its normal p-values are asymptotic and unadjusted for multiple comparisons. It refuses repeated-target daily/monthly nowcasts and tiny samples. The four-quarter smoke backtest is a correctness check and has no significance claim.

## Provenance

Every forecast links to `audit.json` using `audit_id`. `max_vintage_inspected` covers the shared snapshot; `max_vintage_used` restricts to the model's consumed series and fitted GDP range (covariate inspection can include transformation warm-up history). Ensemble selection inputs are separately recorded in its nested audits. Audits expose the information date, observed GDP range, configuration, model input columns, transformed-month counts, and inspected source observations with vintage/end dates and API response IDs. The snapshot may include inspected records outside a rolling fit window; `training_start`/`training_end` identify the GDP values actually fitted, and `model_input_columns` identifies the consumed variables. The run manifest records source/config hashes, dependency versions, commit and dirty-tree status. Fit diagnostics include selected orders, seeds, warnings, preprocessing parameters and explicit failures. Ensembles include nested origin audits and checkpoint-backed forecasts include publisher evidence and artifact hashes. The portable `api_responses.json` export recursively includes the raw response evidence used for fitting, inner validation and truth; the DuckDB file retains the full local history.

## Certification boundary

The strict path supports defensible **retrospective point-in-time, temporally out-of-sample forecasts**, conditional on the archive, configuration and calendar. It does not prove that the specification was selected without hindsight or that forecasts were actually issued historically. The old fev/month-label, model-selection and latest-vintage pipelines are explicitly uncertified. See [AUDIT.md](../AUDIT.md).
