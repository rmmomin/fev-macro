# GDPNow and New York Fed indicator collection

## Coverage and freshness on September 6, 2026

The collection has **46 series: quarterly GDP, 43 monthly indicators, and two
additional quarterly indicators**. The original catalog had GDP and 19 monthly
indicators. `config/pit/nowcast_series.json` adds 22 monthly indicators to the
model panel; `nowcast_auxiliary_series.json` separately collects real GDI, unit
labor costs and two monthly advance inventory series that need different modeling.
All use explicit FRED IDs, native frequencies and predeclared transformations.
The existing catalog configuration is preserved as a comparison universe.

The forecast origin is beginning of **September 6, New York time**. The eligible
information date is **September 5**. The prior store was deliberately bounded
by September 3; it cannot establish information at later origins without a
refresh. Its old forecasts remain valid for their stated cutoff.

[Atlanta Fed GDPNow](https://www.atlantafed.org/research-and-data/data/gdpnow)
reports **4.7% SAAR for 2026Q3**, updated September 3, with the next update
September 10. The supplied workbook's `Table!A16:C16` records September 3 and
4.748663%. GDPNow's publication schedule is a guide to releases, not a universal
information cutoff. Our September 6 origin also admits September 4 ALFRED
employment and vehicle-sales vintages. The workbook describes model forecasts
for missing monthly data, and some historical sheets explicitly use revised
data. None of those values were imported as historical observations.

The supplied New York Fed PDF shows **2026Q2 selected and a July 31 data-flow
table**, despite being printed September 6. Its pages 2–6 identify additional
indicators. It is not evidence of the latest Q3 forecast. Its “Model update”
times are also not a release-time database. The [New York Fed documentation](https://www.newyorkfed.org/research/policy/nowcast/)
describes a Friday publication using data available by 10 a.m.; date-level
ALFRED cannot reproduce that exact intraday information set. We retain the
conservative prior-day rule for both sources' indicator ideas.

Source attachments were read without running workbook formulas/macros or
following embedded instructions. They remain unchanged and are not committed:

- `GDPTrackingModelDataAndForecasts(1).xlsx`, SHA-256
  `f0355aac1763d8c631139d3bb6db4c6c2a069f9f4e1e239bb748142d37c0fd00`.
- `New York Fed Staff Nowcast - FEDERAL RESERVE BANK of NEW YORK.pdf`, SHA-256
  `595c8a3159f0b3d9941d3ddd76e0f384ca04945149b475314e2e449de10f2625`.

## Additional series and economic meaning

The mappings are public FRED equivalents of indicators in the Fed references,
not a replication of either institution's proprietary/vendor panel or models.
The API's metadata was checked for ID, title, native frequency, units, seasonal
adjustment and latest observation. Import/export price indices are **NSA**;
we preserve that definition rather than applying ex-post seasonal adjustment.

Codes: **1** level, **2** first difference, **5** monthly (or quarterly) log
difference. Codes are fixed experimental choices, not estimates of the New
York Fed's exact transformations. Growth features remain natural log changes,
not GDP SAAR. Nominal spending, shipments and inventory levels are not mislabeled
as real GDP components; their own price indicators are separate inputs.

| Release category | Added series | Frequency / code |
|---|---|---|
| Retail sales | `RSAFS` advance retail and food-service sales | M / 5 |
| Personal income and outlays | `DSPIC96` real disposable personal income | M / 5 |
| Personal income and outlays | `PCEPI`, `PCEPILFE` headline/core PCE price indices | M / 5 |
| Consumer prices | `CPILFESL` core CPI | M / 5 |
| International trade | `BOPTEXP`, `BOPTIMP` nominal goods/services exports and imports | M / 5 |
| Trade prices | `IR`, `IQ` import/export price indices, NSA | M / 5 |
| Labor demand | `JTSJOL` job openings in thousands | M / 2 |
| Employment | `ADPMNUSNERSA` current ADP private payroll level, persons | M / 2 |
| Housing | `HSN1F` new single-family home sales | M / 5 |
| Construction | `TTLCONS` total construction spending | M / 5 |
| Durable goods | `DGORDER` new orders; `AMDMVS` shipments | M / 5 |
| Durable goods | `AMDMTI` inventory level | M / 5 |
| Manufacturing | `AMTMUO` unfilled orders, total manufacturing | M / 5 |
| Full inventory reports | `BUSINV`, `WHLSLRIMSA` business/wholesale inventories | M / 5 |
| Advance economic indicators, auxiliary | `AMINVTS`, `ARINVTS` advance wholesale/retail inventories | M / 1 |
| Vehicle sales | `TOTALSA` total vehicle sales, SAAR million units | M / 5 |
| Regional surveys | `GACDISA066MSFRBNY`, `GACDFSA066MSFRBPHI` New York/Philadelphia current general activity | M / 1 |
| National accounts, auxiliary | `A261RX1Q020SBEA` real GDI | Q / 5 |
| Productivity/costs, auxiliary | `ULCNFB` nonfarm business unit labor costs | Q / 5 |

Existing inputs already include payrolls/unemployment, headline CPI, industrial
production, housing starts/permits and the real PCE quantity index
`DPCERA3M086SBEA`. That quantity index covers real consumption's growth concept
without adding a duplicate real-dollar PCE predictor. It is not a dollar-level
replacement in national-accounting identities.

Two mapping traps were rejected: `AMDMIS` is an **inventory/shipments ratio**,
not inventory levels; `ADPWNUSNERSA` is **weekly**, not monthly. Current ADP
history begins in 2010, but its earliest retrieved ALFRED vintage is
**August 31, 2022**. The 2010 observation date does not establish 2010 availability.
No splice to the discontinued ADP methodology is made.

The advance inventory series are essential for the ragged edge: on this cutoff
`BUSINV` and `WHLSLRIMSA` stop in June, while July advance inventories were
available August 27. **The two advance series' current snapshots contain only
two usable months (June and July); older values are missing.** Their raw vintages
are collected, but they are excluded from the shared model panel. Attempting
ordinary growth-feature training made the DFM fail its minimum-history check.
They need a separately tested release-event history/advance-versus-full-report
definition; resurrecting withdrawn values or grafting advance estimates onto
the full-report series would misrepresent the current snapshot. `TOTALSA` August is
first available in ALFRED September 4, although the workbook mentions an auto
release on September 2. That earlier mention never backdates this FRED series.

## Gaps and historical eligibility

- **ISM manufacturing, employment, prices and services:** not admitted. The
  [St. Louis Fed removed ISM series from all FRED services](https://news.research.stlouisfed.org/2016/06/institute-for-supply-management-data-to-be-removed-from-fred/),
  and the queried IDs are unavailable. Historical reports or a licensed vintage
  archive would need a separately validated ingestion path. Regional surveys
  are distinct indicators, not silent ISM substitutes.
- **GDPNow subcomponent details:** no complete inventory valuation adjustments,
  government receipts/outlays, detailed equipment/service trade/price tables,
  or BEA component accounting system. No GDPNow replication claim is made.
- **Other frequencies and proprietary data:** daily financial data, weekly
  claims, existing-home sales and private survey expectations need separately
  sourced vintage coverage and frequency/release rules. They are not inferred
  from the workbook or filled from current FRED observations.
- **Quarterly auxiliaries:** stored and checked, but not passed to the full
  catalog run. The DFM currently requires monthly covariates plus quarterly
  GDP. The foundation monthly-slot adapter can retain a quarterly covariate as
  one channel, but that is a separate declared experiment.
- **Archive starts and unavailable series:** backfilling observations to 2005
  does not imply 2005 vintage coverage. For example, the retrieved earliest
  vintages of exports, durable inventories and the Empire survey are in 2010,
  2011 and 2014. A requested origin must pass actual snapshot admission. The
  full expanded universe is not admissible in 2019 because of ADP. A separate
  experiment explicitly excludes ADP before fitting; production code never
  silently drops it or substitutes older/current data.

## Refresh and independently verify

```bash
python scripts/sync_alfred_asof_store.py \
  --db data/realtime/nowcast.duckdb \
  --series-specs config/pit/nowcast_series.json \
  --observation_start 2005-01-01 --report_json results/sync_nowcast.json

python scripts/sync_alfred_asof_store.py \
  --db data/realtime/nowcast.duckdb \
  --series-specs config/pit/nowcast_auxiliary_series.json \
  --observation_start 2005-01-01 --report_json results/sync_auxiliary.json

python scripts/check_pit_freshness.py \
  --db data/realtime/nowcast.duckdb \
  --series-specs config/pit/nowcast_series.json \
  --origin 2026-09-06 --observation-start 2005-01-01 \
  --out results/freshness_20260906
```

Repeat the checker for the auxiliary specification, or pass an explicit merged
specification. A pre-existing output directory is refused. `FRED_API_KEY` comes
from the environment or `.env`; it is never retained in evidence. Exact IDs
cannot fall through alias resolution, and specification/API frequency mismatch
fails the sync. Sync has no ordinary-FRED fallback.

The checker makes separate `output_type=1`, `units=lin` queries with both
real-time bounds equal to the information date and observation end no later
than that date. It compares **all values in the declared observation range**,
including old revisions, missing/withdrawn observations and new releases.
Absence and a dot both mean no usable observation. Pagination must be complete;
failed or empty queries cannot report fresh. Raw sanitized requests, responses,
retrieval times and hashes are saved. The read-only checker never ingests its
clipped snapshot bounds as original release dates.

`all_fresh` means exact equality to ALFRED for the declared universe/range at
that cutoff. It cannot certify that every agency release has reached ALFRED,
that omitted releases are covered, or that a vintage existed at an earlier
intraday time. Compare publication calendars as well. In particular, never
equate “last refreshed today” with the latest usable observation month.

## Forecast and compare

Use the foundation run command with
`--series-specs config/pit/nowcast_series.json --covariates-from-specs`
and `--foundation-features monthly_slots`. GDP remains quarterly; each monthly
indicator contributes three month-specific values and three masks. With 41
admitted indicators this gives **246 covariate channels**, plus four GDP growth lags for
TabPFN bridge. Other catalog models retain their documented representations.

The monthly-slot representation preserves within-quarter order lost by a mean.
It does not eliminate the mismatch between fully observed historical training
quarters and partially observed nowcast quarters. A useful next research step
is training on explicitly reconstructed historical ragged-edge origins and
testing on held-out realized quarters. No accuracy improvement is claimed from
one current-quarter forecast or from proximity to GDPNow's estimate.

See `audit_nowcast_freshness_20260906.csv`, `audit_nowcast_2026q3.csv` and
the September 6 section of [AUDIT.md](../AUDIT.md) for measured outcomes.
