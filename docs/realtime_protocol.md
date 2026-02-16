# Realtime Protocol

## Truth definition
The benchmark KPI is always q/q SAAR real GDP growth scored against ALFRED release truth.

Release table file:
- `data/panels/gdpc1_releases_first_second_third.csv`

Primary truth columns:
- `qoq_saar_growth_realtime_first_pct`
- `qoq_saar_growth_realtime_second_pct`
- `qoq_saar_growth_realtime_third_pct`

## How realtime SAAR truth is built
For each quarter `q` and release stage `s` (first/second/third):
1. Take the ALFRED release date for `(q, s)`.
2. Choose one panel vintage timestamp from FRED-QD panel:
- `next`: earliest vintage timestamp `>= release_date`
- `prev`: latest vintage timestamp `<= release_date`
3. Pull `GDPC1(q)` and `GDPC1(q-1)` from that same selected vintage.
4. Compute realtime growth:
- `100 * ((GDPC1(q) / GDPC1(q-1))^4 - 1)`

This avoids revised-history leakage from mixing numerator and denominator across different benchmark/reindex scales.

## Validation rules
Validation report (`data/panels/gdpc1_release_validation_report.csv`) flags:
- `abs(g) > 15` post-2010, excluding `{2020Q2, 2020Q3}`
- `abs(g_t - g_{t-1}) > 12` post-2010, excluding transitions touching `{2020Q2, 2020Q3}`
- Panel-vs-release level ratio mismatches outside `[0.98, 1.02]` for 2018+

Optional `--fail_on_validate` exits nonzero when spike flags are present.

## Leakage policy
- Training windows use only historically available vintages.
- Truth comes from release-stage realtime SAAR columns.
- If realtime SAAR truth columns are missing, code emits loud warnings before fallback behavior.

## References
- ALFRED: <https://alfred.stlouisfed.org>
- Historical FRED vintages: <https://www.stlouisfed.org/research/economists/mccracken/fred-databases>
