# fev-macro

## Overview
`fev-macro` is a reproducible US real-GDP forecasting benchmark built on `fev` rolling-window evaluation. It combines historical FRED vintage panels, release-consistent GDP truth from ALFRED, and both classical and factor-style models. The repo is organized around one authoritative pipeline for panel building, evaluation, realtime OOS scoring, and latest-vintage 2025Q4 forecasting.

## What this repo does
1. Build vintage panels for FRED-QD and FRED-MD historical vintages.
2. Build processed panels using FRED transform codes + MD outlier/trimming semantics (code zip + fbi).
3. Build GDP release truth table from ALFRED and compute q/q SAAR growth for first/second/third releases robustly.
4. Run fev evaluation for both processed and unprocessed panels, with different target objectives (LL vs G), but KPI always q/q SAAR.
5. Produce a 2025Q4 one-shot forecast using latest FRED API pulls and processed-mode top models.

## Quickstart
```bash
make setup
make download-historical
make panel-qd panel-md
make panel-qd-processed panel-md-processed
make build-gdp-releases
make eval-unprocessed-standard
make eval-processed-standard
```

## Modes: processed vs unprocessed
| Mode | Covariates | Training objective | KPI for comparison |
|---|---|---|---|
| `unprocessed` | Raw vintage covariates | `log_level` (LL) | q/q SAAR real GDP growth vs ALFRED release truth |
| `processed` | Transform-code + MD outlier/trimming processed covariates | `saar_growth` (G) | q/q SAAR real GDP growth vs ALFRED release truth |

Details: [`docs/data_processing.md`](docs/data_processing.md)

## Models included
Core baselines and multivariate models include `naive_last`, `drift`, `auto_arima`, `local_trend_ssm`, `random_forest`, `xgboost`, `factor_pca_qd`, `mixed_freq_dfm_md`, and BVAR growth variants.

Newly emphasized variants:
- `ar4`

Full model catalog: [`docs/models.md`](docs/models.md)

## Real-time evaluation policy
Evaluation is always scored against release-consistent q/q SAAR GDP truth from `data/panels/gdpc1_releases_first_second_third.csv`, using `qoq_saar_growth_realtime_first_pct`, `qoq_saar_growth_realtime_second_pct`, and `qoq_saar_growth_realtime_third_pct` when available. For each quarter/release stage, realtime truth is built from one as-of panel vintage to avoid revised-history leakage from mixed-vintage denominators around reindex/rebenchmark events. Protocol details: [`docs/realtime_protocol.md`](docs/realtime_protocol.md).

## Latest-vintage one-shot forecast + 2025Q4 comparison
```bash
make fetch-latest && make process-latest && make latest-forecast-processed
make plot-2025q4
```

## Outputs + repo layout
- `data/historical/`: downloaded FRED vintage archives and extracted CSVs
- `data/panels/`: generated vintage panels and GDP release truth table
- `data/latest/`, `data/processed/`: latest API pulls and processed latest snapshots
- `results/`: evaluation outputs, leaderboards, realtime OOS metrics, and forecast plots
- `scripts/`: core pipeline entrypoints
- `scripts/dev/`: non-core development utilities
- `docs/`: deeper protocol/model/data notes

## References
- FRED databases historical vintages: <https://www.stlouisfed.org/research/economists/mccracken/fred-databases>
- FRED databases code zip (transform codes + trimming/outliers): <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/fred-databases_code.zip?sc_lang=en&hash=82A2EEE1EF3498C0820EB2212531D895>
- fbi library: <https://github.com/cykbennie/fbi>
- ALFRED: <https://alfred.stlouisfed.org>
- FRED API: <https://api.stlouisfed.org/fred>
