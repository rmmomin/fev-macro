# fev-macro

## Overview
`fev-macro` is a reproducible US real-GDP forecasting benchmark built on `fev` rolling-window evaluation. It combines historical FRED vintage panels, release-consistent GDP truth from ALFRED, and both classical and factor-style models. The repo is organized around one authoritative pipeline for panel building, evaluation, realtime OOS scoring, and latest-vintage 2025Q4 forecasting.

## What this repo does
1. Build vintage panels for FRED-QD and FRED-MD historical vintages.
2. Build processed panels using FRED transform codes + MD outlier/trimming semantics (code zip + fbi).
3. Build GDP release truth table from ALFRED and compute release-vintage q/q and q/q SAAR growth for first/second/third releases.
4. Run fev evaluation for both processed and unprocessed panels, with different target objectives (LL vs G) and release-truth mappings.
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
| `processed` | Transform-code + MD outlier/trimming processed covariates | `qoq_growth` (G) | q/q real GDP growth vs BEA-verified ALFRED release truth (`qoq_growth_alfred_*`) |

Details: [`docs/data_processing.md`](docs/data_processing.md)

## Models included
Core baselines and multivariate models include `naive_last`, `mean`, `drift`, `ar4`, `auto_arima`, `auto_ets`, `theta`, `local_trend_ssm`, `random_forest`, `xgboost`, `factor_pca_qd`, `mixed_freq_dfm_md`, `bvar_minnesota_8`, `bvar_minnesota_20`, `bvar_minnesota_growth_8`, `bvar_minnesota_growth_20`, and `chronos2` (plus optional ensemble variants).

Full model catalog: [`docs/models.md`](docs/models.md)

## Real-time evaluation policy
**By default, every rolling window trains on an as-of vintage (vintage-correct). Snapshot evaluation is blocked unless you explicitly pass `--allow_snapshot_eval`.** For processed `run_eval`, release truth defaults to BEA-verified ALFRED q/q growth from `data/panels/gdpc1_releases_first_second_third.csv`, using `qoq_growth_alfred_first_pct`, `qoq_growth_alfred_second_pct`, and `qoq_growth_alfred_third_pct` (`--eval_release_metric alfred_qoq`, `--target_transform qoq_growth`). Realtime SAAR truth remains available via `--eval_release_metric realtime_qoq_saar --target_transform saar_growth`.

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
- fev (Forecast EValuation library): <https://github.com/autogluon/fev>
