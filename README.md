# fev-macro

Reproducible macro-forecasting benchmark harness built on `fev` rolling backtests, focused on US real GDP.

## Quickstart

```bash
pip install -r requirements.txt
make download-historical
make panel-md panel-qd
PYTHONPATH=src .venv/bin/python scripts/run_eval.py --models naive_last drift --num_windows 5 --horizons 1
PYTHONPATH=src .venv/bin/python scripts/run_realtime_oos.py --smoke_run
PYTHONPATH=src .venv/bin/python scripts/run_latest_vintage_forecast.py --vintage latest --target_quarter 2025Q4
PYTHONPATH=src .venv/bin/python scripts/plot_2025q4_forecast_selected_models.py --input_csv results/realtime_latest_vintage_forecast.csv
```

Benchmark mode (`scripts/run_eval.py`) defaults to release-table GDP truth, while realtime mode uses vintage panels and release-aware scoring (`scripts/run_realtime_oos.py`).

## Objective

This repository benchmarks multiple forecasting approaches on the same task definition and backtest protocol:

- Core benchmark target: realtime GDP q/q SAAR growth from release-table truth.
- Evaluation: multi-horizon rolling windows (`h = 1, 2, 3, 4`) with comparable metrics and leaderboard tables.
- Visualization: inspect predicted q/q SAAR growth paths directly.

The harness is designed so models can be added modularly and compared fairly under identical windows, metrics, and data handling.

## Task Definition

- Base benchmark dataset source: `data/panels/gdpc1_releases_first_second_third.csv`.
- Evaluation truth source (default): `data/panels/gdpc1_releases_first_second_third.csv` using realtime SAAR columns:
  - `qoq_saar_growth_realtime_first_pct`
  - `qoq_saar_growth_realtime_second_pct`
  - `qoq_saar_growth_realtime_third_pct`
- Historical vintage source (training windows): `data/panels/fred_qd_vintage_panel.parquet` (or raw monthly vintage CSVs under `data/historical/qd/`).
- Target construction:
  - Default benchmark target uses release-table realtime SAAR truth directly (`realtime_qoq_saar`).
  - Optional level mode uses release-stage GDP levels (`first/second/third/latest`) transformed to:
    - `level`: `y_t`
    - `log_level`: `log(y_t)`
    - `saar_growth`: `100 * ((y_t / y_{t-1})**4 - 1)`
- Covariate construction: benchmark task datasets are release-table-only (univariate scaffold), while historical-vintage adaptation continues to use local FRED-QD vintage data.
- Backtest engine: `fev.Task.iter_windows()` (expanding windows).
- Default metric: `RMSE`.
- Exclusion rule: year `2020` is removed from training/evaluation data.

## Model Set And Feature Usage

`scripts/run_eval.py` default full model list:

- `naive_last`
- `mean`
- `drift`
- `seasonal_naive`
- `random_normal`
- `random_uniform`
- `random_permutation`
- `random_forest`
- `xgboost`
- `local_trend_ssm`
- `bvar_minnesota_8`
- `bvar_minnesota_20`
- `factor_pca_qd`
- `mixed_freq_dfm_md`
- `ensemble_avg_top3`
- `ensemble_weighted_top5`
- `auto_arima`
- `chronos2`

### Model summary

| Model | What it is | Features used |
|---|---|---|
| `naive_last` | Random-walk/last-observation baseline | Target history only |
| `rw_drift_log` | Log-space random walk with constant drift, exponentiated back to levels | Target history only (requires strictly positive levels) |
| `mean` | Historical mean baseline | Target history only |
| `drift` | Linear drift from first to last point | Target history only |
| `seasonal_naive` | Seasonal repeat (quarterly season length = 4) | Target history only |
| `random_normal` | Gaussian random forecast from in-sample moments | Target history only |
| `random_uniform` | Uniform random forecast over in-sample range | Target history only |
| `random_permutation` | Randomly permuted historical values | Target history only |
| `auto_arima` | StatsForecast AutoARIMA | Target history only (univariate) |
| `auto_ets` | StatsForecast AutoETS exponential smoothing | Target history only (univariate) |
| `local_trend_ssm` | Unobserved Components local trend/cycle | Target history only (univariate) |
| `random_forest` | AR + exogenous random forest, recursive multi-step | Target lags + ragged-edge FRED-QD covariates + monthly FRED-MD vintage features from `data/panels/fred_md_vintage_panel.parquet` (quarterly aggregated by vintage) |
| `xgboost` | Gradient-boosted AR + exogenous model, recursive multi-step | Target lags + ragged-edge FRED-QD covariates + monthly FRED-MD vintage features from `data/panels/fred_md_vintage_panel.parquet` (quarterly aggregated by vintage) |
| `bvar_minnesota_8` | Minnesota-style shrinkage BVAR (~8 total vars including GDP) | Target + selected macro covariates (~7) from FRED-QD |
| `bvar_minnesota_20` | Minnesota-style shrinkage BVAR (~20 total vars including GDP) | Target + selected macro covariates (~19) from FRED-QD |
| `factor_pca_qd` | Quarterly PCA factor regression | Target lags + PCA factors from up to ~80 FRED-QD covariates |
| `mixed_freq_dfm_md` | Mixed-frequency factor model using local vintage-panel covariates | Target lags + factors from latest local QD vintage panel (`data/panels/fred_qd_vintage_panel.parquet`) |
| `chronos2` | Zero-shot Chronos-2 foundation model adapter | Target history + available dynamic covariates passed as context/future known covariates |
| `ensemble_avg_top3` | Equal-weight ensemble | Drift + AutoARIMA + LocalTrendSSM predictions |
| `ensemble_weighted_top5` | Weighted ensemble | Drift + AutoARIMA + LocalTrendSSM + FactorPCAQD + SeasonalNaive predictions |

Notes:

- Models that use covariates consume all available task dynamic columns unless a model-specific cap/selection is applied.
- `rw_drift_log` and `ar4_growth` are defined in the realtime OOS module and are used by `scripts/run_realtime_oos.py` and `scripts/run_latest_vintage_forecast.py`.
- Registered but not in the default run list: `auto_ets`, `theta`.

## Real-Time Data Policy: Training vs Testing

The harness enforces a vintage-aware protocol:

1. Build the canonical task frame from `gdpc1_releases_first_second_third.csv` (quarter scaffold).
2. Set task target values from release-table realtime q/q SAAR truth.
3. By default, build three evaluation slices (`first`, `second`, `third`) from:
   - `qoq_saar_growth_realtime_first_pct`
   - `qoq_saar_growth_realtime_second_pct`
   - `qoq_saar_growth_realtime_third_pct`
4. Exclude configured years (default: 2020).
5. For each rolling window, identify the training cutoff date.
6. Select the latest historical FRED-QD vintage file with vintage month `<=` that cutoff.
7. Replace `past_data` (training history) with data from that selected vintage.
8. Keep `future_data` and OOS ground truth from the release-table task dataset.

This ensures models train only on information available at that historical point, while scoring uses explicit GDP release truth rather than finalized revised GDP.

## FRED Transform Codes (FredMDQD.jl Port)

This repo ports the transformation logic from [`enweg/FredMDQD.jl`](https://github.com/enweg/FredMDQD.jl) into `src/fev_macro/fred_transforms.py` and applies it during dataset construction:

- `1`: `x`
- `2`: `Δx`
- `3`: `Δ²x`
- `4`: `log(x)`
- `5`: `Δlog(x)`
- `6`: `Δ²log(x)`
- `7`: `Δ(x_t / x_{t-1} - 1)`

By default, transform codes are loaded from the latest file under `--historical_qd_dir` and applied to covariates for both:

- per-window historical-vintage training reconstruction (`HistoricalQuarterlyVintageProvider`)

### Strict vintage coverage

- Historical quarterly vintages available: `2018-05` to `2026-01`.
- Default benchmark request is `num_windows=100` with horizons `h=1,2,3,4`.
- Default benchmark behavior enables earliest-vintage fallback (`--vintage_fallback_to_earliest`), and effective windows are reduced automatically per horizon when needed to keep windows feasible.

Use `--no-vintage_fallback_to_earliest` to enforce strict historical-vintage coverage.

## Historical Vintage Download

Download and extract both FRED-MD and FRED-QD historical vintage archives:

```bash
make download-historical
```

Equivalent script:

```bash
PYTHONPATH=src .venv/bin/python scripts/download_historical_vintages.py
```

Default extraction root: `data/historical/`, organized as:

- `data/historical/qd/` for quarterly vintages
- `data/historical/md/` for monthly vintages

## Build Vintage Panels

Build separate combined panels across all vintages (each row includes `vintage` and `timestamp`):

```bash
make panel-md
make panel-qd
```

Outputs:

- `data/panels/fred_md_vintage_panel.parquet`
- `data/panels/fred_qd_vintage_panel.parquet`

## Processed FRED Datasets

This repository also includes scripts that apply the documented FRED-MD/FRED-QD preprocessing rules from the McCracken/Ng MATLAB code and the `fbi` R package implementation:

- transform codes (`tcode` 1-7) applied column-wise
- MD outlier rule: `abs(x - median) > 10 * IQR`
- MD factor-style trimming: drop first 2 transformed rows (per vintage slice)

### Single-vintage processing (2026m1)

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_fred_2026m1.py
```

Default inputs:

- `data/historical/md/vintages_1999_2026/2026-01.csv`
- `data/historical/qd/vintages_2018_2026/FRED-QD_2026m1.csv`

Default outputs:

- `data/processed/fred_md_2026m1_processed.csv`
- `data/processed/fred_qd_2026m1_processed.csv`
- `data/processed/fred_2026m1_processing_manifest.json`

### Vintage-panel processing (by vintage date)

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_fred_vintage_panels.py
```

Default inputs:

- `data/panels/fred_md_vintage_panel.parquet`
- `data/panels/fred_qd_vintage_panel.parquet`

Default outputs:

- `data/panels/fred_md_vintage_panel_process.parquet`
- `data/panels/fred_qd_vintage_panel_process.parquet`
- `data/panels/fred_vintage_panel_process_manifest.json`

## Real-Time OOS (First-Release Scoring)

This repo includes a vintage-correct rolling OOS evaluator in:

- `src/fev_macro/realtime_oos.py`
- `scripts/run_realtime_oos.py`

### Vintage-to-origin mapping

- For each release-quarter `q`, origin date is:
  - `origin_date = first_release_date(q) - 1 day`
- Each FRED-QD vintage `YYYY-MM` is mapped to:
  - `asof_date = last business day of YYYY-MM` (`BMonthEnd`)
- Training vintage selection:
  - latest vintage with `asof_date <= origin_date`
- Leakage guard:
  - training uses only quarters `<= q-1` at that origin
  - covariates can use ragged-edge rows from the selected vintage for forecast quarters (target GDP remains unobserved and excluded from training targets)

### First-release target definition

- Truth level for quarter `k`:
  - `y_true_level_first(k) = first_release(k)`
- SAAR growth truth (Mode A):
  - `g_true(k) = 100 * ((y_true_level_first(k) / y_true_level_first(k-1))**4 - 1)`
- Forecast SAAR growth:
  - `g_hat(k) = 100 * ((y_hat(k) / y_hat(k-1))**4 - 1)`
  - for `h=1`, `y_hat(k-1)` is the last observed GDPC1 in the selected origin vintage
  - for `h>=2`, `y_hat(k-1)` is the model’s previous-step forecast

### Run command

Smoke run (3 models, ~10 origins):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_realtime_oos.py --smoke_run
```

Full run:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_realtime_oos.py \
  --models naive_last rw_drift_log ar4_growth \
  --horizons 1 2 3 4 \
  --output_dir results/realtime_oos
```

Monthly-origin run (every vintage month-end, scored on first/second/third releases, bucketed by months-to-first-release):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_realtime_oos.py \
  --origin_schedule monthly \
  --models naive_last rw_drift_log ar4_growth \
  --horizons 1 2 3 4 \
  --output_dir results/realtime_oos_monthly
```

For monthly runs, metrics are computed within each `months_to_first_release_bucket` and aggregated at the `target_quarter` level first, so repeated monthly origins do not over-weight the same target quarter inside a bucket.

By default, realtime OOS scoring is filtered to target quarters `>= 2018Q1` (`--min_target_quarter 2018Q1`).

Outputs:

- `predictions.csv`
  - `origin_quarter, target_quarter, horizon, y_hat_level, y_true_level_first, g_hat_saar, g_true_saar, ...`
- `metrics.csv`
  - `model, horizon, release_stage, months_to_first_release_bucket, rmse, mae, rmse_log_level, mae_log_level, rel_rmse, sample_count`

## Latest-Vintage One-Shot Forecast

For a single nowcast/forecast using all information in one chosen vintage (for example `2026-01`) and a requested target quarter:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_latest_vintage_forecast.py \
  --vintage 2026-01 \
  --target_quarter 2025Q4 \
  --output_csv results/realtime_latest_v2026m1_2025Q4_forecasts.csv
```

Behavior:

- Finds the latest observed quarter in the selected vintage.
- Trains each model on observed target history through that quarter.
- Uses covariate data through the requested target quarter when available in the same vintage (ragged-edge nowcast inputs; target values remain unavailable/unseen).
- Forecasts forward to the requested target quarter.
- Writes a CSV including level, q/q %, and SAAR forecast columns.

You can also run default settings with:

```bash
make latest-forecast
```

## Plot 2025Q4 Forecast Comparison

Create a sorted column chart (lowest to highest) for selected models' 2025Q4 q/q SAAR forecasts:

```bash
PYTHONPATH=src .venv/bin/python scripts/plot_2025q4_forecast_selected_models.py \
  --input_csv results/realtime_latest_v2026m1_2025Q4_forecasts.csv \
  --target_quarter 2025Q4 \
  --models rw_drift_log auto_ets drift theta chronos2 \
  --output_png results/forecast_2025Q4_qoq_saar_rw_autoets_drift_theta_chronos2.png
```

Output:

- `results/forecast_2025Q4_qoq_saar_rw_autoets_drift_theta_chronos2.png`

## Running The Benchmark

Setup:

```bash
make setup
```

Inspect data:

```bash
make inspect-data
```

Run default full benchmark:

```bash
make run
```

Equivalent direct command (uses default full model list):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_eval.py
```

Default benchmark CLI settings:

- `--horizons 1 2 3 4`
- `--num_windows 100`
- `--metric RMSE`
- `--target_transform saar_growth`
- `--eval_release_metric realtime_qoq_saar`
- `--eval_release_csv data/panels/gdpc1_releases_first_second_third.csv`
- `--eval_release_stages first second third`
- `--models` full 18-model default list (see section above)
- `--vintage_fallback_to_earliest` (enabled by default; use `--no-vintage_fallback_to_earliest` for strict coverage)

When requested windows are infeasible for a stage/horizon, the harness automatically reduces to the largest valid trailing window count.

### Latest benchmark snapshot (February 15, 2026)

Command run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/run_eval.py
```

Observed run details:

- Target: `LOG_REAL_GDP` (`saar_growth`), excluding year `2020`.
- Evaluation truth: `data/panels/gdpc1_releases_first_second_third.csv` realtime SAAR columns (`first/second/third`).
- Built task datasets:
  - `results/log_real_gdp_dataset_first.parquet`
  - `results/log_real_gdp_dataset_second.parquet`
  - `results/log_real_gdp_dataset_third.parquet`
- Training vintage source: historical FRED-QD CSV vintages at `data/historical/qd/vintages_2018_2026` (93 monthly files from `2018-05` to `2026-01`).
- Effective windows reduced from requested `100` to:
  - `first`: `h=1:25`, `h=2:24`, `h=3:23`, `h=4:22`
  - `second`: `h=1:26`, `h=2:25`, `h=3:24`, `h=4:23`
  - `third`: `h=1:25`, `h=2:24`, `h=3:23`, `h=4:22`

Top leaderboard models (`results/leaderboard.csv`):

- `mean`: `win_rate=1.000000`, `skill_score=0.237981`
- `chronos2`: `win_rate=0.941176`, `skill_score=0.159133`
- `ensemble_weighted_top5`: `win_rate=0.784314`, `skill_score=0.062204`

Direct run with explicit output folder:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/run_eval.py \
  --results_dir results/log_real_gdp_realtime_default \
  --seed 0
```

Generate leaderboard/pairwise from existing summaries:

```bash
make report
```

Plot train history + OOS forecasts for top-5 models (q/q SAAR derived from log GDP):

```bash
make plot
```

## Compute Hotspots (Latest Snapshot)

Snapshot source:

- `results/summaries.csv`
- Computed on February 15, 2026 from `inference_time_s` totals across tasks.

Top total compute methods:

| Model | Total inference seconds | Seconds per forecast |
|---|---:|---:|
| `chronos2` | 22.983 | 0.080 |
| `ensemble_weighted_top5` | 21.318 | 0.075 |
| `ensemble_avg_top3` | 20.735 | 0.073 |
| `auto_arima` | 18.035 | 0.063 |
| `random_forest` | 7.820 | 0.027 |
| `local_trend_ssm` | 1.428 | 0.005 |

Interpretation:

- Foundation model inference (`chronos2`) and ensemble models are the main compute bottlenecks.
- In this default release-CSV run, tree models are no longer the dominant cost contributors.

## Output Artifacts

Primary outputs in each run directory:

- `summaries.jsonl`
- `summaries.csv`
- `leaderboard.csv`
- `pairwise.csv`
- optional plots and OOS forecast CSVs

## Repository Structure

- `src/fev_macro/data.py`: dataset loading, target construction, vintage provider
- `src/fev_macro/tasks.py`: horizon-specific task creation
- `src/fev_macro/eval.py`: window loop and summary generation
- `src/fev_macro/models/`: model implementations and registry
- `src/fev_macro/report.py`: leaderboard/pairwise generation
- `scripts/run_eval.py`: main benchmark CLI
- `scripts/run_realtime_oos.py`: vintage-correct realtime OOS CLI
- `scripts/run_latest_vintage_forecast.py`: one-shot latest-vintage forecast CLI
- `scripts/plot_top_models.py`: top-model OOS plot generation
- `scripts/plot_2025q4_forecast_selected_models.py`: selected-model 2025Q4 forecast bar chart
- `scripts/download_historical_vintages.py`: vintage downloader
- `Makefile`: setup/run/report helpers
