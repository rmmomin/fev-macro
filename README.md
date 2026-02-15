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

Benchmark mode uses a finalized dataset/task (`scripts/run_eval.py`), while realtime mode uses vintage panels and release-aware scoring (`scripts/run_realtime_oos.py`).

## Objective

This repository benchmarks multiple forecasting approaches on the same task definition and backtest protocol:

- Core benchmark target: `LOG_REAL_GDP` (log real GDP level).
- Evaluation: multi-horizon rolling windows (`h = 1, 2, 4`) with comparable metrics and leaderboard tables.
- Visualization: convert log-level forecasts to q/q SAAR growth for train/OOS plotting.

The harness is designed so models can be added modularly and compared fairly under identical windows, metrics, and data handling.

## Task Definition

- Dataset source: `autogluon/fev_datasets` with config `fred_qd_2025`.
- Target construction:
  - If target exists, use it directly.
  - Otherwise compute from real GDP level (e.g., `GDPC1`):
    - `level`: `y_t`
    - `log_level`: `log(y_t)`
    - `saar_growth`: `100 * ((y_t / y_{t-1})**4 - 1)`
- Covariate construction: FRED transform codes (`tcode` 1-7) are applied by default using historical FRED-QD metadata.
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
| `mixed_freq_dfm_md` | Mixed-frequency factor model using FRED-MD monthly panel | Target lags + factors from aggregated FRED-MD monthly data |
| `chronos2` | Zero-shot Chronos-2 foundation model adapter | Target history + available dynamic covariates passed as context/future known covariates |
| `ensemble_avg_top3` | Equal-weight ensemble | Drift + AutoARIMA + LocalTrendSSM predictions |
| `ensemble_weighted_top5` | Weighted ensemble | Drift + AutoARIMA + LocalTrendSSM + FactorPCAQD + SeasonalNaive predictions |

Notes:

- Models that use covariates consume all available task dynamic columns unless a model-specific cap/selection is applied.
- `rw_drift_log` and `ar4_growth` are defined in the realtime OOS module and are used by `scripts/run_realtime_oos.py` and `scripts/run_latest_vintage_forecast.py`.
- Registered but not in the default run list: `auto_ets`, `theta`.

## Real-Time Data Policy: Training vs Testing

The harness enforces a vintage-aware protocol:

1. Build the canonical evaluation dataset from `fred_qd_2025` (target + covariates), excluding 2020.
2. For each rolling window, identify the training cutoff date.
3. Select the latest historical FRED-QD vintage file with vintage month `<=` that cutoff.
4. Replace `past_data` (training history) with data from that selected vintage.
5. Keep `future_data` and OOS ground truth from `fred_qd_2025`.

This ensures models train only on information available at that historical point, while test scoring uses the finalized benchmark target path.

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

- finalized HF dataset construction (`build_real_gdp_target_series`)
- per-window historical-vintage training reconstruction (`HistoricalQuarterlyVintageProvider`)

CLI controls:

- `--disable_fred_transforms`
- `--fred_transform_vintage YYYY-MM`

### Strict vintage coverage

- Historical quarterly vintages available: `2018-05` to `2026-01`.
- With default request `num_windows=80`, strict coverage reduces effective windows to:
  - `h=1`: `24`
  - `h=2`: `23`
  - `h=4`: `21`

Use `--vintage_fallback_to_earliest` to keep earlier windows by falling back to the earliest available vintage.

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

Direct run with explicit output folder:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_eval.py \
  --results_dir results/log_real_gdp_excl2020_vintage_full \
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

- `results/log_real_gdp_excl2020_vintage_full/summaries.csv`
- Computed on February 15, 2026 from `inference_time_s` totals across tasks.

Top total compute methods:

| Model | Total inference seconds | Seconds per forecast |
|---|---:|---:|
| `xgboost` | 161.144 | 2.370 |
| `chronos2` | 105.505 | 1.552 |
| `ensemble_weighted_top5` | 34.802 | 0.512 |
| `ensemble_avg_top3` | 28.616 | 0.421 |
| `auto_arima` | 17.294 | 0.254 |
| `random_forest` | 16.606 | 0.244 |

Interpretation:

- Tree-based boosted model (`xgboost`) and foundation model inference (`chronos2`) are the main compute bottlenecks.
- Ensemble models are also expensive because they execute multiple member models per window.

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
