# fev-macro

Reproducible macro-forecasting benchmark harness built on `fev` rolling backtests, focused on US real GDP.

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
| `mean` | Historical mean baseline | Target history only |
| `drift` | Linear drift from first to last point | Target history only |
| `seasonal_naive` | Seasonal repeat (quarterly season length = 4) | Target history only |
| `random_normal` | Gaussian random forecast from in-sample moments | Target history only |
| `random_uniform` | Uniform random forecast over in-sample range | Target history only |
| `random_permutation` | Randomly permuted historical values | Target history only |
| `auto_arima` | StatsForecast AutoARIMA | Target history only (univariate) |
| `local_trend_ssm` | Unobserved Components local trend/cycle | Target history only (univariate) |
| `random_forest` | AR + exogenous random forest, recursive multi-step | Target lags + lagged/known dynamic covariates from FRED-QD task columns |
| `xgboost` | Gradient-boosted AR + exogenous model, recursive multi-step | Target lags + lagged/known dynamic covariates from FRED-QD task columns |
| `bvar_minnesota_8` | Minnesota-style shrinkage BVAR (~8 total vars including GDP) | Target + selected macro covariates (~7) from FRED-QD |
| `bvar_minnesota_20` | Minnesota-style shrinkage BVAR (~20 total vars including GDP) | Target + selected macro covariates (~19) from FRED-QD |
| `factor_pca_qd` | Quarterly PCA factor regression | Target lags + PCA factors from up to ~80 FRED-QD covariates |
| `mixed_freq_dfm_md` | Mixed-frequency factor model using FRED-MD monthly panel | Target lags + factors from aggregated FRED-MD monthly data |
| `chronos2` | Zero-shot Chronos-2 foundation model adapter | Target history + available dynamic covariates passed as context/future known covariates |
| `ensemble_avg_top3` | Equal-weight ensemble | Drift + AutoARIMA + LocalTrendSSM predictions |
| `ensemble_weighted_top5` | Weighted ensemble | Drift + AutoARIMA + LocalTrendSSM + FactorPCAQD + SeasonalNaive predictions |

Notes:

- Models that use covariates consume all available task dynamic columns unless a model-specific cap/selection is applied.
- Registered but not in the default run list: `auto_ets`, `theta`.

## Real-Time Data Policy: Training vs Testing

The harness enforces a vintage-aware protocol:

1. Build the canonical evaluation dataset from `fred_qd_2025` (target + covariates), excluding 2020.
2. For each rolling window, identify the training cutoff date.
3. Select the latest historical FRED-QD vintage file with vintage month `<=` that cutoff.
4. Replace `past_data` (training history) with data from that selected vintage.
5. Keep `future_data` and OOS ground truth from `fred_qd_2025`.

This ensures models train only on information available at that historical point, while test scoring uses the finalized benchmark target path.

### Strict vintage coverage

- Historical quarterly vintages available: `2018-05` to `2024-12`.
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

Default extraction root: `data/historical/`.

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
- `scripts/plot_top_models.py`: top-model OOS plot generation
- `scripts/download_historical_vintages.py`: vintage downloader
- `Makefile`: setup/run/report helpers
