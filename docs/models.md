# Models

This file documents the model registry used by evaluation (`scripts/run_eval_unprocessed.py`, `scripts/run_eval_processed.py`) and realtime OOS (`scripts/run_realtime_oos.py`).

## Registry highlights
- Baselines: `naive_last`, `mean`, `drift`, `seasonal_naive`
- AR/statsforecast: `ar4`, `auto_arima`, `auto_ets`, `theta`
- ML/state-space: `random_forest`, `xgboost`, `local_trend_ssm`
- Multivariate macro: `bvar_minnesota_*`, `factor_pca_qd`, `mixed_freq_dfm_md`
- Foundation model adapter: `chronos2`
- Ensembling: `ensemble_avg_top3`, `ensemble_weighted_top5`

Processed-standard profiles include: `ar4`.

## New/important variants
| Model | Description | Fallback behavior |
|---|---|---|
| `ar4` | Univariate AR(4) via `statsmodels.AutoReg` | Reverts to naive last-value forecast if not enough history or fit fails |
