# Models

This file documents the model registry used by evaluation (`scripts/run_eval_unprocessed.py`, `scripts/run_eval_processed.py`) and realtime OOS (`scripts/run_realtime_oos.py`).

## Registry highlights
- Baselines: `naive_last`, `mean`, `drift`, `seasonal_naive`
- AR/statsforecast: `ar4`, `auto_arima`, `auto_ets`, `theta`
- ML/state-space: `random_forest`, `xgboost`, `local_trend_ssm`
- Multivariate macro: `bvar_minnesota_*`, `factor_pca_qd`, `mixed_freq_dfm_md`
- Nowcast variants: `nyfed_nowcast_mqdfm`, `ecb_nowcast_mqdfm`
- Foundation model adapter: `chronos2`
- Ensembling: `ensemble_avg_top3`, `ensemble_weighted_top5`

Processed-standard profiles include: `ar4`, `nyfed_nowcast_mqdfm`, `ecb_nowcast_mqdfm`.

## New/important variants
| Model | Description | Fallback behavior |
|---|---|---|
| `ar4` | Univariate AR(4) via `statsmodels.AutoReg` | Reverts to naive last-value forecast if not enough history or fit fails |
| `nyfed_nowcast_mqdfm` | NY Fed-inspired mixed-frequency block DFM approximation using `DynamicFactorMQ` | Uses available MD/QD intersection only; if insufficient usable inputs or fit fails, falls back to naive last-value forecast |
| `ecb_nowcast_mqdfm` | ECB-toolbox-inspired mixed-frequency DFM approximation using `DynamicFactorMQ` | Uses available monthly coverage; if insufficient usable inputs or fit fails, falls back to naive last-value forecast |

## Important interpretation note
NY Fed / ECB variants here are benchmarking approximations under a unified pipeline, not reproductions of official institutional nowcasting systems.

## Source inspirations
- NY Fed Staff Nowcast codebase: <https://github.com/FRBNY-RG/New-York-Nowcast>
- ECB nowcasting toolbox: <https://github.com/baptiste-meunier/Nowcasting_toolbox>
