# Models

This file documents the model registry used by evaluation (`scripts/run_eval_unprocessed.py`, `scripts/run_eval_processed.py`) and realtime OOS (`scripts/run_realtime_oos.py`).

## Registry highlights
- Baselines: `naive_last`, `mean`, `drift`, `seasonal_naive`
- AR/statsforecast: `ar4`, `auto_arima`, `auto_ets`, `theta`
- ML/state-space: `random_forest`, `xgboost`, `local_trend_ssm`
- Neural (opt-in): `lstm_univariate`, `lstm_multivariate` (requires `torch>=2.2.0`)
- Multivariate macro: `bvar_minnesota_*`, `factor_pca_qd`, `mixed_freq_dfm_md`
- Foundation model adapter: `chronos2`
- Ensembling: `ensemble_avg_top3`, `ensemble_weighted_top5`
- GDPNow benchmarks:
  - `atlantafed_gdpnow_final_pre_release` (intentionally advantaged, latest update strictly before BEA first release)
  - `atlantafed_gdpnow_asof_window_cutoff` (fair as-of-window variant, may emit missing predictions early in quarter)
  - legacy aliases `atlantafed_gdpnow` / `gdpnow` map to final-pre-release behavior

Processed-standard profiles include: `ar4`.

GDPNow models are opt-in and are not included in default standard/full model lists. Include them explicitly with `--models ...` when needed.

## New/important variants
| Model | Description | Fallback behavior |
|---|---|---|
| `ar4` | Univariate AR(4) via `statsmodels.AutoReg` | Reverts to naive last-value forecast if not enough history or fit fails |
| `lstm_univariate` | Per-window, per-item LSTM trained on target history only | Uses drift/naive fallback for short history or unstable fits |
| `lstm_multivariate` | Per-window, per-item LSTM trained on target + covariates, optionally conditioning on known future covariates | Uses drift/naive fallback for short history or unstable fits |

Example (opt-in LSTM run):

```bash
python scripts/run_eval_processed.py --models lstm_univariate lstm_multivariate --num_windows 20
```
