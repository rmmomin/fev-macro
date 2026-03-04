# BoE Forecast Evaluation Workflow

This project includes optional adapters for the Bank of England `forecast_evaluation` package.

Install optional dependencies:

```bash
pip install -r requirements-boe.txt
```

## 1) Export fev-macro outputs to BoE schema

Use either realtime OOS `predictions.csv` or evaluation `predictions_per_window.csv`.

```bash
python -m fev_macro.boe export \
  --predictions_csv results/realtime_oos/predictions.csv \
  --release_table_csv data/panels/gdpc1_releases_first_second_third.csv \
  --out_dir results/boe_export \
  --truth first \
  --variable GDPC1 \
  --metric levels \
  --forecast_value_col y_hat_level
```

Writes:
- `boe_forecasts.csv`
- `boe_outturns.csv`

## 2) Run BoE evaluation tables

`k=0` means evaluate against first-release truth along the BoE k-diagonal.

```bash
python -m fev_macro.boe eval \
  --forecasts_csv results/boe_export/boe_forecasts.csv \
  --outturns_csv results/boe_export/boe_outturns.csv \
  --k 0 \
  --benchmark_model naive_last \
  --out_dir results/boe_results
```

Writes:
- `accuracy.csv`
- `diebold_mariano.csv`
- `rolling_dm.csv`
- `fluctuation_dm.csv`
- optional `main_table.csv` when available

To evaluate later data releases, change `k` (for example `k=1` for second-release style evaluation, depending on release timing conventions).

## 3) Generate BoE diagnostics

```bash
python -m fev_macro.boe plots \
  --forecasts_csv results/boe_export/boe_forecasts.csv \
  --outturns_csv results/boe_export/boe_outturns.csv \
  --variable GDPC1 \
  --source naive_last \
  --metric levels \
  --frequency Q \
  --k 0 \
  --horizon 0 \
  --ma_window 4 \
  --out_dir results/boe_plots
```

Writes:
- `hedgehog.png`
- `error_density.png`
- `errors_across_time.png`

## Regime-shift checks

The `eval` command writes both `rolling_dm.csv` and `fluctuation_dm.csv`.
Use them to identify periods where relative performance vs the benchmark model breaks down over time.
