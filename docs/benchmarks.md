# Benchmarks

## Standard runs
```bash
make eval-unprocessed-standard
make eval-processed-standard
```

## Profile defaults
- `smoke`: `num_windows=10`, `horizons={1,4}`, models `{naive_last, drift, auto_arima}`
- `standard`:
  - unprocessed LL: `num_windows=60`, `horizons={1,2,4}`, excludes `chronos2` and ensemble models
  - processed G (`saar_growth`, `alfred_qoq_saar` truth, first release by default): `num_windows=40`, `horizons={1,2,4}`, excludes `chronos2` and ensemble models
- `full`: full/default long-run profile

## Realtime OOS
Leaderboard-driven processed realtime run:
```bash
make realtime-oos-processed
```

This reads `results/processed_standard/leaderboard.csv`, selects models above the skill threshold, persists the selection JSON, and appends posthoc top-3/top-5 growth-space ensembles.
