# Data Artifacts

This directory contains both downloaded inputs and generated outputs.

## Downloaded / external inputs
- `data/historical/`: historical FRED vintage archives and extracted vintage CSVs
- `data/documentation/`: reference docs/code snapshots used for transform and outlier semantics

## Generated artifacts (rebuild anytime)
- `data/panels/*.parquet`: combined vintage panels (unprocessed + processed)
- `data/panels/gdpc1_releases_first_second_third.csv`: GDP release truth table
- `data/panels/gdpc1_release_validation_report.csv`: release-truth validation report
- `data/latest/*`: latest FRED API pulls
- `data/processed/*`: processed latest snapshots

## Repro steps
```bash
make download-historical
make panel-qd panel-md
make panel-qd-processed panel-md-processed
make build-gdp-releases
make fetch-latest
make process-latest
```

The repo treats generated artifacts as non-source outputs; they are intentionally gitignored.
