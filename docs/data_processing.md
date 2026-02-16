# Data Processing

## Objective
Produce one authoritative pair of panel artifacts per frequency:
- Unprocessed: `data/panels/fred_qd_vintage_panel.parquet`, `data/panels/fred_md_vintage_panel.parquet`
- Processed: `data/panels/fred_qd_vintage_panel_processed.parquet`, `data/panels/fred_md_vintage_panel_processed.parquet`

Canonical processed naming is `*_vintage_panel_processed.parquet`.

## Pipeline
1. Download historical vintage archives (`make download-historical`).
2. Build unprocessed panels (`make panel-qd`, `make panel-md`).
3. Build processed panels (`make panel-qd-processed`, `make panel-md-processed`).
4. Build GDP release truth (`make build-gdp-releases`).

## Processing semantics
- QD/MD transform codes follow FRED database code semantics.
- MD processing includes outlier/trimming behavior aligned with the published reference code path.
- Legacy `*_process.parquet` files are only read as compatibility fallback in selected loaders; they are not produced by the core pipeline and not referenced by Makefile/README.

## Latest data path
- Pull current snapshots via FRED API (`make fetch-latest`).
- Process snapshots into `data/processed/` (`make process-latest`).
- Use processed latest snapshots for one-shot forecast publication (`make latest-forecast-processed`).

## Sources
- Historical vintages: <https://www.stlouisfed.org/research/economists/mccracken/fred-databases>
- FRED database transform/outlier code zip: <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/fred-databases_code.zip?sc_lang=en&hash=82A2EEE1EF3498C0820EB2212531D895>
- `fbi` package reference: <https://github.com/cykbennie/fbi>
- FRED API: <https://api.stlouisfed.org/fred>
