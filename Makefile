PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup download-historical panel-qd panel-md panel-qd-processed panel-md-processed build-gdp-releases eval-unprocessed-standard eval-processed-standard eval-processed-snapshot realtime-oos-processed fetch-latest process-latest latest-forecast-processed plot-2025q4 eval-unprocessed-smoke eval-processed-smoke eval-unprocessed-full eval-processed-full

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download-historical:
	$(PY) scripts/download_historical_vintages.py

panel-qd:
	$(PY) scripts/build_qd_vintage_panel.py

panel-md:
	$(PY) scripts/build_md_vintage_panel.py

panel-qd-processed:
	$(PY) scripts/build_qd_vintage_panel.py --mode processed

panel-md-processed:
	$(PY) scripts/build_md_vintage_panel.py --mode processed

build-gdp-releases:
	$(PY) scripts/build_gdp_releases.py

eval-unprocessed-standard:
	$(PY) scripts/run_eval_unprocessed.py --profile standard --results_dir results/unprocessed_standard

eval-processed-standard:
	$(PY) scripts/run_eval_processed.py --profile standard --results_dir results/processed_standard

eval-processed-snapshot:
	$(PY) scripts/run_eval_processed.py --profile smoke --disable_historical_vintages --allow_snapshot_eval --results_dir results/processed_snapshot_debug

realtime-oos-processed:
	$(PY) scripts/run_realtime_oos.py --mode processed --select_models_from_leaderboard

fetch-latest:
	$(PY) scripts/fetch_latest_fred_api.py --output_dir data/latest

process-latest:
	$(PY) scripts/process_fred_latest.py --output_dir data/processed

latest-forecast-processed:
	$(PY) scripts/run_latest_vintage_forecast.py --mode processed

plot-2025q4:
	$(PY) scripts/plot_2025q4_forecast_selected_models.py

eval-unprocessed-smoke:
	$(PY) scripts/run_eval_unprocessed.py --profile smoke --results_dir results/unprocessed_smoke

eval-processed-smoke:
	$(PY) scripts/run_eval_processed.py --profile smoke --results_dir results/processed_smoke

eval-unprocessed-full:
	$(PY) scripts/run_eval_unprocessed.py --profile full --results_dir results/unprocessed_full

eval-processed-full:
	$(PY) scripts/run_eval_processed.py --profile full --results_dir results/processed_full
