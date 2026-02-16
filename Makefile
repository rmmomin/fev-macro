PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup inspect-data run report plot download-historical panel-md panel-qd panel-md-processed panel-qd-processed panels panels-processed process-fred-latest latest-forecast latest-forecast-processed realtime-oos-processed eval-unprocessed-standard eval-processed-standard eval-unprocessed-smoke eval-processed-smoke eval-unprocessed-full eval-processed-full

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

inspect-data:
	$(PY) scripts/inspect_data.py

run:
	$(PY) scripts/run_eval.py

report:
	$(PY) scripts/report_results.py --summaries results/summaries.jsonl

plot:
	$(PY) scripts/plot_top_models.py

download-historical:
	$(PY) scripts/download_historical_vintages.py

panel-md:
	$(PY) scripts/build_md_vintage_panel.py

panel-qd:
	$(PY) scripts/build_qd_vintage_panel.py

panel-md-processed:
	$(PY) scripts/build_md_vintage_panel.py --mode processed

panel-qd-processed:
	$(PY) scripts/build_qd_vintage_panel.py --mode processed

panels: panel-md panel-qd

panels-processed: panel-md-processed panel-qd-processed

process-fred-latest:
	$(PY) scripts/process_fred_latest.py

latest-forecast:
	$(PY) scripts/run_latest_vintage_forecast.py

latest-forecast-processed:
	$(PY) scripts/run_latest_vintage_forecast.py --mode processed

realtime-oos-processed:
	$(PY) scripts/run_realtime_oos.py --mode processed

eval-unprocessed-standard:
	$(PY) scripts/run_eval_unprocessed.py --profile standard --results_dir results/unprocessed_standard

eval-processed-standard:
	$(PY) scripts/run_eval_processed.py --profile standard --results_dir results/processed_standard

eval-unprocessed-smoke:
	$(PY) scripts/run_eval_unprocessed.py --profile smoke --results_dir results/unprocessed_smoke

eval-processed-smoke:
	$(PY) scripts/run_eval_processed.py --profile smoke --results_dir results/processed_smoke

eval-unprocessed-full:
	$(PY) scripts/run_eval_unprocessed.py --profile full --results_dir results/unprocessed_full

eval-processed-full:
	$(PY) scripts/run_eval_processed.py --profile full --results_dir results/processed_full
