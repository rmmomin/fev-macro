PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup inspect-data run report plot download-historical panel-md panel-qd panels

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

panels: panel-md panel-qd
