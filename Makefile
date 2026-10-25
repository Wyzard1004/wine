PYTHON ?= python3

.PHONY: install analyze test serve report

install:
	$(PYTHON) -m pip install -r requirements.txt

analyze:
	$(PYTHON) -m src.analysis

test:
	$(PYTHON) -m pytest -q

serve:
	$(PYTHON) -m http.server 8000 --bind 127.0.0.1

report:
	$(PYTHON) scripts/render_report.py
