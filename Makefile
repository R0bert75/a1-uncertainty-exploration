# A1 — When Do Randomized Value Estimates Buy Exploration?
# Reproducibility entry points. `make figures` rebuilds every figure from logs/*.csv ALONE.

PYTHON ?= python
LOGS   ?= logs
FIGS   ?= figures

.DEFAULT_GOAL := help

.PHONY: help env test lint smoke dummy figures audit clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env:  ## Create/refresh the pinned CPU environment (torch from the CPU wheel index)
	$(PYTHON) -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt

lint:  ## ruff lint
	ruff check src analysis audits tests

test:  ## Run the smoke tests
	pytest

dummy:  ## Regenerate the dummy smoke CSV (schema-correct synthetic data)
	$(PYTHON) analysis/make_dummy_logs.py --out $(LOGS)/dummy_smoke.csv

figures: $(LOGS)/dummy_smoke.csv  ## Rebuild every figure from logs/*.csv ALONE
	$(PYTHON) analysis/make_figures.py --logs $(LOGS) --out $(FIGS)

# Bootstrap the dummy CSV on demand so `make figures` works on a fresh clone.
$(LOGS)/dummy_smoke.csv:
	$(PYTHON) analysis/make_dummy_logs.py --out $@

audit:  ## Run the C13 configuration-identity audit
	$(PYTHON) audits/c13_audit.py --configs $(LOGS) --out audits/c13

smoke: test figures audit  ## Full local smoke check (tests + figures + audit)
	@echo "smoke OK"

clean:  ## Remove generated figures (logs are the source of truth; kept)
	rm -f $(FIGS)/*.png
