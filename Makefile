.PHONY: dashboard test-dashboard

VENV_PYTHON := .venv/bin/python

$(VENV_PYTHON):
	python3 -m venv .venv

dashboard: $(VENV_PYTHON)
	@$(VENV_PYTHON) -c 'import plotly, streamlit' 2>/dev/null || $(VENV_PYTHON) -m pip install -r requirements-dashboard.txt
	$(VENV_PYTHON) -m streamlit run dashboard/app.py

test-dashboard: $(VENV_PYTHON)
	@$(VENV_PYTHON) -c 'import pytest' 2>/dev/null || $(VENV_PYTHON) -m pip install pytest
	$(VENV_PYTHON) -m pytest tests/test_dashboard_data.py
