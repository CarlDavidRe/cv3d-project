.PHONY: dashboard

VENV_PYTHON := .venv/bin/python

$(VENV_PYTHON):
	python3 -m venv .venv

dashboard: $(VENV_PYTHON)
	@$(VENV_PYTHON) -c 'import streamlit' 2>/dev/null || $(VENV_PYTHON) -m pip install -r requirements-dashboard.txt
	$(VENV_PYTHON) -m streamlit run dashboard/app.py
