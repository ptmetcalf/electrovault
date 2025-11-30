VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

.PHONY: help venv deps install lint format test test-cov coverage api frontend-install frontend-dev frontend-build ingest-sample dev

help: ## List available make commands
	@echo "Available make commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' Makefile | sort | awk 'BEGIN {FS=":.*##"} {printf "  %-18s %s\n", $$1, $$2}'

venv: ## Create virtual environment at $(VENV)
	python3 -m venv $(VENV)

deps: ## Create venv (if needed) and install Python requirements only
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install: deps frontend-install ## Install Python + frontend deps (full setup)

lint: ## Run Python (ruff) and frontend (eslint) linters
	$(VENV)/bin/ruff check .
	cd frontend && npm run lint

format: ## Format Python code (black + ruff --fix)
	$(VENV)/bin/black .
	$(VENV)/bin/ruff check --fix .

test: ## Run pytest
	$(VENV)/bin/pytest -q

test-cov: ## Run pytest with coverage report
	$(VENV)/bin/pytest --cov=photo_brain --cov-report=term

coverage: ## Run pytest with coverage summary (missing lines)
	$(VENV)/bin/pytest --cov=photo_brain --cov-report=term-missing

api: ## Start FastAPI server on port 8000
	$(PYTHON) -m uvicorn photo_brain.api.http_api:app --host 0.0.0.0 --port 8000

frontend-install: ## Install frontend dependencies
	cd frontend && npm install

frontend-dev: ## Run Vite dev server
	cd frontend && npm run dev

frontend-build: ## Build frontend assets
	cd frontend && npm run build

ingest-sample: ## Ingest sample phototest directory
	$(PYTHON) scripts/ingest.py $(PWD)/phototest

dev: ## Run API + Vite dev server together (Ctrl+C stops both)
	@echo "Starting API on :8000 and Vite dev server..."
	@bash -c 'trap "kill 0" INT TERM EXIT; \
		($(PYTHON) -m uvicorn photo_brain.api.http_api:app --host 0.0.0.0 --port 8000 --reload &); \
		(cd frontend && npm run dev -- --host); \
		wait'
