#* Variables
SHELL := /bin/bash
PYTHON := .venv/bin/python

.PHONY: help
help:
	@echo "Commands:"
	@echo "uv-download     : downloads and installs the uv package manager"
	@echo "uv-activate	   : activates the uv python environment"
	@echo "install         : installs required dependencies"
	@echo "install-dev     : installs the dev dependencies for the project"
	@echo "update-deps     : updates the dependencies and writes them to requirements.txt"
	@echo "check-style     : run checks on all files without fixing them."
	@echo "fix-style       : run checks on files and potentially modifies them."
	@echo "test            : run all tests."
	@echo "build-docs      : build mkdocs documentation."
	@echo "serve-docs      : serve documentation locally."
	@echo "deploy-docs     : deploy documentation to https://gsarti.com/wqe (gh-pages branch)"
	@echo "docs            : shortcut to build and serve generated documentation locally."
	@echo "clean           : cleans all unecessary files."

#* UV
uv-download:
	@echo "Downloading uv package manager..."
	@if [[ $OS == "Windows_NT" ]]; then \
		irm https://astral.sh/uv/install.ps1 | iex; \
	else \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv venv


.PHONY: uv-activate
uv-activate:
	@if [[ "$(OS)" == "Windows_NT" ]]; then \
		./uv/Scripts/activate.ps1 \
	else \
		source .venv/bin/activate; \
	fi

#* Installation

.PHONY: install
install:
	make uv-activate && uv pip install -r requirements.txt && uv pip install -e .

.PHONY: install-dev
install-dev:
	make uv-activate && uv pip install -r requirements-dev.txt && pre-commit install && pre-commit autoupdate


.PHONY: update-deps
update-deps:
	uv pip compile pyproject.toml -o requirements.txt
	uv pip compile --all-extras pyproject.toml -o requirements-dev.txt

#* Linting
.PHONY: check-style
check-style:
	$(PYTHON) -m ruff format --check --config pyproject.toml ./
	$(PYTHON) -m ruff check --no-fix --config pyproject.toml ./

.PHONY: fix-style
fix-style:
	$(PYTHON) -m ruff format --config pyproject.toml ./
	$(PYTHON) -m ruff check --config pyproject.toml ./

#* Linting
.PHONY: test
test:
	$(PYTHON) -m pytest -n auto -c pyproject.toml -vv


#* Docs
.PHONY: build-docs
build-docs:
	make uv-activate && mkdocs build

.PHONY: serve-docs
serve-docs:
	make uv-activate && mkdocs serve

.PHONY: deploy-docs
deploy-docs:
	make uv-activate && mkdocs gh-deploy

.PHONY: docs
docs: build-docs serve-docs

#* Remove
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: clean
clean: pycache-remove build-remove # docker-remove
