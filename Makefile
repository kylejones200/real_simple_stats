.PHONY: help install install-dev test test-cov lint format type-check clean build upload docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package in development mode with all dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=real_simple_stats --cov-report=html --cov-report=term

lint:  ## Run linting with ruff (matches CI)
	ruff check .

lint-legacy:  ## Run linting with flake8 (legacy, use lint instead)
	flake8 real_simple_stats/ tests/

format:  ## Format code with ruff-format (matches CI)
	ruff format .

format-legacy:  ## Format code with black (legacy, use format instead)
	black real_simple_stats/ tests/

format-check:  ## Check code formatting
	ruff format --check .

type-check:  ## Run type checking (matches CI)
	mypy .

quality:  ## Run all quality checks (matches CI)
	make format-check
	make lint
	make type-check
	make test

ci:  ## Run all CI checks locally (matches GitHub Actions CI)
	@echo "Running CI checks..."
	ruff check .
	mypy .
	pytest -q --maxfail=1
	@echo "All CI checks passed!"

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python -m build

upload:  ## Upload to PyPI (requires twine)
	twine upload dist/*

upload-test:  ## Upload to Test PyPI
	twine upload --repository testpypi dist/*

docs:  ## Build documentation
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built in docs/build/html/"

docs-serve:  ## Serve documentation locally
	@echo "Serving documentation locally..."
	cd docs/build/html && python -m http.server 8000

docs-clean:  ## Clean documentation build
	@echo "Cleaning documentation build..."
	cd docs && make clean

setup-dev:  ## Set up development environment
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && make install-dev
	@echo "Development environment set up! Activate with: source venv/bin/activate"
