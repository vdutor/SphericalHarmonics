.PHONY: help install docs format check test check-and-test


LIB_NAME = spherical_harmonics
TESTS_NAME = tests
LINT_NAMES = src/$(LIB_NAME) $(TESTS_NAME)
TYPE_NAMES = src/$(LIB_NAME)
SUCCESS='\033[0;32m'
UNAME_S = $(shell uname -s)

# the --per-file-ignores are to ignore "unused import" warnings in __init__.py files (F401)
# the F403 ignore in gpflux/__init__.py allows the `from .<submodule> import *`
LINT_FILE_IGNORES = "src/$(LIB_NAME)/__init__.py:F401,F403"


help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


install:  ## Install repo for developement
	@echo "\n=== pip install package with requirements =============="
	LAB_NO_BUILD=1 pip install backends
	@echo "\n=== pip install dev requirements ======================"
	pip install -r requirements.txt
	@echo "\n=== pip install test requirements ======================"
	pip install -r dev_requirements.txt
	pip install -e .


format: ## Formats code with `black` and `isort`
	@echo "\n=== Autoflake =============================================="
	autoflake --remove-all-unused-imports --recursive \
			--remove-unused-variables --in-place --exclude=__init__.py \
			$(LINT_NAMES)
	@echo "\n=== black =============================================="
	black $(LINT_NAMES)
	@echo "\n=== isort =============================================="
	isort $(LINT_NAMES)


check: ## Runs all static checks such as code formatting checks, linting, mypy
	@echo "\n=== flake8 (linting)===================================="
	flake8 --statistics --exclude=.ipynb_checkpoints
	@echo "\n=== black (formatting) ================================="
	black --check --diff $(LINT_NAMES)
	@echo "\n=== isort (formatting) ================================="
	isort --check --diff $(LINT_NAMES)
	@echo "\n=== mypy (static type checking) ========================"
	mypy $(TYPE_NAMES)

test: ## Run unit and integration tests with pytest
	pytest -v --tb=short --durations=10 $(TESTS_NAME)

check-and-test: check test  ## Run pytest and static tests
