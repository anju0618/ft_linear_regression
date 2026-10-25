.PHONY: help install clean fclean train predict precision vision lint

train: ## Train the model and save theta.json
	uv run python3 train.py

predict: ## Predict the price from a mileage
	uv run python3 predict.py

precision: ## Show the accuracy (MSE, RMSE, MAE, R²)
	uv run python3 precision.py

vision: ## Plot the data and the regression line
	uv run python3 visualizer.py

lint: ## Run flake8 and mypy
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

install: ## Install dependencies (uv sync)
	uv sync

clean: ## Remove caches and generated files (theta.json)
	rm -f theta.json
	find . -name .venv -prune -o \( -name __pycache__ -o -name .mypy_cache \) \
		-type d -prune -exec rm -rf {} +

fclean: clean ## Remove caches, .venv and data.csv
	rm -rf .venv data.csv

help: ## Show this help
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'
