.PHONY: install clean fclean train predict vision lint

train:
	uv run python3 train.py

predict:
	uv run python3 predict.py

vision:
	uv run python3 visualizer.py

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

install:
	uv sync

clean:
	find . -name .venv -prune -o \( -name __pycache__ -o -name .mypy_cache \) \
		-type d -prune -exec rm -rf {} +

fclean: clean
	rm -rf .venv
