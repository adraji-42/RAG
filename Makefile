install:
	@uv sync

run:
	@uv run python3 -m src

debug:
	@uv run python3 -m pdb -m src

lint:
	@uv run flake8 . --exclude .venv,data/ || true
	@uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude '(.venv|data)'  || true

lint-strict:
	@uv run flake8 . --exclude .venv,data/ || true
	@uv run mypy . --strict --exclude '(.venv|data)' || true

clean:
	@find . -name "__pycache__" -type d | xargs rm -rf
	@find . -name ".mypy_cache" -type d | xargs rm -rf
	@rm -rf .venv

.PHONY: install run debug clean lint lint-strict
