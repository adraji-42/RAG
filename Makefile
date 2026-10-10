VENV_PATH = $(shell uv run python3 -c "import sys; print(sys.prefix)")
install:
	@uv sync

run:
	@uv run python3 -m src

debug:
	@uv run python3 -m pdb -m src

lint:
	@flake8 . --exclude $(VENV_PATH),data/
	@mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude '($(VENV_PATH)|data)'

lint-strict:
	@flake8 . --exclude $(VENV_PATH),data/
	@mypy . --strict --exclude '($(VENV_PATH)|data)'

clean:
	@find . -name "__pycache__" -type d | xargs rm -rf
	@find . -name ".mypy_cache" -type d | xargs rm -rf
	@rm -rf $(VENV_PATH)

.PHONY: install run debug lint lint-strict clean