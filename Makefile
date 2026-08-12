install:
	uv sync

run:
	uv run -m python3 src

debug:
	uv run python3 -m pdb src

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null; true
	@echo "cleaned!"



lint:
	@echo "running mypy..."
	@mypy . --warn-return-any \
	        --warn-unused-ignores \
	        --ignore-missing-imports \
	        --disallow-untyped-defs \
	        --check-untyped-defs
	@echo "running flake8..."
	@flake8 .

.PHONY: clean