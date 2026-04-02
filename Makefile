.PHONY: docs docs-build docs-strict lint test check

docs:
	mkdocs serve

docs-build:
	mkdocs build

docs-strict:
	mkdocs build --strict

lint:
	ruff check .

test:
	pytest

check:
	python -m compileall bff
	ruff check .
	pytest
	mkdocs build --strict
