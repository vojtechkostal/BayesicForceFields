.PHONY: docs lint test check

docs:
	mkdocs serve

lint:
	ruff check .

test:
	python -m pytest -q

check:
	python -m compileall -q bff
	ruff check .
	python -m pytest -q
	mkdocs build --strict
