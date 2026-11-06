.PHONY: check lint type test api web
check: lint type test
lint:
	uv run ruff check . && uv run ruff format --check .
fmt:
	uv run ruff check --fix . && uv run ruff format .
type:
	uv run mypy
test:
	uv run pytest
api:
	uv run uvicorn api.main:app --reload --port 8000
web:
	cd web && pnpm dev
