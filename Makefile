.PHONY: ingest dbt train serve lint test

ingest:
	uv run python -m rpe.ingest.healthkit

dbt:
	cd dbt && uv run dbt build --profiles-dir .

train:
	uv run python -m rpe.models.train_recovery

serve:
	uv run uvicorn rpe.serve.app:app --host 0.0.0.0 --port 8000

lint:
	uv run ruff check src tests

test:
	uv run pytest tests -q
