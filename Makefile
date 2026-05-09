# Author: Patrik Kiseda
# File: Makefile
# Description: Small local commands for starting Qdrant and checking if the app can connect to it.

.PHONY: qdrant-up qdrant-down qdrant-logs qdrant-check app-run test

qdrant-up:
	docker compose up -d qdrant

qdrant-down:
	docker compose down

qdrant-logs:
	docker compose logs -f qdrant

qdrant-check:
	uv run python scripts/qdrant_connectivity_check.py

app-run:
	uv run uvicorn app.main:app --app-dir src --reload

test:
	uv run python -m unittest discover -v -s tests -p "test_*.py"
