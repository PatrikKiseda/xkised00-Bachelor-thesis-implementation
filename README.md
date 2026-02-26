# RAG Thesis Implementation 

This repository contains the source code for the thesis implementation.

Current goal: making the system work reliably on `localhost` first (ingestion -> indexing -> retrieval -> answer generation). Possibility of UI and remote deployment.

## Structure of the repository

The structure is intentionally simple and modular:

- keep thesis-relevant components separated by responsibility (retrieval, storage, generation, etc.)
- modularity
- avoid complex abstractions hiding the core RAG logic
- allow scalability

Thus allowing direct mapping of the modules to the components presented in the thesis.

## Why Docker (already now)

We use Docker early mainly for infrastructure consistency.

Reasons:

- `Qdrant` runs the same way on every machine
- allows to reproduce the behavior on different machines
- later deployment is easier because infrastructure is already containerized

Currently, Docker is primarily for services (especially the vector database).

## Repository layout 

- `.env.example` - example environment variables for local development
- `docker-compose.yml` - local infrastructure services (Qdrant)
- `Makefile` - common local commands (run, test, compose, etc.)
- `pyproject.toml` - Python project metadata and dependencies managed with `uv`
- `uv.lock` - locked dependency versions for reproducible environments
- `src/` - application source code
- `tests/` - tests 
- `scripts/` - helper scripts 
- `docs/` - technical notes /  implementation documentation
- `data/` - local data (uploads, SQLite DB, temporary files)

## `src/app/...` structure

The `src/app` directory is split into modules with different responsibilities that directly map to the RAG pipeline:

- `api/`
  - API endpoints and request handling
  - contains no direct logic processing

- `core/`
  - Utilities and shared infrastructure code
  - examples: configuration loading, logging, helpers

- `ingestion/`
  - document processing pipeline -> extraction/chunking
  - this is the “input side” of the RAG system

- `retrieval/`
  - dense retrieval, hybrid retrieval, ranking/fusion logic (e.g., RRF/MMR)

- `generation/`
  - prompt construction and LLM provider calls (LiteLLM wrapper)

- `storage/`
  - adapters for persistence layers (Qdrant - ve, SQLite - documents themselves and chunk metadata, FTS5 virtual table in SQLite for efficient search)
  - isolates database-specific code from the rest of the app

- `models/`
  - request/response schemas and typed models

## Qdrant

- `docker-compose.yml` with a `qdrant` service
- `scripts/check_qdrant_connectivity.py` used to verify the local Qdrant instance is reachable
- `Makefile` commands to start/stop/log/check Qdrant

### Local usage

1. Start Qdrant:
   - `make qdrant-up`
2. Check connectivity:
   - `make qdrant-check`
3. View logs (optional):
   - `make qdrant-logs`
4. Stop Qdrant:
   - `make qdrant-down`