# RAG Thesis Implementation

This repository contains the main implementation for the bachelor thesis RAG prototype. It is kept small on purpose: local app, local storage, Qdrant for vectors, and a simple UI for trying uploads and queries.

## Sections

- Quick start
- Repository layout
- `src/app` structure
- API endpoints
- Uploads
- Tests
- Extra branch and temporary deployment

## Quick start

Requirements:

- Python `3.11+`
- Docker
- Docker Compose
- `uv`

From a new deployment, start with the environment file:

```bash
cp .env.example .env
```

The main variables to set in `.env` are:

- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `QDRANT_VECTOR_SIZE`
- `LITELLM_MODEL`
- `EMBEDDING_PROVIDER`
- `EMBEDDING_MODEL`
- `EMBEDDING_API_ENABLED`
- `OPENAI_API_KEY`

### Embeddings

Embeddings are configured through `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_ENABLED`, `OPENAI_API_KEY`, and `QDRANT_VECTOR_SIZE`.

For the default OpenAI setup in `.env.example`, `QDRANT_VECTOR_SIZE=1536` matches `text-embedding-3-small`.

### Generation

Answer generation is configured through `LITELLM_MODEL` and `OPENAI_API_KEY`.

### Run locally

Start Qdrant:

```bash
make qdrant-up
```

Check that the app can reach Qdrant:

```bash
make qdrant-check
```

Start the app:

```bash
make app-run
```

Then open the local app at:

```text
http://127.0.0.1:8000/
```

The health endpoint is available at:

```text
http://127.0.0.1:8000/api/health
```

## Repository layout

- `.env.example` - example local environment variables
- `docker-compose.yml` - local Docker Compose setup with Qdrant
- `Makefile` - short commands for Qdrant, running the app, and tests
- `pyproject.toml` - Python project metadata and dependencies
- `uv.lock` - locked dependency versions
- `src/` - application source code
- `tests/` - unit and API tests
- `scripts/` - helper scripts, including the Qdrant connectivity check
- `docs/` - extra development and pipeline notes
- `data/` - local SQLite database and uploaded files

## `src/app` structure

- `api/` - FastAPI routes and request handling
- `core/` - shared app settings and small infrastructure helpers
- `ingestion/` - document extraction, chunking, and indexing pipeline
- `embeddings/` - embedding adapter and provider setup
- `retrieval/` - dense, lexical, and hybrid retrieval service
- `generation/` - prompt building and answer generation service
- `storage/` - SQLite and Qdrant storage code
- `ui/` - simple local HTML interface

## API endpoints

- `GET /` - local UI
- `GET /api/health` - app, Qdrant, and SQLite health snapshot
- `POST /api/documents/upload` - upload a document for ingestion
- `GET /api/documents` - list stored documents
- `GET /api/jobs` - list indexing jobs
- `POST /api/query/dense` - dense vector retrieval
- `POST /api/query/lexical` - lexical SQLite FTS retrieval
- `POST /api/query/hybrid` - combined dense and lexical retrieval
- `POST /api/query/answer` - retrieve sources and generate an answer
- `POST /api/query/prompt-debug` - build the final prompt without calling the model

## Uploads

The app ingests PDF, TXT, and Markdown files.

Upload errors are intentionally simple:

- `400` - empty file
- `415` - unsupported extension
- `422` - extraction failed
- `500` - storage or metadata failure

## Tests

Run the test suite with:

```bash
make test
```

## Extra branch and temporary deployment

The main repo is here:

```text
https://github.com/PatrikKiseda/xkised00-Bachelor-thesis-implementation
```

There is also a separate branch with code for longer conversations and lightweight deployment through Cloudflare and Caddy(DISCLAIMER: This is not a part of the submission, it is supposed to demonstrate the functionality of the system):

```text
https://github.com/PatrikKiseda/xkised00-Bachelor-thesis-implementation/commits/feat/lightweight-web-deployment
```

For now(8.4.2026-31.6.2026), the running app is hosted on a private service temporarily available here:

```text
https://bp.patrikkiseda.com/
```
