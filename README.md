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
  - document processing pipeline -> extraction/chunking/embedding orchestration
  - this is the “input side” of the RAG system

- `embeddings/`
  - embedding adapter abstraction and provider registry
  - OpenAI API-first client + deterministic runtime mode for tests/local CI

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
- `scripts/qdrant_connectivity_check.py` used to verify the local Qdrant instance is reachable
- `Makefile` commands to start/stop/log/check Qdrant

## SQLite metadata schema 

The app initializes the local SQLite metadata schema during startup (`create_app`).

- schema initializer: `src/app/storage/sqlite_schema.py`
- configurable DB path: `SQLITE_PATH` (default: `./data/app.db`)
- configurable upload storage dir: `STORAGE_DIR` (default: `./data/uploads`)
- tables:
  - `documents`
  - `chunks`
  - `jobs`
  - `chunks_fts` (FTS5 virtual table for lexical search placeholders)
- documents metadata fields now include:
  - `source_type` (`txt`/`md`/`pdf`)
  - `filename`
  - `size_bytes`

This creates the local metadata/job-tracking foundation for upcoming ingestion and reindex flows.

## API endpoints

- `GET /api/health`
  - runtime status snapshot for app + Qdrant + SQLite initialization state

- `POST /api/documents/upload`
  - accepts multipart file upload (`file`)
  - saves file to `STORAGE_DIR`
  - stores document metadata row in SQLite with status `pending`
  - creates indexing job row in SQLite (`jobs` table)
  - schedules background indexing pipeline (`extract -> chunk -> embed -> chunk persistence`)
  - supported extensions: `.txt`, `.md`, `.pdf`

- `GET /api/documents`
  - returns list of stored documents from SQLite metadata

- `GET /api/jobs`
  - returns persisted indexing job records
  - supports filters: `document_id` and `limit`
  - used to observe pending/running/success/fail job transitions and persisted errors

- `POST /api/query/dense`
  - runs dense retrieval against Qdrant
  - request body: `query`, optional `top_k` (default `5`)
  - response returns chunk references + similarity score + hydrated chunk content

### Upload response payload (`201`)

- `id`
- `job_id`
- `filename`
- `source_type`
- `status`
- `storage_path`
- `created_at`

### Upload errors

- `400` - uploaded file is empty
- `415` - unsupported file extension
- `422` - extraction failure for supported file type (for example malformed PDF)
- `500` - unexpected storage/metadata persistence failure

Extraction failures for supported file types (for example malformed PDF) are now persisted as failed indexing jobs (`status=fail`, `error_message`) and can be inspected via `GET /api/jobs`.

## Extraction behavior

- `.txt` and `.md` are decoded as UTF-8 text
- `.pdf` is extracted using `pypdf`
- extracted text is normalized by:
  - line ending normalization (`\r\n` / `\r` -> `\n`)
  - trimming trailing whitespace on each line
  - trimming outer leading/trailing whitespace

## Indexing and chunking behavior

- indexing pipeline runs in background after upload:
  - `documents.status`: `pending -> running -> success/fail`
  - `jobs.status`: `pending -> running -> success/fail`
- chunking is recursive and character-based
  - default `CHUNK_SIZE_CHARS=1000`
  - default `CHUNK_OVERLAP_CHARS=150`
  - deterministic chunk IDs: `<document_id>:<zero-padded chunk_index>` (example `doc-1:000005`)
- chunk records are persisted into:
  - `chunks` table (metadata/content)
  - `chunks_fts` table (FTS5 lexical index content mirror)
- embedding generation is executed per chunk via adapter interface:
  - provider/model come from `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL`
  - if `EMBEDDING_API_ENABLED=true`, provider registry uses API client (OpenAI in this prototype)
  - if `EMBEDDING_API_ENABLED=false`, deterministic non-API client is used for tests/dev
  - successful vectors are persisted into Qdrant using minimal payload refs (`chunk_id`, `document_id`, `chunk_index`)
- embedding failure policy:
  - partial embedding failures keep job `success` and write warning stats into `jobs.payload_json`
  - if all chunk embeddings fail, job is marked `fail` with persisted payload/error summary
- dense indexing policy:
  - deterministic vector schema is enforced via required `QDRANT_VECTOR_SIZE`
  - Qdrant collection uses cosine distance in v1
  - if Qdrant upsert fails after SQLite chunk persistence, job/document are marked `fail`

## Dense query behavior

- endpoint: `POST /api/query/dense`
- request body:
  - `query` (required, non-empty)
  - `top_k` (optional, `1..50`, default `5`)
- response body:
  - `mode`, `query`
  - `hits`: list of `{chunk_id, document_id, chunk_index, score, content}`
- if the dense index is empty, returns `200` with `hits: []`

## Enable embedding service

To run API embeddings currently:

- `EMBEDDING_PROVIDER=openai`
- `EMBEDDING_MODEL=text-embedding-3-small` (or another supported OpenAI embedding model)
- `OPENAI_API_KEY=<real-provider-key>`
- `EMBEDDING_API_ENABLED=true`
- `QDRANT_VECTOR_SIZE=1536` (for `text-embedding-3-small`)

### Local usage

1. Start Qdrant:
   - `make qdrant-up`
2. Check connectivity:
   - `make qdrant-check`
3. View logs (optional):
   - `make qdrant-logs`
4. Stop Qdrant:
   - `make qdrant-down`
5. Run app:
   - `make app-run`

### Quick API examples

```bash
# health
curl http://127.0.0.1:8000/api/health

# upload txt/md/pdf
curl -X POST http://127.0.0.1:8000/api/documents/upload \
  -F "file=@./example.txt"

# list uploaded documents
curl http://127.0.0.1:8000/api/documents

# list indexing jobs
curl "http://127.0.0.1:8000/api/jobs?limit=20"

# dense query (after indexing)
curl -X POST http://127.0.0.1:8000/api/query/dense \
  -H "Content-Type: application/json" \
  -d '{"query":"alpha beta","top_k":5}'
```

## Future opportunities / roadmap

- chunking strategy experiments:
  - compare character/word/token-based chunking
  - chunk size and overlap values, measure retrieval/answer quality and latency
- indexing pipeline:
  - in-process background tasks -> dedicated worker/queue model
  - better invalid input handling
  - add richer job payload and progress metadata for observability
- retrieval and ranking:
  - dense-only vs hybrid retrieval benchmarking
  - large enough dataset probe for stronger statistics
- embedding adapter and provider behavior:
  - additional tests for different providers 
  - compare deterministic vectors vs API vectors 
