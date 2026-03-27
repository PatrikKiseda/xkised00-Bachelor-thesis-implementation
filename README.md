# RAG Thesis Implementation 

This repository contains the source code for the thesis implementation.

Current goal: making the system work reliably on `localhost` first (ingestion -> indexing -> retrieval -> answer generation), with a simple local UI and a retrieval contract that stays ready for later hybrid retrieval work.

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
  - dense retrieval, lexical retrieval, hybrid retrieval, ranking/fusion logic (e.g., RRF/MMR)

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
  - `chunks_fts` (FTS5 virtual table for lexical search)
- documents metadata fields now include:
  - `source_type` (`txt`/`md`/`pdf`)
  - `filename`
  - `size_bytes`

This creates the local metadata/job-tracking foundation for ingestion, dense retrieval hydration, and SQLite FTS5 lexical lookup.

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

- `POST /api/query/lexical`
  - runs lexical retrieval against SQLite FTS5
  - request body: `query`, optional `top_k` (default `5`)
  - response returns chunk references + lexical score + hydrated chunk content

- `POST /api/query/answer`
  - runs retrieval first, then generates a grounded answer from the retrieved chunks
  - request body: `query`, optional `top_k` (default `5`), optional `mode` (default `dense`), optional `include_context_in_prompt` (default `true`)
  - response returns `mode`, `query`, `answer`, and `sources`

- `POST /api/query/prompt-debug`
  - builds the final answer-generation prompt without calling the LLM
  - request body matches `POST /api/query/answer`
  - response returns `mode`, `query`, `include_context_in_prompt`, `prompt`, and `sources`

- `GET /`
  - serves a plain localhost HTML page for upload, prompt debug, answer generation, and source inspection

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
- lexical indexing policy:
  - no separate lexical indexing job is required because chunk content is mirrored into `chunks_fts` during chunk persistence
  - lexical retrieval queries `chunks_fts` and hydrates chunk metadata through SQLite joins
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

## Lexical query behavior

- endpoint: `POST /api/query/lexical`
- request body:
  - `query` (required, non-empty)
  - `top_k` (optional, `1..50`, default `5`)
- implementation:
  - lexical search runs through SQLite FTS5 using the existing `chunks_fts` table
  - free-text queries are normalized into quoted terms joined with `AND`
  - retrieval ordering uses `bm25(chunks_fts)` with deterministic tie-breaking on `chunks.id`
  - returned API scores are normalized into higher-is-better values derived from `bm25`
- response body:
  - `mode`, `query`
  - `hits`: list of `{chunk_id, document_id, chunk_index, score, content}`
- if the lexical index is empty, or the normalized query has no searchable tokens, returns `200` with `hits: []`

## Answer generation behavior

- retrieval and generation are separated:
  - dense retrieval and lexical retrieval are resolved through a shared retrieval contract
  - answer generation only receives hydrated retrieved chunks and does not call Qdrant or SQLite FTS5 directly
- endpoint: `POST /api/query/answer`
- request body:
  - `query` (required, non-empty)
  - `top_k` (optional, `1..50`, default `5`)
  - `mode` (optional, default `dense`)
  - `include_context_in_prompt` (optional, default `true`)
- response body:
  - `mode`, `query`, `answer`
  - `sources`: list of `{source_id, chunk_id, document_id, filename, chunk_index, score, content}`
- prompt construction:
  - uses numbered source ids like `[S1]`, `[S2]` when retrieved chunks are included in the prompt
- no-hit behavior:
  - if retrieval returns no chunks and `include_context_in_prompt=true`, the API still calls the LLM with a simple prompt that says there was no retrieved context
  - if retrieval returns no chunks, the response still returns `sources: []`
- prompt debug:
  - `POST /api/query/prompt-debug` returns the final prompt plus retrieved sources
  - if `include_context_in_prompt=false`, the final prompt is just the raw user query
- mode support today:
  - `dense` is implemented
  - `lexical` is implemented
  - `hybrid` is reserved in the request contract and currently returns `501 Not Implemented`

Prompt modes:

- `include_context_in_prompt=true` and retrieval returns hits:
  - grounded prompt with retrieved chunks
- `include_context_in_prompt=true` and retrieval returns no hits:
  - simple no-context prompt sent to the LLM
- `include_context_in_prompt=false`:
  - raw user query sent directly to the LLM

## Enable embedding service

To run API embeddings currently:

- `EMBEDDING_PROVIDER=openai`
- `EMBEDDING_MODEL=text-embedding-3-small` (or another supported OpenAI embedding model)
- `OPENAI_API_KEY=<real-provider-key>`
- `EMBEDDING_API_ENABLED=true`
- `QDRANT_VECTOR_SIZE=1536` (for `text-embedding-3-small`)

## Generation service and required env vars

For the current answer-generation flow:

- `LITELLM_MODEL=openai/gpt-5.4-mini`
- `OPENAI_API_KEY=<real-provider-key>`

Notes:

- the same `OPENAI_API_KEY` is used for both embeddings and answer generation when you run the OpenAI path
- test mode can still use deterministic local embeddings with `EMBEDDING_API_ENABLED=false`
- even if embeddings are local, answer generation still needs a working LiteLLM provider if you want the full end-to-end answer flow

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
6. Open the localhost UI:
   - `http://127.0.0.1:8000/`

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

# lexical query (after indexing)
curl -X POST http://127.0.0.1:8000/api/query/lexical \
  -H "Content-Type: application/json" \
  -d '{"query":"alpha beta","top_k":5}'

# answer generation (after indexing)
curl -X POST http://127.0.0.1:8000/api/query/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"What does the document say about alpha?","top_k":5,"mode":"dense","include_context_in_prompt":true}'

# lexical answer generation (after indexing)
curl -X POST http://127.0.0.1:8000/api/query/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"alpha beta","top_k":5,"mode":"lexical","include_context_in_prompt":true}'

# prompt debug without calling the LLM
curl -X POST http://127.0.0.1:8000/api/query/prompt-debug \
  -H "Content-Type: application/json" \
  -d '{"query":"What does the document say about alpha?","top_k":5,"mode":"dense","include_context_in_prompt":false}'
```

## Run the frontend locally

- start the API with `make app-run`
- open `http://127.0.0.1:8000/`
- use the page to:
  - upload a document
  - enter a query
  - choose `dense` or `lexical` retrieval mode
  - toggle whether retrieved context is included in the final prompt
  - inspect retrieved chunks
  - inspect the built prompt through the debug button
  - generate the final answer and review cited sources

## End-to-end localhost test flow

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Start Qdrant with `make qdrant-up`.
3. Run the API with `make app-run`.
4. Open `http://127.0.0.1:8000/`.
5. Upload a `.txt`, `.md`, or `.pdf` document.
6. Wait a moment for background indexing to finish.
7. Run a dense query or lexical query through the page, `POST /api/query/dense`, `POST /api/query/lexical`, or `POST /api/query/answer`.
8. Use the prompt-debug button or `POST /api/query/prompt-debug` to inspect the exact prompt.
9. Confirm the answer includes grounded citations and that `sources` match retrieved chunks.
10. Run tests with `make test`.

## Pipeline walkthrough

- A short file-by-file explanation of the whole pipeline lives in [docs/PIPELINE_WALKTHROUGH.md](/home/patrik/Desktop/skola/BP/prototypes/issue-13-lexical-search/docs/PIPELINE_WALKTHROUGH.md).

## Future opportunities / roadmap

- chunking strategy experiments:
  - compare character/word/token-based chunking
  - chunk size and overlap values, measure retrieval/answer quality and latency
- indexing pipeline:
  - in-process background tasks -> dedicated worker/queue model
  - better invalid input handling
  - add richer job payload and progress metadata for observability
- retrieval and ranking:
  - dense-only vs lexical-only vs hybrid retrieval benchmarking
  - large enough dataset probe for stronger statistics
- embedding adapter and provider behavior:
  - additional tests for different providers 
  - compare deterministic vectors vs API vectors 
- lexical retrieval follow-ups:
  - add punctuation-heavy and low-signal lexical query regression cases to document current FTS5 normalization behavior
  - compare alternative tokenization and lexical score normalization choices before hybrid fusion evaluation
