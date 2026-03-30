"""
Author: Patrik Kiseda
File: tests/test_answer_api.py
Description: API tests for answer generation, prompt debug, and localhost UI.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")
os.environ.setdefault("QDRANT_COLLECTION", "documents")
os.environ.setdefault("QDRANT_VECTOR_SIZE", "8")
os.environ.setdefault("LITELLM_MODEL", "openai/gpt-5.4-mini")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("EMBEDDING_API_ENABLED", "false")

from app.core.settings import Settings
from app.main import create_app
from app.storage.document_repository import insert_document
from app.storage.indexing_repository import ChunkUpsert, replace_document_chunks
from app.storage.qdrant_store import ChunkVector, DenseSearchHit, QdrantConnectionStatus


class _InMemoryDenseStore:
    def __init__(self) -> None:
        self._vectors: list[ChunkVector] = []
        self.dense_hits_override: list[DenseSearchHit] | None = None

    def check_connection(self) -> QdrantConnectionStatus:
        return QdrantConnectionStatus(reachable=True)

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        return None

    def upsert_chunk_vectors(self, *, collection_name: str, vectors: list[ChunkVector]) -> None:
        self._vectors = vectors

    def search_dense(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[DenseSearchHit]:
        if self.dense_hits_override is not None:
            return self.dense_hits_override[:limit]

        return [
            DenseSearchHit(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                chunk_index=item.chunk_index,
                score=1.0 - (index * 0.01),
            )
            for index, item in enumerate(self._vectors[:limit])
        ]


class _RecordingGenerationClient:
    model = "openai/gpt-5.4-mini"

    def __init__(self, response_text: str = "Grounded answer [S1]") -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, float]] = []

    def generate_text(self, *, prompt: str, temperature: float) -> str:
        self.calls.append((prompt, temperature))
        return self.response_text


def _build_settings(sqlite_path: str, storage_dir: str, **overrides: object) -> Settings:
    payload = {
        "app_name": "test-app",
        "qdrant_url": "http://test-qdrant:6333",
        "qdrant_collection": "documents",
        "qdrant_vector_size": 8,
        "sqlite_path": sqlite_path,
        "storage_dir": storage_dir,
        "chunk_size_chars": 120,
        "chunk_overlap_chars": 20,
        "litellm_model": "openai/gpt-5.4-mini",
        "embedding_provider": "local",
        "embedding_model": "text-embedding-3-small",
        "embedding_api_enabled": False,
    }
    payload.update(overrides)
    return Settings(**payload)


class TestAnswerApi(unittest.TestCase):
    def test_answer_endpoint_returns_answer_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient("Grounded answer [S1]")
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 30), "text/plain")},
                )
                answer_response = client.post(
                    "/api/query/answer",
                    json={"query": "What does the note mention?", "top_k": 3, "mode": "dense"},
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(answer_response.status_code, 200)
            payload = answer_response.json()
            self.assertEqual(payload["mode"], "dense")
            self.assertEqual(payload["answer"], "Grounded answer [S1]")
            self.assertGreater(len(payload["sources"]), 0)
            self.assertEqual(payload["sources"][0]["filename"], "notes.txt")
            self.assertTrue(payload["sources"][0]["content"])
            self.assertEqual(len(generation_client.calls), 1)

    def test_prompt_debug_omits_context_text_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient()
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 30), "text/plain")},
                )
                debug_response = client.post(
                    "/api/query/prompt-debug",
                    json={
                        "query": "What does the note mention?",
                        "top_k": 3,
                        "mode": "dense",
                        "include_context_in_prompt": False,
                    },
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(debug_response.status_code, 200)
            payload = debug_response.json()
            self.assertEqual(payload["prompt"], "What does the note mention?")
            self.assertGreater(len(payload["sources"]), 0)
            self.assertEqual(len(generation_client.calls), 0)

    def test_answer_endpoint_still_generates_when_context_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient("Answer without retrieval in prompt.")
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 30), "text/plain")},
                )
                answer_response = client.post(
                    "/api/query/answer",
                    json={
                        "query": "What does the note mention?",
                        "top_k": 3,
                        "mode": "dense",
                        "include_context_in_prompt": False,
                    },
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(answer_response.status_code, 200)
            payload = answer_response.json()
            self.assertEqual(payload["answer"], "Answer without retrieval in prompt.")
            self.assertGreater(len(payload["sources"]), 0)
            self.assertEqual(len(generation_client.calls), 1)
            self.assertEqual(generation_client.calls[0][0], "What does the note mention?")
    
    def test_answer_endpoint_supports_lexical_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient("Lexical answer [S1]")
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 30), "text/plain")},
                )
                answer_response = client.post(
                    "/api/query/answer",
                    json={"query": "alpha beta", "top_k": 3, "mode": "lexical"},
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(answer_response.status_code, 200)
            payload = answer_response.json()
            self.assertEqual(payload["mode"], "lexical")
            self.assertEqual(payload["answer"], "Lexical answer [S1]")
            self.assertGreater(len(payload["sources"]), 0)
            self.assertEqual(payload["sources"][0]["filename"], "notes.txt")
            self.assertGreaterEqual(payload["sources"][0]["score"], 0.0)
            self.assertEqual(len(generation_client.calls), 1)
            self.assertIn("alpha beta", generation_client.calls[0][0])

    def test_prompt_debug_supports_lexical_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient()
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 30), "text/plain")},
                )
                debug_response = client.post(
                    "/api/query/prompt-debug",
                    json={
                        "query": "alpha beta",
                        "top_k": 3,
                        "mode": "lexical",
                        "include_context_in_prompt": True,
                    },
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(debug_response.status_code, 200)
            payload = debug_response.json()
            self.assertEqual(payload["mode"], "lexical")
            self.assertGreater(len(payload["sources"]), 0)
            self.assertIn("[S1]", payload["prompt"])
            self.assertEqual(len(generation_client.calls), 0)

    def test_prompt_debug_lexical_mode_keeps_context_for_natural_language_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient()
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 30), "text/plain")},
                )
                debug_response = client.post(
                    "/api/query/prompt-debug",
                    json={
                        "query": "What does alpha mean?",
                        "top_k": 3,
                        "mode": "lexical",
                        "include_context_in_prompt": True,
                    },
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(debug_response.status_code, 200)
            payload = debug_response.json()
            self.assertGreater(len(payload["sources"]), 0)
            self.assertIn("Retrieved context:", payload["prompt"])
            self.assertIn("[S1]", payload["prompt"])

    def test_answer_endpoint_supports_hybrid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient("Hybrid answer [S1]")
            store = _InMemoryDenseStore()
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: store,  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                self._seed_document_chunks(
                    app.state.sqlite_db_path,
                    document_id="doc-1",
                    filename="notes.txt",
                    chunks=[
                        ChunkUpsert(
                            id="doc-1:000000",
                            chunk_index=0,
                            content="alpha alpha beta",
                        ),
                        ChunkUpsert(
                            id="doc-1:000001",
                            chunk_index=1,
                            content="alpha beta gamma",
                        ),
                    ],
                )
                store.dense_hits_override = [
                    DenseSearchHit(
                        chunk_id="doc-1:000001",
                        document_id="doc-1",
                        chunk_index=1,
                        score=0.99,
                    ),
                    DenseSearchHit(
                        chunk_id="doc-1:000000",
                        document_id="doc-1",
                        chunk_index=0,
                        score=0.98,
                    ),
                ]

                response = client.post(
                    "/api/query/answer",
                    json={"query": "alpha beta", "top_k": 2, "mode": "hybrid"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "hybrid")
            self.assertEqual(payload["answer"], "Hybrid answer [S1]")
            self.assertEqual([source["chunk_id"] for source in payload["sources"]], ["doc-1:000000", "doc-1:000001"])
            self.assertEqual(payload["sources"][0]["source_id"], "S1")
            self.assertEqual(len(generation_client.calls), 1)

    # prompt_debug in hybrid mode should include retrieved context in the prompt when enabled, and the sources should reflect the fused results.
    def test_prompt_debug_supports_hybrid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient()
            store = _InMemoryDenseStore()
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: store,  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                self._seed_document_chunks(
                    app.state.sqlite_db_path,
                    document_id="doc-1",
                    filename="notes.txt",
                    chunks=[
                        ChunkUpsert(
                            id="doc-1:000000",
                            chunk_index=0,
                            content="alpha alpha beta",
                        ),
                        ChunkUpsert(
                            id="doc-1:000001",
                            chunk_index=1,
                            content="alpha beta gamma",
                        ),
                    ],
                )
                store.dense_hits_override = [
                    DenseSearchHit(
                        chunk_id="doc-1:000001",
                        document_id="doc-1",
                        chunk_index=1,
                        score=0.99,
                    ),
                    DenseSearchHit(
                        chunk_id="doc-1:000000",
                        document_id="doc-1",
                        chunk_index=0,
                        score=0.98,
                    ),
                ]

                response = client.post(
                    "/api/query/prompt-debug",
                    json={
                        "query": "alpha beta",
                        "top_k": 2,
                        "mode": "hybrid",
                        "include_context_in_prompt": True,
                    },
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "hybrid")
            self.assertEqual([source["chunk_id"] for source in payload["sources"]], ["doc-1:000000", "doc-1:000001"])
            self.assertIn("[S1]", payload["prompt"])
            self.assertEqual(len(generation_client.calls), 0)

    def test_answer_endpoint_still_generates_when_no_hits_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = _RecordingGenerationClient("Answer with no retrieved context.")
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
                generation_client_factory=lambda _: generation_client,
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/query/answer",
                    json={"query": "What does the note mention?", "mode": "dense"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["sources"], [])
            self.assertEqual(payload["answer"], "Answer with no retrieved context.")
            self.assertEqual(len(generation_client.calls), 1)
            self.assertIn("No retrieved context was available", generation_client.calls[0][0])

    def test_root_serves_localhost_ui(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                settings=_build_settings(
                    sqlite_path=str(Path(temp_dir) / "app.db"),
                    storage_dir=str(Path(temp_dir) / "uploads"),
                ),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
                generation_client_factory=lambda _: _RecordingGenerationClient(),
            )

            with TestClient(app) as client:
                response = client.get("/")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Localhost RAG app UI", response.text)
            self.assertIn("/api/query/answer", response.text)
            self.assertIn("Retrieval mode", response.text)
            self.assertIn("value=\"lexical\"", response.text)
            self.assertIn("value=\"hybrid\"", response.text)

    def _seed_document_chunks(
        self,
        db_path: str,
        *,
        document_id: str,
        filename: str,
        chunks: list[ChunkUpsert],
    ) -> None:
        insert_document(
            db_path,
            document_id=document_id,
            filename=filename,
            source_type="txt",
            source_path=f"/tmp/{filename}",
            size_bytes=123,
            checksum=f"checksum-{document_id}",
            status="success",
        )
        replace_document_chunks(
            db_path,
            document_id=document_id,
            chunks=chunks,
        )
