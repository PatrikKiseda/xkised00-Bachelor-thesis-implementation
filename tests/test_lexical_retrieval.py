"""
Author: Patrik Kiseda
File: tests/test_lexical_retrieval.py
Description: Unit tests for SQLite FTS5 lexical retrieval helpers.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.storage.document_repository import insert_document
from app.storage.indexing_repository import (
    ChunkUpsert,
    normalize_fts5_query,
    replace_document_chunks,
    search_chunks_lexical,
)
from app.storage.sqlite_schema import initialize_sqlite_schema

# TestLexicalRetrieval: tests for the lexical retrieval helper functions, including query normalization and actual retrieval with SQLite FTS5. 
# Tests cover correct normalization of queries, handling of edge cases, and ensuring that retrieval returns correctly ranked and hydrated results
# based on the indexed content.
class TestLexicalRetrieval(unittest.TestCase):
    def test_normalize_fts5_query_quotes_terms_and_joins_with_and(self) -> None:
        normalized = normalize_fts5_query("alpha, beta! gamma?")

        self.assertEqual(normalized, '"alpha" AND "beta" AND "gamma"')

    def test_normalize_fts5_query_returns_none_when_no_terms_exist(self) -> None:
        normalized = normalize_fts5_query("... !!! ---")

        self.assertIsNone(normalized)

    def test_search_chunks_lexical_returns_ranked_hydrated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))
            self._insert_document(db_path, document_id="doc-1", filename="notes.txt")

            replace_document_chunks(
                db_path,
                document_id="doc-1",
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
                    ChunkUpsert(
                        id="doc-1:000002",
                        chunk_index=2,
                        content="gamma delta epsilon",
                    ),
                ],
            )

            rows = search_chunks_lexical(db_path, query_text="alpha beta", limit=5)

            self.assertEqual([row.chunk_id for row in rows], ["doc-1:000000", "doc-1:000001"])
            self.assertEqual(rows[0].document_id, "doc-1")
            self.assertEqual(rows[0].filename, "notes.txt")
            self.assertEqual(rows[0].chunk_index, 0)
            self.assertEqual(rows[0].content, "alpha alpha beta")
            self.assertLess(rows[0].raw_score, rows[1].raw_score)

    # test_search_chunks_lexical_uses_chunk_id_tiebreaker_for_identical_scores: if multiple chunks have same score, they should be consistently ranked by chunk_id to ensure stable results.
    def test_search_chunks_lexical_uses_chunk_id_tiebreaker_for_identical_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))
            self._insert_document(db_path, document_id="doc-1", filename="a.txt")
            self._insert_document(db_path, document_id="doc-2", filename="b.txt")

            replace_document_chunks(
                db_path,
                document_id="doc-1",
                chunks=[
                    ChunkUpsert(
                        id="doc-1:000000",
                        chunk_index=0,
                        content="shared lexical token",
                    )
                ],
            )
            replace_document_chunks(
                db_path,
                document_id="doc-2",
                chunks=[
                    ChunkUpsert(
                        id="doc-2:000000",
                        chunk_index=0,
                        content="shared lexical token",
                    )
                ],
            )

            rows = search_chunks_lexical(db_path, query_text="shared lexical", limit=5)

            self.assertEqual([row.chunk_id for row in rows], ["doc-1:000000", "doc-2:000000"])

    # test_search_chunks_lexical_returns_empty_list_for_tokenless_query: if query has no valid tokens, retrieval should return empty list, 
    # not error or irrelevant results.
    def test_search_chunks_lexical_returns_empty_list_for_tokenless_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))

            rows = search_chunks_lexical(db_path, query_text="!!!", limit=5)

            self.assertEqual(rows, [])

    # Helper to insert a document record into the database for testing. This is needed to set up the necessary metadata for chunks to be associated with a document.
    def _insert_document(self, db_path: str, *, document_id: str, filename: str) -> None:
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
