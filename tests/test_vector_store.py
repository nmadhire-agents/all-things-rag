"""Tests for vector_store.py — build_chroma_collection and dense_search.

Uses real Chroma PersistentClient via pytest's tmp_path fixture so no mocking
of Chroma internals is needed.  Embeddings are tiny synthetic vectors (3-dim)
to keep tests fast and deterministic.
"""
from __future__ import annotations

import pytest

from rag_tutorials.schema import Chunk, RetrievalResult
from rag_tutorials.vector_store import build_chroma_collection, dense_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str, doc_id: str = "D-1", section: str = "S") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, section=section, text=text)


# Three-dimensional test embeddings — avoids OpenAI dependency entirely.
EMBS = {
    "C-remote":        [1.0, 0.0, 0.0],
    "C-international": [0.0, 1.0, 0.0],
    "C-security":      [0.0, 0.0, 1.0],
}


@pytest.fixture()
def collection(tmp_path):
    chunks = [
        _make_chunk("C-remote",        "remote work policy"),
        _make_chunk("C-international", "international work policy"),
        _make_chunk("C-security",      "security policy lost devices"),
    ]
    embeddings = [EMBS[c.chunk_id] for c in chunks]
    return build_chroma_collection(
        chunks=chunks,
        embeddings=embeddings,
        collection_name="test_collection",
        persist_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# build_chroma_collection
# ---------------------------------------------------------------------------

class TestBuildChromaCollection:
    def test_returns_collection(self, tmp_path):
        chunks = [_make_chunk("C-1", "text")]
        col = build_chroma_collection(
            chunks=chunks,
            embeddings=[[1.0, 0.0, 0.0]],
            collection_name="test_build",
            persist_dir=str(tmp_path),
        )
        assert col is not None

    def test_collection_has_correct_count(self, collection):
        assert collection.count() == 3

    def test_rebuilds_when_collection_already_exists(self, tmp_path):
        chunks = [_make_chunk("C-1", "first")]
        build_chroma_collection(
            chunks=chunks,
            embeddings=[[1.0, 0.0]],
            collection_name="dup_test",
            persist_dir=str(tmp_path),
        )
        # Build again with different chunks — should replace cleanly
        chunks2 = [_make_chunk("C-2", "second"), _make_chunk("C-3", "third")]
        col2 = build_chroma_collection(
            chunks=chunks2,
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            collection_name="dup_test",
            persist_dir=str(tmp_path),
        )
        assert col2.count() == 2


# ---------------------------------------------------------------------------
# dense_search
# ---------------------------------------------------------------------------

class TestDenseSearch:
    def test_returns_retrieval_results(self, collection):
        results = dense_search(collection, EMBS["C-remote"], top_k=3)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_top_k_limits_results(self, collection):
        results = dense_search(collection, EMBS["C-remote"], top_k=1)
        assert len(results) == 1

    def test_source_field_is_dense(self, collection):
        results = dense_search(collection, EMBS["C-remote"], top_k=3)
        assert all(r.source == "dense" for r in results)

    def test_nearest_chunk_ranks_first(self, collection):
        # Query identical to C-remote embedding → C-remote must be rank 1
        results = dense_search(collection, EMBS["C-remote"], top_k=3)
        assert results[0].chunk_id == "C-remote"

    def test_best_match_scores_highest(self, collection):
        # The nearest-neighbour chunk must have the highest score (= 1.0 for exact match)
        results = dense_search(collection, EMBS["C-remote"], top_k=3)
        assert results[0].chunk_id == "C-remote"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_scores_ordered_descending(self, collection):
        results = dense_search(collection, EMBS["C-remote"], top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_text_field_matches_original(self, collection):
        results = dense_search(collection, EMBS["C-security"], top_k=1)
        assert results[0].text == "security policy lost devices"

    def test_chunk_id_field_present(self, collection):
        results = dense_search(collection, EMBS["C-international"], top_k=1)
        assert results[0].chunk_id == "C-international"
