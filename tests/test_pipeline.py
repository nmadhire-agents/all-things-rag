"""Tests for pipeline.py — build_dense_retriever, build_hybrid_retriever, top_scores_preview.

embed_texts and build_chroma_collection are mocked so no OpenAI key is needed.
build_dense_retriever's Chroma collection is injected via a patched
build_chroma_collection that uses a real in-process tmp_path store.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_tutorials.pipeline import (
    build_dense_retriever,
    build_hybrid_retriever,
    prepare_chunks,
    top_scores_preview,
)
from rag_tutorials.schema import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 3  # embedding dimension used in all pipeline tests


def _fake_embed(texts: list[str], model: str = "") -> np.ndarray:
    """Deterministic fake embedder: returns one-hot-like vectors."""
    rng = np.random.default_rng(len(texts))
    return rng.random((len(texts), DIM)).astype(np.float32)


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="D-1", section="S", text=text)


# ---------------------------------------------------------------------------
# prepare_chunks
# ---------------------------------------------------------------------------

class TestPrepareChunks:
    def test_fixed_mode_returns_chunks(self, tmp_path):
        from rag_tutorials.data_generation import HANDBOOK_TEXT

        hb = tmp_path / "handbook_manual.txt"
        hb.write_text(HANDBOOK_TEXT)
        chunks = prepare_chunks("fixed", handbook_path=str(hb))
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_semantic_mode_returns_chunks(self, tmp_path):
        from rag_tutorials.data_generation import HANDBOOK_TEXT

        hb = tmp_path / "handbook_manual.txt"
        hb.write_text(HANDBOOK_TEXT)
        chunks = prepare_chunks("semantic", handbook_path=str(hb))
        assert len(chunks) > 0

    def test_fixed_chunk_ids_contain_fix(self, tmp_path):
        from rag_tutorials.data_generation import HANDBOOK_TEXT

        hb = tmp_path / "handbook_manual.txt"
        hb.write_text(HANDBOOK_TEXT)
        chunks = prepare_chunks("fixed", handbook_path=str(hb))
        assert all("-FIX-" in c.chunk_id for c in chunks)

    def test_semantic_chunk_ids_contain_sem(self, tmp_path):
        from rag_tutorials.data_generation import HANDBOOK_TEXT

        hb = tmp_path / "handbook_manual.txt"
        hb.write_text(HANDBOOK_TEXT)
        chunks = prepare_chunks("semantic", handbook_path=str(hb))
        assert all("-SEM-" in c.chunk_id for c in chunks)


# ---------------------------------------------------------------------------
# build_dense_retriever
# ---------------------------------------------------------------------------

class TestBuildDenseRetriever:
    @pytest.fixture()
    def dense_retriever_and_vectors(self, tmp_path):
        chunks = [
            _make_chunk("C-remote", "remote work allowed"),
            _make_chunk("C-security", "lost devices reported"),
            _make_chunk("C-travel", "travel approval required"),
        ]
        with patch("rag_tutorials.pipeline.embed_texts", side_effect=_fake_embed), \
             patch("rag_tutorials.pipeline.build_chroma_collection") as mock_build:
            # Use a real Chroma collection backed by tmp_path
            from rag_tutorials.vector_store import build_chroma_collection as real_build
            mock_build.side_effect = lambda chunks, embeddings, collection_name: real_build(
                chunks=chunks,
                embeddings=embeddings,
                collection_name=collection_name,
                persist_dir=str(tmp_path),
            )
            retrieve_fn, vectors = build_dense_retriever(
                chunks=chunks,
                collection_name="test_dense",
                embedding_model="text-embedding-3-small",
            )
        return retrieve_fn, vectors, chunks

    def test_returns_callable_and_matrix(self, dense_retriever_and_vectors):
        retrieve_fn, vectors, _ = dense_retriever_and_vectors
        assert callable(retrieve_fn)
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (3, DIM)

    def test_retrieve_returns_results(self, dense_retriever_and_vectors, tmp_path):
        retrieve_fn, _, _ = dense_retriever_and_vectors
        with patch("rag_tutorials.pipeline.embed_texts", side_effect=_fake_embed):
            results = retrieve_fn("remote work policy", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_results_have_dense_source(self, dense_retriever_and_vectors):
        retrieve_fn, _, _ = dense_retriever_and_vectors
        with patch("rag_tutorials.pipeline.embed_texts", side_effect=_fake_embed):
            results = retrieve_fn("query", top_k=3)
        assert all(r.source == "dense" for r in results)


# ---------------------------------------------------------------------------
# build_hybrid_retriever
# ---------------------------------------------------------------------------

class TestBuildHybridRetriever:
    @pytest.fixture()
    def hybrid_setup(self, tmp_path):
        chunks = [
            _make_chunk("C-remote", "remote work allowed from home"),
            _make_chunk("C-security", "lost devices reported within one hour"),
            _make_chunk("C-travel", "travel approval required 14 days"),
        ]

        def fake_dense(question: str, top_k: int = 5) -> list[RetrievalResult]:
            return [
                RetrievalResult(chunk_id=c.chunk_id, score=0.8, source="dense", text=c.text)
                for c in chunks[:top_k]
            ]

        return build_hybrid_retriever(chunks, fake_dense), chunks

    def test_returns_callable(self, hybrid_setup):
        retrieve_fn, _ = hybrid_setup
        assert callable(retrieve_fn)

    def test_retrieve_returns_results(self, hybrid_setup):
        retrieve_fn, _ = hybrid_setup
        results = retrieve_fn("remote work", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_have_hybrid_source(self, hybrid_setup):
        retrieve_fn, _ = hybrid_setup
        results = retrieve_fn("travel approval", top_k=3)
        assert all(r.source == "hybrid" for r in results)


# ---------------------------------------------------------------------------
# top_scores_preview
# ---------------------------------------------------------------------------

class TestTopScoresPreview:
    def test_returns_ranked_dicts(self):
        chunks = [_make_chunk(f"C-{i}", f"text {i}") for i in range(4)]
        vectors = np.eye(4, dtype=np.float32)  # identity matrix

        with patch("rag_tutorials.pipeline.embed_texts") as mock_embed:
            # Query vector identical to row 2 → C-2 should rank first
            mock_embed.return_value = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
            previews = top_scores_preview("query", chunks, vectors, "model", top_k=3)

        assert len(previews) == 3
        assert previews[0]["chunk_id"] == "C-2"
        assert "rank" in previews[0]
        assert "score" in previews[0]
        assert "text" in previews[0]

    def test_top_k_limits_output(self):
        chunks = [_make_chunk(f"C-{i}", f"text {i}") for i in range(5)]
        vectors = np.eye(5, dtype=np.float32)
        with patch("rag_tutorials.pipeline.embed_texts") as mock_embed:
            mock_embed.return_value = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            previews = top_scores_preview("query", chunks, vectors, "model", top_k=2)
        assert len(previews) == 2

    def test_ranks_start_at_1(self):
        chunks = [_make_chunk("C-0", "text")]
        vectors = np.array([[1.0]], dtype=np.float32)
        with patch("rag_tutorials.pipeline.embed_texts") as mock_embed:
            mock_embed.return_value = np.array([[1.0]], dtype=np.float32)
            previews = top_scores_preview("query", chunks, vectors, "model", top_k=1)
        assert previews[0]["rank"] == 1
