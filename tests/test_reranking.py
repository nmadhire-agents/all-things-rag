"""Tests for reranking.py — LocalCrossEncoderReranker (CrossEncoder mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_tutorials.reranking import LocalCrossEncoderReranker
from rag_tutorials.schema import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, text: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, source="dense", text=text)


# ---------------------------------------------------------------------------
# LocalCrossEncoderReranker
# ---------------------------------------------------------------------------

class TestLocalCrossEncoderReranker:
    @pytest.fixture()
    def reranker(self):
        """Return a reranker whose underlying CrossEncoder is fully mocked."""
        with patch("rag_tutorials.reranking.CrossEncoder") as mock_cls:
            mock_model = MagicMock()
            mock_cls.return_value = mock_model
            r = LocalCrossEncoderReranker(model_name="cross-encoder/test-model")
            r._mock_model = mock_model  # expose for per-test score control
            yield r

    # --- rerank ---

    def test_rerank_empty_results_returns_empty(self, reranker):
        assert reranker.rerank("query", []) == []

    def test_rerank_returns_retrieval_results(self, reranker):
        results = [_make_result("C-1", "text one"), _make_result("C-2", "text two")]
        reranker._mock_model.predict.return_value = np.array([0.9, 0.3])
        output = reranker.rerank("query", results)
        assert all(isinstance(r, RetrievalResult) for r in output)

    def test_rerank_source_field_is_reranked(self, reranker):
        results = [_make_result("C-1", "text")]
        reranker._mock_model.predict.return_value = np.array([0.7])
        output = reranker.rerank("query", results)
        assert output[0].source == "reranked"

    def test_rerank_sorts_by_score_descending(self, reranker):
        results = [
            _make_result("C-low", "low relevance text"),
            _make_result("C-high", "high relevance text"),
            _make_result("C-mid", "medium relevance text"),
        ]
        reranker._mock_model.predict.return_value = np.array([0.2, 0.9, 0.5])
        output = reranker.rerank("query", results, top_k=3)
        assert output[0].chunk_id == "C-high"
        assert output[1].chunk_id == "C-mid"
        assert output[2].chunk_id == "C-low"

    def test_rerank_respects_top_k(self, reranker):
        results = [_make_result(f"C-{i}", f"text {i}") for i in range(5)]
        reranker._mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        output = reranker.rerank("query", results, top_k=2)
        assert len(output) == 2

    def test_rerank_preserves_chunk_id_and_text(self, reranker):
        results = [_make_result("C-42", "the important text")]
        reranker._mock_model.predict.return_value = np.array([0.88])
        output = reranker.rerank("query", results)
        assert output[0].chunk_id == "C-42"
        assert output[0].text == "the important text"

    def test_rerank_score_comes_from_cross_encoder(self, reranker):
        results = [_make_result("C-1", "text")]
        reranker._mock_model.predict.return_value = np.array([0.77])
        output = reranker.rerank("query", results)
        assert output[0].score == pytest.approx(0.77)

    def test_rerank_passes_correct_pairs_to_model(self, reranker):
        results = [
            _make_result("C-1", "alpha text"),
            _make_result("C-2", "beta text"),
        ]
        reranker._mock_model.predict.return_value = np.array([0.5, 0.6])
        reranker.rerank("my query", results)
        pairs_sent = reranker._mock_model.predict.call_args[0][0]
        assert pairs_sent == [["my query", "alpha text"], ["my query", "beta text"]]

    def test_constructor_uses_provided_model_name(self):
        with patch("rag_tutorials.reranking.CrossEncoder") as mock_cls:
            mock_cls.return_value = MagicMock()
            LocalCrossEncoderReranker(model_name="cross-encoder/custom-model")
            mock_cls.assert_called_once_with("cross-encoder/custom-model")
