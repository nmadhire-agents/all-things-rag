"""Tests for embeddings.py — cosine_similarity (pure) and embed_texts (mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_tutorials.embeddings import cosine_similarity, embed_texts


# ---------------------------------------------------------------------------
# cosine_similarity — pure NumPy, no mocking needed
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_score_1(self):
        v = np.array([1.0, 0.0, 0.0])
        scores = cosine_similarity(v, np.array([v]))
        assert scores[0] == pytest.approx(1.0)

    def test_opposite_vectors_score_minus_1(self):
        v = np.array([1.0, 0.0, 0.0])
        scores = cosine_similarity(v, np.array([[-1.0, 0.0, 0.0]]))
        assert scores[0] == pytest.approx(-1.0)

    def test_orthogonal_vectors_score_0(self):
        q = np.array([1.0, 0.0])
        m = np.array([[0.0, 1.0]])
        scores = cosine_similarity(q, m)
        assert scores[0] == pytest.approx(0.0)

    def test_multiple_candidates_returns_aligned_scores(self):
        q = np.array([1.0, 0.0, 0.0])
        matrix = np.array([
            [1.0, 0.0, 0.0],   # identical → 1.0
            [0.0, 1.0, 0.0],   # orthogonal → 0.0
            [-1.0, 0.0, 0.0],  # opposite → -1.0
        ])
        scores = cosine_similarity(q, matrix)
        assert scores.shape == (3,)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(-1.0)

    def test_zero_query_vector_does_not_raise(self):
        q = np.array([0.0, 0.0, 0.0])
        m = np.array([[1.0, 0.0, 0.0]])
        # denominator clamped to 1e-12 — should not raise
        scores = cosine_similarity(q, m)
        assert scores.shape == (1,)

    def test_returns_float32_compatible_array(self):
        q = np.array([0.5, 0.5], dtype=np.float32)
        m = np.array([[0.5, 0.5]], dtype=np.float32)
        scores = cosine_similarity(q, m)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_ranking_order_is_correct(self):
        # query is close to v1, far from v2
        q = np.array([1.0, 0.0])
        m = np.array([[0.9, 0.1], [0.1, 0.9]])
        scores = cosine_similarity(q, m)
        assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# embed_texts — OpenAI API mocked
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    def _make_mock_response(self, texts: list[str], dim: int = 4) -> MagicMock:
        """Build a fake OpenAI embeddings response."""
        response = MagicMock()
        response.data = [
            MagicMock(embedding=[float(i) * 0.1] * dim)
            for i in range(len(texts))
        ]
        return response

    @patch("rag_tutorials.embeddings.OpenAI")
    def test_returns_numpy_array(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = self._make_mock_response(["hello"], dim=4)

        result = embed_texts(["hello"])
        assert isinstance(result, np.ndarray)

    @patch("rag_tutorials.embeddings.OpenAI")
    def test_shape_matches_input(self, mock_openai_cls):
        texts = ["first", "second", "third"]
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = self._make_mock_response(texts, dim=8)

        result = embed_texts(texts)
        assert result.shape == (3, 8)

    @patch("rag_tutorials.embeddings.OpenAI")
    def test_dtype_is_float32(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = self._make_mock_response(["hi"], dim=4)

        result = embed_texts(["hi"])
        assert result.dtype == np.float32

    @patch("rag_tutorials.embeddings.OpenAI")
    def test_passes_model_to_api(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = self._make_mock_response(["x"], dim=2)

        embed_texts(["x"], model="text-embedding-3-large")
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large", input=["x"]
        )
