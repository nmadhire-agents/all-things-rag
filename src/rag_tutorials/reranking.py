from __future__ import annotations

from sentence_transformers import CrossEncoder

from .schema import RetrievalResult


class LocalCrossEncoderReranker:
    """Second-stage reranker using a local cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder used for pairwise query-chunk scoring.

        Args:
            model_name: Sentence-transformers cross-encoder model identifier.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        """Reorder first-pass retrieval results with cross-encoder relevance scores.

        Args:
            query: User query string.
            results: First-pass retrieval candidates to re-score.
            top_k: Number of reranked results to return.

        Returns:
            Top `top_k` results sorted by cross-encoder score.
        """
        if not results:
            return []

        pairs = [[query, result.text] for result in results]
        scores = self.model.predict(pairs)

        ranked = sorted(
            [
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    score=float(score),
                    source="reranked",
                    text=result.text,
                )
                for result, score in zip(results, scores, strict=True)
            ],
            key=lambda row: row.score,
            reverse=True,
        )
        return ranked[:top_k]
