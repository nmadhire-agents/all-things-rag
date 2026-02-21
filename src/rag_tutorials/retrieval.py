from __future__ import annotations

from collections import defaultdict

from rank_bm25 import BM25Okapi

from .schema import Chunk, RetrievalResult


def build_bm25(chunks: list[Chunk]) -> tuple[BM25Okapi, list[str], list[str]]:
    corpus = [chunk.text for chunk in chunks]
    tokenized = [text.lower().split() for text in corpus]
    index = BM25Okapi(tokenized)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    return index, corpus, chunk_ids


def bm25_search(index: BM25Okapi, query: str, corpus: list[str], chunk_ids: list[str], top_k: int = 5):
    scores = index.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

    results: list[RetrievalResult] = []
    for idx in ranked:
        results.append(
            RetrievalResult(
                chunk_id=chunk_ids[idx],
                score=float(scores[idx]),
                source="keyword",
                text=corpus[idx],
            )
        )
    return results


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult], keyword_results: list[RetrievalResult], k: int = 60
) -> list[RetrievalResult]:
    fused_scores: dict[str, float] = defaultdict(float)
    text_lookup: dict[str, str] = {}

    for rank, result in enumerate(dense_results, start=1):
        fused_scores[result.chunk_id] += 1 / (k + rank)
        text_lookup[result.chunk_id] = result.text

    for rank, result in enumerate(keyword_results, start=1):
        fused_scores[result.chunk_id] += 1 / (k + rank)
        text_lookup[result.chunk_id] = result.text

    sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        RetrievalResult(chunk_id=chunk_id, score=score, source="hybrid", text=text_lookup[chunk_id])
        for chunk_id, score in sorted_results
    ]
