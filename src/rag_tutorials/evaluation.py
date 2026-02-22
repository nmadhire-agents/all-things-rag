from __future__ import annotations

from dataclasses import dataclass
import re
import time

from .schema import QueryExample, RetrievalResult


@dataclass(slots=True)
class EvalRow:
    """Single-query evaluation output used for aggregate reporting."""

    query_id: str
    recall_at_k: float
    mrr: float
    latency_ms: float
    groundedness: float


def _normalize(text: str) -> set[str]:
    """Normalize text to comparable lowercase token set for overlap checks."""
    return set(re.findall(r"[a-zA-Z0-9-]+", text.lower()))


def recall_at_k(results: list[RetrievalResult], query: QueryExample, k: int = 5) -> float:
    """Compute binary Recall@k using query's target document label."""
    window = results[:k]
    for result in window:
        if query.target_doc_id in result.chunk_id:
            return 1.0
    return 0.0


def reciprocal_rank(results: list[RetrievalResult], query: QueryExample) -> float:
    """Compute reciprocal rank for the first correctly targeted retrieval hit."""
    for rank, result in enumerate(results, start=1):
        if query.target_doc_id in result.chunk_id:
            return 1.0 / rank
    return 0.0


def groundedness_score(answer: str, contexts: list[str]) -> float:
    """Estimate groundedness as lexical overlap between answer and contexts."""
    answer_tokens = _normalize(answer)
    if not answer_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for context in contexts:
        context_tokens.update(_normalize(context))

    overlap = len(answer_tokens.intersection(context_tokens))
    return overlap / max(len(answer_tokens), 1)


def evaluate_single(
    query: QueryExample,
    retrieval_fn,
    answer_fn,
    top_k: int = 5,
) -> EvalRow:
    """Run retrieval + generation for one query and compute core metrics.

    Args:
        query: Query example containing expected relevance targets.
        retrieval_fn: Callable that returns ranked retrieval results.
        answer_fn: Callable that generates answer text from question + contexts.
        top_k: Number of contexts considered for metrics and answer grounding.

    Returns:
        `EvalRow` with recall, MRR, latency, and groundedness values.
    """
    started = time.perf_counter()
    try:
        retrieved: list[RetrievalResult] = retrieval_fn(query.question, top_k=top_k)
    except TypeError:
        retrieved = retrieval_fn(query.question)
    contexts = [result.text for result in retrieved[:top_k]]
    answer = answer_fn(query.question, contexts)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return EvalRow(
        query_id=query.query_id,
        recall_at_k=recall_at_k(retrieved, query, k=top_k),
        mrr=reciprocal_rank(retrieved, query),
        latency_ms=elapsed_ms,
        groundedness=groundedness_score(answer, contexts),
    )


def summarize(rows: list[EvalRow]) -> dict[str, float]:
    """Aggregate per-query metrics into simple mean summary values."""
    if not rows:
        return {"recall_at_k": 0.0, "mrr": 0.0, "latency_ms": 0.0, "groundedness": 0.0}

    return {
        "recall_at_k": sum(row.recall_at_k for row in rows) / len(rows),
        "mrr": sum(row.mrr for row in rows) / len(rows),
        "latency_ms": sum(row.latency_ms for row in rows) / len(rows),
        "groundedness": sum(row.groundedness for row in rows) / len(rows),
    }
