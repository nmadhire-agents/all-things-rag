from __future__ import annotations

from dataclasses import dataclass
import re
import time

from .schema import QueryExample, RetrievalResult


@dataclass(slots=True)
class EvalRow:
    query_id: str
    recall_at_k: float
    mrr: float
    latency_ms: float
    groundedness: float


def _normalize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9-]+", text.lower()))


def recall_at_k(results: list[RetrievalResult], query: QueryExample, k: int = 5) -> float:
    window = results[:k]
    for result in window:
        if query.target_doc_id in result.chunk_id:
            return 1.0
    return 0.0


def reciprocal_rank(results: list[RetrievalResult], query: QueryExample) -> float:
    for rank, result in enumerate(results, start=1):
        if query.target_doc_id in result.chunk_id:
            return 1.0 / rank
    return 0.0


def groundedness_score(answer: str, contexts: list[str]) -> float:
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
    if not rows:
        return {"recall_at_k": 0.0, "mrr": 0.0, "latency_ms": 0.0, "groundedness": 0.0}

    return {
        "recall_at_k": sum(row.recall_at_k for row in rows) / len(rows),
        "mrr": sum(row.mrr for row in rows) / len(rows),
        "latency_ms": sum(row.latency_ms for row in rows) / len(rows),
        "groundedness": sum(row.groundedness for row in rows) / len(rows),
    }
