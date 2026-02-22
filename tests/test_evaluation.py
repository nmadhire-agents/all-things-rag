"""Tests for evaluation.py — recall, MRR, groundedness, evaluate_single, summarize."""
from __future__ import annotations

import pytest

from rag_tutorials.evaluation import (
    EvalRow,
    _normalize,
    evaluate_single,
    groundedness_score,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from rag_tutorials.schema import QueryExample, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, score: float = 0.8, text: str = "text") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, source="dense", text=text)


def _make_query(target_doc_id: str = "DOC-001") -> QueryExample:
    return QueryExample(
        query_id="Q-0",
        question="test?",
        relevant_chunk_ids=[],
        target_doc_id=target_doc_id,
        target_section="Remote Work",
        rationale="R",
    )


# ---------------------------------------------------------------------------
# _normalize (internal helper)
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert "hello" in _normalize("Hello")

    def test_extracts_alphanumeric_tokens(self):
        tokens = _normalize("Hello, World! 123.")
        assert tokens == {"hello", "world", "123"}

    def test_handles_hyphens(self):
        assert "form-a12" in _normalize("Form-A12")

    def test_empty_string(self):
        assert _normalize("") == set()


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_hit_at_rank_1(self):
        results = [_make_result("DOC-001-FIX-00")]
        query = _make_query(target_doc_id="DOC-001")
        assert recall_at_k(results, query, k=5) == 1.0

    def test_hit_at_last_rank(self):
        results = [
            _make_result("DOC-002-FIX-00"),
            _make_result("DOC-001-FIX-00"),
        ]
        query = _make_query(target_doc_id="DOC-001")
        assert recall_at_k(results, query, k=5) == 1.0

    def test_miss_when_target_absent(self):
        results = [_make_result("DOC-002-FIX-00")]
        query = _make_query(target_doc_id="DOC-001")
        assert recall_at_k(results, query, k=5) == 0.0

    def test_respects_k_cutoff(self):
        # target at rank 3 but k=2 → miss
        results = [
            _make_result("DOC-002-FIX-00"),
            _make_result("DOC-003-FIX-00"),
            _make_result("DOC-001-FIX-00"),
        ]
        query = _make_query(target_doc_id="DOC-001")
        assert recall_at_k(results, query, k=2) == 0.0

    def test_empty_results(self):
        assert recall_at_k([], _make_query(), k=5) == 0.0


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_rank_1_gives_score_1(self):
        results = [_make_result("DOC-001-FIX-00")]
        query = _make_query(target_doc_id="DOC-001")
        assert reciprocal_rank(results, query) == pytest.approx(1.0)

    def test_rank_2_gives_score_half(self):
        results = [_make_result("DOC-002-FIX-00"), _make_result("DOC-001-FIX-00")]
        query = _make_query(target_doc_id="DOC-001")
        assert reciprocal_rank(results, query) == pytest.approx(0.5)

    def test_rank_5_gives_score_0_2(self):
        results = [_make_result(f"DOC-00{i}-FIX-00") for i in range(2, 7)]
        results.append(_make_result("DOC-001-FIX-00"))
        query = _make_query(target_doc_id="DOC-001")
        assert reciprocal_rank(results, query) == pytest.approx(1 / 6)

    def test_no_hit_gives_0(self):
        results = [_make_result("DOC-999-FIX-00")]
        query = _make_query(target_doc_id="DOC-001")
        assert reciprocal_rank(results, query) == 0.0

    def test_empty_results(self):
        assert reciprocal_rank([], _make_query()) == 0.0


# ---------------------------------------------------------------------------
# groundedness_score
# ---------------------------------------------------------------------------

class TestGroundednessScore:
    def test_full_overlap_is_1(self):
        assert groundedness_score("remote work policy", ["remote work policy"]) == pytest.approx(1.0)

    def test_no_overlap_is_0(self):
        assert groundedness_score("alpha beta", ["gamma delta"]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # answer has 4 tokens, 2 overlap
        score = groundedness_score("remote work policy document", ["remote work approved"])
        assert 0.0 < score < 1.0

    def test_empty_answer_returns_0(self):
        assert groundedness_score("", ["some context"]) == pytest.approx(0.0)

    def test_empty_contexts(self):
        assert groundedness_score("hello world", []) == pytest.approx(0.0)

    def test_multiple_contexts_union(self):
        # answer tokens covered across two context chunks
        score = groundedness_score("remote work vpn", ["remote work allowed", "vpn required"])
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_single
# ---------------------------------------------------------------------------

class TestEvaluateSingle:
    def test_returns_eval_row(self, sample_query):
        def retrieval_fn(q, top_k=5):
            return [_make_result("DOC-001-FIX-00", text="remote work")]

        def answer_fn(q, contexts):
            return "remote work allowed"

        row = evaluate_single(sample_query, retrieval_fn, answer_fn, top_k=5)
        assert isinstance(row, EvalRow)
        assert row.query_id == sample_query.query_id
        assert 0.0 <= row.recall_at_k <= 1.0
        assert 0.0 <= row.mrr <= 1.0
        assert 0.0 <= row.groundedness <= 1.0
        assert row.latency_ms >= 0.0

    def test_correct_retrieval_gives_recall_1(self, sample_query):
        def retrieval_fn(q, top_k=5):
            return [_make_result("DOC-001-FIX-00")]

        row = evaluate_single(sample_query, retrieval_fn, lambda q, c: "ans", top_k=5)
        assert row.recall_at_k == 1.0
        assert row.mrr == pytest.approx(1.0)

    def test_wrong_retrieval_gives_recall_0(self, sample_query):
        def retrieval_fn(q, top_k=5):
            return [_make_result("DOC-999-FIX-00")]

        row = evaluate_single(sample_query, retrieval_fn, lambda q, c: "ans", top_k=5)
        assert row.recall_at_k == 0.0
        assert row.mrr == 0.0

    def test_accepts_retrieval_fn_without_top_k_kwarg(self, sample_query):
        """evaluate_single falls back to positional call if top_k kwarg raises TypeError."""
        def retrieval_fn_no_kwarg(q):
            return [_make_result("DOC-001-FIX-00")]

        row = evaluate_single(sample_query, retrieval_fn_no_kwarg, lambda q, c: "ans", top_k=5)
        assert row.recall_at_k == 1.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_empty_returns_zeros(self):
        result = summarize([])
        assert result == {"recall_at_k": 0.0, "mrr": 0.0, "latency_ms": 0.0, "groundedness": 0.0}

    def test_single_row(self):
        row = EvalRow(query_id="Q-0", recall_at_k=1.0, mrr=0.5, latency_ms=100.0, groundedness=0.8)
        result = summarize([row])
        assert result["recall_at_k"] == pytest.approx(1.0)
        assert result["mrr"] == pytest.approx(0.5)
        assert result["latency_ms"] == pytest.approx(100.0)
        assert result["groundedness"] == pytest.approx(0.8)

    def test_multiple_rows_average(self):
        rows = [
            EvalRow(query_id="Q-0", recall_at_k=1.0, mrr=1.0, latency_ms=100.0, groundedness=0.8),
            EvalRow(query_id="Q-1", recall_at_k=0.0, mrr=0.0, latency_ms=200.0, groundedness=0.4),
        ]
        result = summarize(rows)
        assert result["recall_at_k"] == pytest.approx(0.5)
        assert result["mrr"] == pytest.approx(0.5)
        assert result["latency_ms"] == pytest.approx(150.0)
        assert result["groundedness"] == pytest.approx(0.6)
