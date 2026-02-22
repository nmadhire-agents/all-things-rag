"""Tests for retrieval.py — BM25 index/search and Reciprocal Rank Fusion."""
from __future__ import annotations

import pytest

from rag_tutorials.retrieval import bm25_search, build_bm25, reciprocal_rank_fusion
from rag_tutorials.schema import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str, doc_id: str = "D-1") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, section="S", text=text)


def _make_result(chunk_id: str, score: float = 0.5, text: str = "t") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, source="dense", text=text)


# ---------------------------------------------------------------------------
# build_bm25
# ---------------------------------------------------------------------------

class TestBuildBm25:
    def test_returns_three_tuple(self, sample_chunks):
        index, corpus, chunk_ids = build_bm25(sample_chunks)
        assert corpus == [c.text for c in sample_chunks]
        assert chunk_ids == [c.chunk_id for c in sample_chunks]

    def test_corpus_length_matches_chunks(self, sample_chunks):
        _, corpus, chunk_ids = build_bm25(sample_chunks)
        assert len(corpus) == len(sample_chunks)
        assert len(chunk_ids) == len(sample_chunks)

    def test_accepts_single_chunk(self):
        chunks = [_make_chunk("C-1", "remote work policy")]
        index, corpus, ids = build_bm25(chunks)
        assert len(corpus) == 1
        assert ids == ["C-1"]


# ---------------------------------------------------------------------------
# bm25_search
# ---------------------------------------------------------------------------

class TestBm25Search:
    @pytest.fixture()
    def index_and_corpus(self):
        chunks = [
            _make_chunk("C-remote", "employees may work remotely from home vpn required"),
            _make_chunk("C-international", "working from another country capped at 14 days"),
            _make_chunk("C-security", "lost devices must be reported within one hour"),
        ]
        index, corpus, chunk_ids = build_bm25(chunks)
        return index, corpus, chunk_ids

    def test_returns_retrieval_results(self, index_and_corpus):
        index, corpus, chunk_ids = index_and_corpus
        results = bm25_search(index, "remote work", corpus, chunk_ids, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_top_k_limits_results(self, index_and_corpus):
        index, corpus, chunk_ids = index_and_corpus
        results = bm25_search(index, "work policy", corpus, chunk_ids, top_k=1)
        assert len(results) == 1

    def test_source_field_is_keyword(self, index_and_corpus):
        index, corpus, chunk_ids = index_and_corpus
        results = bm25_search(index, "remote", corpus, chunk_ids)
        assert all(r.source == "keyword" for r in results)

    def test_lexical_match_ranks_first(self, index_and_corpus):
        index, corpus, chunk_ids = index_and_corpus
        results = bm25_search(index, "lost devices reported", corpus, chunk_ids, top_k=3)
        assert results[0].chunk_id == "C-security"

    def test_scores_are_non_negative(self, index_and_corpus):
        index, corpus, chunk_ids = index_and_corpus
        results = bm25_search(index, "work", corpus, chunk_ids)
        assert all(r.score >= 0.0 for r in results)

    def test_results_sorted_descending(self, index_and_corpus):
        index, corpus, chunk_ids = index_and_corpus
        results = bm25_search(index, "work remote", corpus, chunk_ids, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_returns_retrieval_results(self):
        dense = [_make_result("C-1"), _make_result("C-2")]
        keyword = [_make_result("C-2"), _make_result("C-3")]
        results = reciprocal_rank_fusion(dense, keyword)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_source_field_is_hybrid(self):
        dense = [_make_result("C-1")]
        keyword = [_make_result("C-1")]
        results = reciprocal_rank_fusion(dense, keyword)
        assert all(r.source == "hybrid" for r in results)

    def test_chunk_appearing_in_both_lists_scores_highest(self):
        # C-shared ranks first in both lists → should have highest fused score
        dense = [_make_result("C-shared"), _make_result("C-dense-only")]
        keyword = [_make_result("C-shared"), _make_result("C-keyword-only")]
        results = reciprocal_rank_fusion(dense, keyword)
        assert results[0].chunk_id == "C-shared"

    def test_deduplicates_chunk_ids(self):
        dense = [_make_result("C-1"), _make_result("C-2")]
        keyword = [_make_result("C-1"), _make_result("C-2")]
        results = reciprocal_rank_fusion(dense, keyword)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids))

    def test_scores_sorted_descending(self):
        dense = [_make_result("C-1"), _make_result("C-2"), _make_result("C-3")]
        keyword = [_make_result("C-3"), _make_result("C-2"), _make_result("C-1")]
        results = reciprocal_rank_fusion(dense, keyword)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_formula_correctness(self):
        # C-1 is rank 1 in both lists with k=60
        # expected score = 1/(60+1) + 1/(60+1) = 2/61
        dense = [_make_result("C-1")]
        keyword = [_make_result("C-1")]
        results = reciprocal_rank_fusion(dense, keyword, k=60)
        expected = 2 / 61
        assert results[0].score == pytest.approx(expected)

    def test_union_of_all_chunks_present(self):
        dense = [_make_result("C-a"), _make_result("C-b")]
        keyword = [_make_result("C-b"), _make_result("C-c")]
        results = reciprocal_rank_fusion(dense, keyword)
        ids = {r.chunk_id for r in results}
        assert ids == {"C-a", "C-b", "C-c"}

    def test_empty_dense_list(self):
        keyword = [_make_result("C-1"), _make_result("C-2")]
        results = reciprocal_rank_fusion([], keyword)
        assert len(results) == 2

    def test_empty_keyword_list(self):
        dense = [_make_result("C-1")]
        results = reciprocal_rank_fusion(dense, [])
        assert len(results) == 1
