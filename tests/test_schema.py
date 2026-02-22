"""Tests for schema dataclasses."""
from __future__ import annotations

import pytest

from rag_tutorials.schema import Chunk, Document, QueryExample, RetrievalResult


class TestDocument:
    def test_instantiation(self):
        doc = Document(doc_id="D-1", title="T", section="S", text="some text")
        assert doc.doc_id == "D-1"
        assert doc.title == "T"
        assert doc.section == "S"
        assert doc.text == "some text"

    def test_slots_prevent_arbitrary_attributes(self):
        doc = Document(doc_id="D-1", title="T", section="S", text="text")
        with pytest.raises(AttributeError):
            doc.unexpected_field = "oops"  # type: ignore[attr-defined]


class TestChunk:
    def test_instantiation(self):
        chunk = Chunk(chunk_id="C-1", doc_id="D-1", section="S", text="chunk text")
        assert chunk.chunk_id == "C-1"
        assert chunk.doc_id == "D-1"
        assert chunk.section == "S"
        assert chunk.text == "chunk text"

    def test_slots_prevent_arbitrary_attributes(self):
        chunk = Chunk(chunk_id="C-1", doc_id="D-1", section="S", text="text")
        with pytest.raises(AttributeError):
            chunk.unexpected_field = "oops"  # type: ignore[attr-defined]


class TestQueryExample:
    def test_instantiation(self):
        q = QueryExample(
            query_id="Q-0",
            question="What is the policy?",
            relevant_chunk_ids=["C-1", "C-2"],
            target_doc_id="D-1",
            target_section="Remote Work",
            rationale="Policy context needed.",
        )
        assert q.query_id == "Q-0"
        assert q.question == "What is the policy?"
        assert q.relevant_chunk_ids == ["C-1", "C-2"]
        assert q.target_doc_id == "D-1"
        assert q.target_section == "Remote Work"
        assert q.rationale == "Policy context needed."

    def test_relevant_chunk_ids_empty_list(self):
        q = QueryExample(
            query_id="Q-0",
            question="?",
            relevant_chunk_ids=[],
            target_doc_id="D-1",
            target_section="S",
            rationale="R",
        )
        assert q.relevant_chunk_ids == []


class TestRetrievalResult:
    def test_instantiation(self):
        r = RetrievalResult(chunk_id="C-1", score=0.9, source="dense", text="some text")
        assert r.chunk_id == "C-1"
        assert r.score == 0.9
        assert r.source == "dense"
        assert r.text == "some text"

    def test_score_is_float(self):
        r = RetrievalResult(chunk_id="C-1", score=1, source="dense", text="t")
        # score stored as-is; confirm it can be numeric
        assert r.score == 1
