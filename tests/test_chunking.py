"""Tests for chunking.py — fixed and semantic chunking strategies."""
from __future__ import annotations

import pytest

from rag_tutorials.chunking import fixed_chunk_documents, semantic_chunk_documents
from rag_tutorials.schema import Chunk, Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str = "D-1", section: str = "Remote Work", text: str = "") -> Document:
    return Document(doc_id=doc_id, title="T", section=section, text=text)


# ---------------------------------------------------------------------------
# fixed_chunk_documents
# ---------------------------------------------------------------------------

class TestFixedChunkDocuments:
    def test_returns_list_of_chunks(self, sample_documents):
        chunks = fixed_chunk_documents(sample_documents)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_count_covers_all_text(self):
        text = "x" * 100
        doc = _make_doc(text=text)
        chunks = fixed_chunk_documents([doc], chunk_size=30)
        reassembled = "".join(c.text for c in chunks)
        assert reassembled == text

    def test_no_chunk_exceeds_chunk_size(self):
        doc = _make_doc(text="a" * 500)
        chunks = fixed_chunk_documents([doc], chunk_size=100)
        assert all(len(c.text) <= 100 for c in chunks)

    def test_chunk_ids_are_unique(self, sample_documents):
        chunks = fixed_chunk_documents(sample_documents)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_include_fix_suffix(self):
        doc = _make_doc(doc_id="DOC-XYZ", text="hello world")
        chunks = fixed_chunk_documents([doc], chunk_size=5)
        assert all("-FIX-" in c.chunk_id for c in chunks)

    def test_metadata_propagates(self):
        doc = _make_doc(doc_id="D-99", section="Security", text="VPN required.")
        chunks = fixed_chunk_documents([doc], chunk_size=50)
        for chunk in chunks:
            assert chunk.doc_id == "D-99"
            assert chunk.section == "Security"

    def test_empty_documents_list(self):
        assert fixed_chunk_documents([]) == []

    def test_single_short_document_produces_one_chunk(self):
        doc = _make_doc(text="short")
        chunks = fixed_chunk_documents([doc], chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0].text == "short"

    def test_multiple_documents_produce_separate_chunk_ids(self):
        docs = [_make_doc(doc_id=f"D-{i}", text="text") for i in range(3)]
        chunks = fixed_chunk_documents(docs, chunk_size=50)
        ids = {c.chunk_id for c in chunks}
        assert len(ids) == 3  # each doc produces one chunk


# ---------------------------------------------------------------------------
# semantic_chunk_documents
# ---------------------------------------------------------------------------

class TestSemanticChunkDocuments:
    def test_returns_list_of_chunks(self, sample_documents):
        chunks = semantic_chunk_documents(sample_documents)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_are_unique(self, sample_documents):
        chunks = semantic_chunk_documents(sample_documents)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_include_sem_suffix(self, sample_document):
        chunks = semantic_chunk_documents([sample_document])
        assert all("-SEM-" in c.chunk_id for c in chunks)

    def test_metadata_propagates(self):
        doc = _make_doc(doc_id="D-42", section="Travel Approval", text="Request travel. Get approved.")
        chunks = semantic_chunk_documents([doc])
        for chunk in chunks:
            assert chunk.doc_id == "D-42"
            assert chunk.section == "Travel Approval"

    def test_short_document_produces_single_chunk(self):
        doc = _make_doc(text="Only one sentence here")
        chunks = semantic_chunk_documents([doc])
        assert len(chunks) == 1

    def test_long_document_splits_into_two_groups(self):
        # Three or more sentences → two groups
        doc = _make_doc(
            text="First sentence. Second sentence. Third sentence. Fourth sentence."
        )
        chunks = semantic_chunk_documents([doc])
        assert len(chunks) == 2

    def test_empty_documents_list(self):
        assert semantic_chunk_documents([]) == []

    def test_text_is_not_empty_in_chunks(self, sample_documents):
        chunks = semantic_chunk_documents(sample_documents)
        assert all(len(c.text) > 0 for c in chunks)
