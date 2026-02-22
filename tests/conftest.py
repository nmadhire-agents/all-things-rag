"""Shared pytest fixtures for rag_tutorials unit tests."""
from __future__ import annotations

import pytest

from rag_tutorials.schema import Chunk, Document, QueryExample, RetrievalResult


@pytest.fixture()
def sample_document() -> Document:
    return Document(
        doc_id="DOC-001",
        title="Test Doc",
        section="Remote Work",
        text="Employees may work remotely from home. VPN is required for all connections.",
    )


@pytest.fixture()
def sample_documents() -> list[Document]:
    return [
        Document(
            doc_id="DOC-001",
            title="Remote Work Policy",
            section="Remote Work",
            text="Employees may work remotely from home. VPN is required for all connections.",
        ),
        Document(
            doc_id="DOC-002",
            title="International Work Policy",
            section="International Work",
            text=(
                "Working from another country is capped at 14 days. "
                "Beyond 14 days, employees must open a Global Mobility case."
            ),
        ),
        Document(
            doc_id="DOC-003",
            title="Security Policy",
            section="Security",
            text="Lost devices must be reported within one hour. Use VPN and encrypted storage.",
        ),
    ]


@pytest.fixture()
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="DOC-001-FIX-00",
        doc_id="DOC-001",
        section="Remote Work",
        text="Employees may work remotely from home.",
    )


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="DOC-001-FIX-00",
            doc_id="DOC-001",
            section="Remote Work",
            text="Employees may work remotely from home.",
        ),
        Chunk(
            chunk_id="DOC-002-FIX-00",
            doc_id="DOC-002",
            section="International Work",
            text="Working from another country is capped at 14 days.",
        ),
        Chunk(
            chunk_id="DOC-003-FIX-00",
            doc_id="DOC-003",
            section="Security",
            text="Lost devices must be reported within one hour.",
        ),
    ]


@pytest.fixture()
def sample_query() -> QueryExample:
    return QueryExample(
        query_id="Q-0001",
        question="What is the remote work policy?",
        relevant_chunk_ids=["DOC-001-FIX-00"],
        target_doc_id="DOC-001",
        target_section="Remote Work",
        rationale="Requires remote work policy context.",
    )


@pytest.fixture()
def sample_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="DOC-001-FIX-00",
        score=0.92,
        source="dense",
        text="Employees may work remotely from home.",
    )


@pytest.fixture()
def sample_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="DOC-002-FIX-00",
            score=0.85,
            source="dense",
            text="Working from another country is capped at 14 days.",
        ),
        RetrievalResult(
            chunk_id="DOC-001-FIX-00",
            score=0.72,
            source="dense",
            text="Employees may work remotely from home.",
        ),
        RetrievalResult(
            chunk_id="DOC-003-FIX-00",
            score=0.60,
            source="dense",
            text="Lost devices must be reported within one hour.",
        ),
    ]
