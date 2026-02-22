from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    """Section-level source document derived from handbook input text."""

    doc_id: str
    title: str
    section: str
    text: str


@dataclass(slots=True)
class Chunk:
    """Chunked segment of a source document used for retrieval."""

    chunk_id: str
    doc_id: str
    section: str
    text: str


@dataclass(slots=True)
class QueryExample:
    """Evaluation query with expected relevance targets and rationale."""

    query_id: str
    question: str
    relevant_chunk_ids: list[str]
    target_doc_id: str
    target_section: str
    rationale: str


@dataclass(slots=True)
class RetrievalResult:
    """Standard retrieval output used across dense, BM25, and hybrid paths."""

    chunk_id: str
    score: float
    source: str
    text: str
