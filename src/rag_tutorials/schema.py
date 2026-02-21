from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    section: str
    text: str


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str


@dataclass(slots=True)
class QueryExample:
    query_id: str
    question: str
    relevant_chunk_ids: list[str]
    target_doc_id: str
    target_section: str
    rationale: str


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    score: float
    source: str
    text: str
