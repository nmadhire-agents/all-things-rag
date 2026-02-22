"""Tests for io_utils.py — JSONL load/save helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_tutorials.data_generation import HANDBOOK_TEXT
from rag_tutorials.io_utils import (
    load_chunks,
    load_handbook_documents,
    load_queries,
    save_chunks,
)
from rag_tutorials.schema import Chunk, Document, QueryExample


# ---------------------------------------------------------------------------
# load_handbook_documents
# ---------------------------------------------------------------------------

class TestLoadHandbookDocuments:
    def test_loads_from_file(self, tmp_path):
        handbook_file = tmp_path / "handbook.txt"
        handbook_file.write_text(HANDBOOK_TEXT, encoding="utf-8")
        docs = load_handbook_documents(handbook_file)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_sections_present(self, tmp_path):
        handbook_file = tmp_path / "handbook.txt"
        handbook_file.write_text(HANDBOOK_TEXT, encoding="utf-8")
        docs = load_handbook_documents(handbook_file)
        sections = {d.section for d in docs}
        assert "Remote Work" in sections
        assert "Security" in sections


# ---------------------------------------------------------------------------
# load_queries
# ---------------------------------------------------------------------------

class TestLoadQueries:
    def _write_queries(self, tmp_path: Path, count: int = 3) -> Path:
        path = tmp_path / "queries.jsonl"
        with path.open("w") as f:
            for i in range(count):
                record = {
                    "query_id": f"Q-{i:04d}",
                    "question": f"Question {i}?",
                    "relevant_chunk_ids": [],
                    "target_doc_id": "DOC-001",
                    "target_section": "Remote Work",
                    "rationale": "Test rationale.",
                }
                f.write(json.dumps(record) + "\n")
        return path

    def test_loads_correct_count(self, tmp_path):
        path = self._write_queries(tmp_path, count=5)
        queries = load_queries(path)
        assert len(queries) == 5

    def test_returns_query_examples(self, tmp_path):
        path = self._write_queries(tmp_path, count=2)
        queries = load_queries(path)
        assert all(isinstance(q, QueryExample) for q in queries)

    def test_fields_preserved(self, tmp_path):
        path = self._write_queries(tmp_path, count=1)
        q = load_queries(path)[0]
        assert q.query_id == "Q-0000"
        assert q.question == "Question 0?"
        assert q.target_doc_id == "DOC-001"


# ---------------------------------------------------------------------------
# save_chunks / load_chunks roundtrip
# ---------------------------------------------------------------------------

class TestSaveAndLoadChunks:
    def test_roundtrip_preserves_all_fields(self, tmp_path, sample_chunks):
        path = tmp_path / "chunks.jsonl"
        save_chunks(sample_chunks, path)
        loaded = load_chunks(path)

        assert len(loaded) == len(sample_chunks)
        for original, restored in zip(sample_chunks, loaded):
            assert restored.chunk_id == original.chunk_id
            assert restored.doc_id == original.doc_id
            assert restored.section == original.section
            assert restored.text == original.text

    def test_save_creates_file(self, tmp_path, sample_chunks):
        path = tmp_path / "out.jsonl"
        save_chunks(sample_chunks, path)
        assert path.exists()

    def test_saved_file_is_valid_jsonl(self, tmp_path, sample_chunks):
        path = tmp_path / "chunks.jsonl"
        save_chunks(sample_chunks, path)
        lines = path.read_text().splitlines()
        assert len(lines) == len(sample_chunks)
        for line in lines:
            obj = json.loads(line)
            assert "chunk_id" in obj

    def test_save_creates_parent_directories(self, tmp_path):
        nested_path = tmp_path / "a" / "b" / "chunks.jsonl"
        chunk = Chunk(chunk_id="C-1", doc_id="D-1", section="S", text="hello")
        save_chunks([chunk], nested_path)
        assert nested_path.exists()

    def test_empty_list_creates_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        save_chunks([], path)
        assert path.read_text() == ""
