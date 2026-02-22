"""Tests for data_generation.py — parsing, generation, and serialisation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_tutorials.data_generation import (
    HANDBOOK_TEXT,
    SECTIONS,
    build_and_save_dataset,
    generate_documents,
    generate_queries,
    parse_handbook_to_documents,
    save_dataset,
)
from rag_tutorials.schema import Document, QueryExample


# ---------------------------------------------------------------------------
# parse_handbook_to_documents
# ---------------------------------------------------------------------------

class TestParseHandbookToDocuments:
    def test_returns_list_of_documents(self):
        docs = parse_handbook_to_documents(HANDBOOK_TEXT)
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_section_count_matches_handbook(self):
        docs = parse_handbook_to_documents(HANDBOOK_TEXT)
        # HANDBOOK_TEXT has exactly 5 ## headings
        assert len(docs) == len(SECTIONS)

    def test_section_names_match_constants(self):
        docs = parse_handbook_to_documents(HANDBOOK_TEXT)
        parsed_sections = {d.section for d in docs}
        assert parsed_sections == set(SECTIONS)

    def test_doc_ids_are_unique(self):
        docs = parse_handbook_to_documents(HANDBOOK_TEXT)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_doc_id_format(self):
        docs = parse_handbook_to_documents(HANDBOOK_TEXT)
        for doc in docs:
            assert doc.doc_id.startswith("DOC-HB-")

    def test_text_is_non_empty(self):
        docs = parse_handbook_to_documents(HANDBOOK_TEXT)
        for doc in docs:
            assert len(doc.text) > 0

    def test_minimal_handbook_text(self):
        minimal = "# Handbook\n\n## Remote Work\nWork from home allowed.\n"
        docs = parse_handbook_to_documents(minimal)
        assert len(docs) == 1
        assert docs[0].section == "Remote Work"
        assert "Work from home" in docs[0].text

    def test_empty_string_returns_empty(self):
        docs = parse_handbook_to_documents("")
        assert docs == []


# ---------------------------------------------------------------------------
# generate_documents
# ---------------------------------------------------------------------------

class TestGenerateDocuments:
    def test_returns_documents(self):
        docs = generate_documents()
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_result_matches_parse_handbook(self):
        docs = generate_documents()
        expected = parse_handbook_to_documents(HANDBOOK_TEXT)
        assert len(docs) == len(expected)
        assert {d.doc_id for d in docs} == {d.doc_id for d in expected}


# ---------------------------------------------------------------------------
# generate_queries
# ---------------------------------------------------------------------------

class TestGenerateQueries:
    @pytest.fixture()
    def documents(self):
        return generate_documents()

    def test_returns_query_examples(self, documents):
        queries = generate_queries(documents, query_count=10)
        assert len(queries) == 10
        assert all(isinstance(q, QueryExample) for q in queries)

    def test_query_ids_are_unique(self, documents):
        queries = generate_queries(documents, query_count=20)
        ids = [q.query_id for q in queries]
        assert len(ids) == len(set(ids))

    def test_target_doc_ids_are_valid(self, documents):
        valid_ids = {d.doc_id for d in documents}
        queries = generate_queries(documents, query_count=20)
        for q in queries:
            assert q.target_doc_id in valid_ids

    def test_deterministic_with_same_seed(self, documents):
        q1 = generate_queries(documents, query_count=10, seed=42)
        q2 = generate_queries(documents, query_count=10, seed=42)
        assert [q.query_id for q in q1] == [q.query_id for q in q2]
        assert [q.question for q in q1] == [q.question for q in q2]

    def test_different_seeds_change_results_when_multiple_docs(self):
        # Build a corpus where every section has TWO documents so random.choice
        # actually has a choice and the seed can change the outcome.
        # International Tax docs must contain "Form <code>" for generate_queries.
        from rag_tutorials.data_generation import SECTIONS

        form_codes = ["A-12", "TX-88"]
        docs = []
        for section in SECTIONS:
            for i in range(2):
                if section == "International Tax":
                    text = f"Employees must submit Form {form_codes[i]} before travel."
                else:
                    text = f"{section} text variant {i}."
                docs.append(
                    Document(
                        doc_id=f"DOC-{section.replace(' ', '')}-{i}",
                        title=section,
                        section=section,
                        text=text,
                    )
                )
        q1 = generate_queries(docs, query_count=10, seed=1)
        q2 = generate_queries(docs, query_count=10, seed=42)
        targets1 = [q.target_doc_id for q in q1]
        targets2 = [q.target_doc_id for q in q2]
        # With two docs per section, different seeds must produce different picks
        assert targets1 != targets2

    def test_rationale_is_non_empty(self, documents):
        queries = generate_queries(documents, query_count=5)
        for q in queries:
            assert len(q.rationale) > 0


# ---------------------------------------------------------------------------
# save_dataset
# ---------------------------------------------------------------------------

class TestSaveDataset:
    def test_creates_documents_and_queries_files(self, tmp_path):
        docs = generate_documents()
        queries = generate_queries(docs, query_count=5)
        save_dataset(docs, queries, output_dir=str(tmp_path))

        assert (tmp_path / "documents.jsonl").exists()
        assert (tmp_path / "queries.jsonl").exists()

    def test_documents_file_is_valid_jsonl(self, tmp_path):
        docs = generate_documents()
        save_dataset(docs, [], output_dir=str(tmp_path))
        lines = (tmp_path / "documents.jsonl").read_text().splitlines()
        assert len(lines) == len(docs)
        for line in lines:
            obj = json.loads(line)
            assert "doc_id" in obj

    def test_queries_file_is_valid_jsonl(self, tmp_path):
        docs = generate_documents()
        queries = generate_queries(docs, query_count=3)
        save_dataset(docs, queries, output_dir=str(tmp_path))
        lines = (tmp_path / "queries.jsonl").read_text().splitlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "query_id" in obj


# ---------------------------------------------------------------------------
# build_and_save_dataset
# ---------------------------------------------------------------------------

class TestBuildAndSaveDataset:
    def test_creates_handbook_and_jsonl_files(self, tmp_path):
        build_and_save_dataset(output_dir=str(tmp_path), query_count=5)
        assert (tmp_path / "handbook_manual.txt").exists()
        assert (tmp_path / "documents.jsonl").exists()
        assert (tmp_path / "queries.jsonl").exists()

    def test_handbook_text_content(self, tmp_path):
        build_and_save_dataset(output_dir=str(tmp_path))
        content = (tmp_path / "handbook_manual.txt").read_text()
        assert "Remote Work" in content
        assert "International Work" in content
