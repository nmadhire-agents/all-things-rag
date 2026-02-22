"""Tests for qa.py — build_context (pure) and answer_with_context (mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_tutorials.qa import answer_with_context, build_context


# ---------------------------------------------------------------------------
# build_context — pure function, no mocking needed
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_single_chunk_format(self):
        result = build_context(["Hello policy."])
        assert result == "Chunk 1: Hello policy."

    def test_multiple_chunks_numbered(self):
        result = build_context(["First.", "Second.", "Third."])
        assert "Chunk 1: First." in result
        assert "Chunk 2: Second." in result
        assert "Chunk 3: Third." in result

    def test_chunks_separated_by_double_newline(self):
        result = build_context(["A", "B"])
        assert "\n\n" in result

    def test_empty_list_returns_empty_string(self):
        assert build_context([]) == ""

    def test_numbering_starts_at_1(self):
        result = build_context(["only chunk"])
        assert result.startswith("Chunk 1:")

    def test_preserves_chunk_text_exactly(self):
        text = "  some text with spaces  "
        result = build_context([text])
        assert text in result


# ---------------------------------------------------------------------------
# answer_with_context — OpenAI Responses API mocked
# ---------------------------------------------------------------------------

class TestAnswerWithContext:
    def _make_mock_client(self, output_text: str = "The policy allows it. [Chunk 1]") -> MagicMock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = output_text
        mock_client.responses.create.return_value = mock_response
        return mock_client

    @patch("rag_tutorials.qa.OpenAI")
    def test_returns_string(self, mock_openai_cls):
        mock_openai_cls.return_value = self._make_mock_client()
        result = answer_with_context("What is allowed?", ["Policy text here."])
        assert isinstance(result, str)

    @patch("rag_tutorials.qa.OpenAI")
    def test_returns_model_output_text(self, mock_openai_cls):
        expected = "Remote work is allowed. [Chunk 1]"
        mock_openai_cls.return_value = self._make_mock_client(expected)
        result = answer_with_context("Can I work remotely?", ["Remote work allowed."])
        assert result == expected

    @patch("rag_tutorials.qa.OpenAI")
    def test_passes_model_name_to_api(self, mock_openai_cls):
        mock_client = self._make_mock_client()
        mock_openai_cls.return_value = mock_client
        answer_with_context("Q?", ["context"], model="gpt-4o")
        call_kwargs = mock_client.responses.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    @patch("rag_tutorials.qa.OpenAI")
    def test_prompt_includes_question(self, mock_openai_cls):
        mock_client = self._make_mock_client()
        mock_openai_cls.return_value = mock_client
        question = "What are the remote work rules?"
        answer_with_context(question, ["some context"])
        prompt_sent = mock_client.responses.create.call_args.kwargs["input"]
        assert question in prompt_sent

    @patch("rag_tutorials.qa.OpenAI")
    def test_prompt_includes_context_chunks(self, mock_openai_cls):
        mock_client = self._make_mock_client()
        mock_openai_cls.return_value = mock_client
        answer_with_context("Q?", ["Chunk text alpha.", "Chunk text beta."])
        prompt_sent = mock_client.responses.create.call_args.kwargs["input"]
        assert "Chunk text alpha." in prompt_sent
        assert "Chunk text beta." in prompt_sent

    @patch("rag_tutorials.qa.OpenAI")
    def test_empty_context_list(self, mock_openai_cls):
        mock_openai_cls.return_value = self._make_mock_client("No context.")
        result = answer_with_context("Q?", [])
        assert isinstance(result, str)
