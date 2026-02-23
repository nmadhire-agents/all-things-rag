"""Tests for agent_loop.py — run_react_loop, AgentStep, AgentResult.

OpenAI calls are mocked so no API key is needed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rag_tutorials.agent_loop import AgentResult, AgentStep, run_react_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_response(content: str):
    """Build a minimal fake OpenAI chat response object."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _finish_response(answer: str):
    payload = json.dumps({"thought": "I have the answer.", "action": "finish", "action_input": answer})
    return _make_chat_response(payload)


def _tool_response(tool: str, tool_input: str):
    payload = json.dumps({"thought": "Let me look this up.", "action": tool, "action_input": tool_input})
    return _make_chat_response(payload)


# ---------------------------------------------------------------------------
# AgentStep and AgentResult dataclasses
# ---------------------------------------------------------------------------


class TestAgentDataclasses:
    def test_agent_step_fields(self):
        step = AgentStep(thought="t", action="retrieve", action_input="query", observation="result")
        assert step.thought == "t"
        assert step.action == "retrieve"
        assert step.action_input == "query"
        assert step.observation == "result"

    def test_agent_result_defaults_empty_steps(self):
        result = AgentResult(question="q", answer="a")
        assert result.steps == []

    def test_agent_result_with_steps(self):
        step = AgentStep(thought="t", action="finish", action_input="done", observation="")
        result = AgentResult(question="q", answer="done", steps=[step])
        assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# run_react_loop
# ---------------------------------------------------------------------------


class TestRunReactLoop:
    def test_immediate_finish(self):
        """Agent finishes on the first LLM call."""
        with patch("rag_tutorials.agent_loop.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _finish_response("42 days")

            result = run_react_loop("How many days?", tools={})

        assert isinstance(result, AgentResult)
        assert result.answer == "42 days"
        assert result.steps == []

    def test_one_tool_call_then_finish(self):
        """Agent calls a tool once, then finishes."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("retrieve", "leave policy")
            return _finish_response("14 days leave")

        with patch("rag_tutorials.agent_loop.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = side_effect

            fake_tool = MagicMock(return_value="Employees get 14 days leave.")
            result = run_react_loop("What is the leave policy?", tools={"retrieve": fake_tool})

        assert result.answer == "14 days leave"
        assert len(result.steps) == 1
        assert result.steps[0].action == "retrieve"
        fake_tool.assert_called_once_with("leave policy")

    def test_unknown_tool_produces_error_observation(self):
        """Calling an unknown tool results in an error observation, not a crash."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("nonexistent_tool", "input")
            return _finish_response("done")

        with patch("rag_tutorials.agent_loop.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = side_effect

            result = run_react_loop("question", tools={})

        assert len(result.steps) == 1
        assert "Unknown tool" in result.steps[0].observation

    def test_malformed_json_returns_raw_text(self):
        """If the LLM returns non-JSON, run_react_loop returns it as the answer."""
        with patch("rag_tutorials.agent_loop.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_chat_response("not json at all")

            result = run_react_loop("question", tools={})

        assert result.answer == "not json at all"

    def test_max_steps_terminates_loop(self):
        """Loop terminates after max_steps even if agent never finishes."""
        with patch("rag_tutorials.agent_loop.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            # Always return a tool call, never finish
            mock_client.chat.completions.create.return_value = _tool_response("retrieve", "q")

            fake_tool = MagicMock(return_value="some observation")
            result = run_react_loop("q", tools={"retrieve": fake_tool}, max_steps=3)

        assert len(result.steps) == 3

    def test_tool_exception_is_caught(self):
        """A tool that raises an exception produces an error observation."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("bad_tool", "input")
            return _finish_response("done")

        with patch("rag_tutorials.agent_loop.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = side_effect

            def exploding_tool(x):
                raise RuntimeError("boom")

            result = run_react_loop("q", tools={"bad_tool": exploding_tool})

        assert "Tool error" in result.steps[0].observation
