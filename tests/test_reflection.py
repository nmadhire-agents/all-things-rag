"""Tests for reflection.py — worker_answer, critic_review, run_reflection_loop.

OpenAI calls are mocked so no API key is needed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rag_tutorials.reflection import (
    CriticFeedback,
    ReflectionResult,
    critic_review,
    run_reflection_loop,
    worker_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _approved_response():
    return _make_chat_response(json.dumps({"approved": True, "feedback": ""}))


def _rejected_response(feedback: str):
    return _make_chat_response(json.dumps({"approved": False, "feedback": feedback}))


# ---------------------------------------------------------------------------
# CriticFeedback dataclass
# ---------------------------------------------------------------------------


class TestCriticFeedback:
    def test_approved_true(self):
        fb = CriticFeedback(approved=True, feedback="")
        assert fb.approved is True
        assert fb.feedback == ""

    def test_approved_false_has_feedback(self):
        fb = CriticFeedback(approved=False, feedback="Missing citation.")
        assert not fb.approved
        assert "Missing" in fb.feedback


# ---------------------------------------------------------------------------
# ReflectionResult dataclass
# ---------------------------------------------------------------------------


class TestReflectionResult:
    def test_fields_present(self):
        result = ReflectionResult(
            question="q",
            final_answer="a",
            rounds=1,
            history=[{"round": 1, "answer": "a", "approved": True, "feedback": ""}],
        )
        assert result.rounds == 1
        assert len(result.history) == 1

    def test_default_history_empty(self):
        result = ReflectionResult(question="q", final_answer="a", rounds=0)
        assert result.history == []


# ---------------------------------------------------------------------------
# worker_answer
# ---------------------------------------------------------------------------


class TestWorkerAnswer:
    def test_returns_string(self):
        with patch("rag_tutorials.reflection.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_chat_response("The policy allows 14 days.")
            answer = worker_answer("What is the policy?", context="14 days allowed.")
        assert answer == "The policy allows 14 days."

    def test_includes_feedback_in_prompt_when_provided(self):
        with patch("rag_tutorials.reflection.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_chat_response("Revised answer.")
            worker_answer("q", context="ctx", feedback="Add a citation.")
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            user_message = next(m["content"] for m in messages if m["role"] == "user")
            assert "Critic feedback" in user_message


# ---------------------------------------------------------------------------
# critic_review
# ---------------------------------------------------------------------------


class TestCriticReview:
    def test_approved_true_when_llm_approves(self):
        with patch("rag_tutorials.reflection.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _approved_response()
            result = critic_review("q", "good answer", "context")
        assert result.approved is True
        assert result.feedback == ""

    def test_approved_false_with_feedback(self):
        with patch("rag_tutorials.reflection.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _rejected_response("Needs citation.")
            result = critic_review("q", "weak answer", "context")
        assert result.approved is False
        assert "citation" in result.feedback.lower()

    def test_malformed_json_returns_approved(self):
        """Parsing failure should not crash; defaults to approved=True."""
        with patch("rag_tutorials.reflection.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_chat_response("not valid json")
            result = critic_review("q", "answer", "ctx")
        assert result.approved is True


# ---------------------------------------------------------------------------
# run_reflection_loop
# ---------------------------------------------------------------------------


class TestRunReflectionLoop:
    def test_approves_on_first_round(self):
        with patch("rag_tutorials.reflection.worker_answer", return_value="good answer"), \
             patch("rag_tutorials.reflection.critic_review",
                   return_value=CriticFeedback(approved=True, feedback="")):
            result = run_reflection_loop("q", context="ctx")
        assert result.rounds == 1
        assert result.final_answer == "good answer"
        assert result.history[0]["approved"] is True

    def test_revises_on_second_round(self):
        answers = ["first draft", "revised draft"]
        answer_iter = iter(answers)

        def fake_worker(question, context, feedback="", model="gpt-4.1-mini"):
            return next(answer_iter)

        feedbacks = [
            CriticFeedback(approved=False, feedback="Needs improvement."),
            CriticFeedback(approved=True, feedback=""),
        ]
        feedback_iter = iter(feedbacks)

        def fake_critic(question, answer, context, model="gpt-4.1-mini"):
            return next(feedback_iter)

        with patch("rag_tutorials.reflection.worker_answer", side_effect=fake_worker), \
             patch("rag_tutorials.reflection.critic_review", side_effect=fake_critic):
            result = run_reflection_loop("q", context="ctx")

        assert result.rounds == 2
        assert result.final_answer == "revised draft"
        assert len(result.history) == 2

    def test_stops_at_max_rounds(self):
        """Loop ends after max_rounds even if never approved."""
        with patch("rag_tutorials.reflection.worker_answer", return_value="draft"), \
             patch("rag_tutorials.reflection.critic_review",
                   return_value=CriticFeedback(approved=False, feedback="Still wrong.")):
            result = run_reflection_loop("q", context="ctx", max_rounds=2)
        assert result.rounds == 2
        assert len(result.history) == 2

    def test_history_contains_round_numbers(self):
        feedbacks = [
            CriticFeedback(approved=False, feedback="bad"),
            CriticFeedback(approved=True, feedback=""),
        ]
        fb_iter = iter(feedbacks)
        with patch("rag_tutorials.reflection.worker_answer", return_value="a"), \
             patch("rag_tutorials.reflection.critic_review", side_effect=lambda *a, **kw: next(fb_iter)):
            result = run_reflection_loop("q", context="ctx")
        round_numbers = [h["round"] for h in result.history]
        assert round_numbers == [1, 2]
