"""Tests for tracing.py - OpenTelemetry span recording utilities.

These tests use the in-memory exporter and do not require a Phoenix server
or an OpenAI API key.
"""
from __future__ import annotations

import time

import pytest

from rag_tutorials.schema import RetrievalResult
from rag_tutorials.tracing import (
    build_in_memory_tracer,
    record_agent_step_span,
    record_generation_span,
    record_retrieval_span,
    spans_to_dicts,
)


# ---------------------------------------------------------------------------
# build_in_memory_tracer
# ---------------------------------------------------------------------------


class TestBuildInMemoryTracer:
    def test_returns_tracer_and_exporter(self):
        tracer, exporter = build_in_memory_tracer()
        assert tracer is not None
        assert exporter is not None

    def test_starts_with_no_spans(self):
        _, exporter = build_in_memory_tracer()
        assert exporter.get_finished_spans() == ()

    def test_span_is_captured(self):
        tracer, exporter = build_in_memory_tracer()
        with tracer.start_as_current_span("test_span"):
            pass
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test_span"

    def test_multiple_tracers_are_independent(self):
        tracer1, exporter1 = build_in_memory_tracer()
        tracer2, exporter2 = build_in_memory_tracer()
        with tracer1.start_as_current_span("span_a"):
            pass
        assert len(exporter1.get_finished_spans()) == 1
        assert len(exporter2.get_finished_spans()) == 0


# ---------------------------------------------------------------------------
# record_retrieval_span
# ---------------------------------------------------------------------------


class TestRecordRetrievalSpan:
    def test_records_span_with_correct_name(self):
        tracer, exporter = build_in_memory_tracer()
        record_retrieval_span(tracer, query="leave policy", results=[], top_k=5)
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "retrieval"

    def test_records_query_attribute(self):
        tracer, exporter = build_in_memory_tracer()
        record_retrieval_span(tracer, query="remote work", results=[], top_k=3)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["retrieval.query"] == "remote work"

    def test_records_top_k_attribute(self):
        tracer, exporter = build_in_memory_tracer()
        record_retrieval_span(tracer, query="q", results=[], top_k=7)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["retrieval.top_k"] == 7

    def test_records_result_count_when_empty(self):
        tracer, exporter = build_in_memory_tracer()
        record_retrieval_span(tracer, query="q", results=[], top_k=5)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["retrieval.result_count"] == 0

    def test_records_result_count_with_results(self):
        tracer, exporter = build_in_memory_tracer()
        results = [
            RetrievalResult(chunk_id="C1", score=0.9, source="dense", text="chunk one"),
            RetrievalResult(chunk_id="C2", score=0.8, source="dense", text="chunk two"),
        ]
        record_retrieval_span(tracer, query="q", results=results, top_k=5)
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["retrieval.result_count"] == 2

    def test_records_top_score_when_results_present(self):
        tracer, exporter = build_in_memory_tracer()
        results = [RetrievalResult(chunk_id="C1", score=0.75, source="dense", text="text")]
        record_retrieval_span(tracer, query="q", results=results, top_k=5)
        spans = exporter.get_finished_spans()
        assert abs(spans[0].attributes["retrieval.top_score"] - 0.75) < 1e-6

    def test_no_top_score_when_results_empty(self):
        tracer, exporter = build_in_memory_tracer()
        record_retrieval_span(tracer, query="q", results=[], top_k=5)
        spans = exporter.get_finished_spans()
        assert "retrieval.top_score" not in spans[0].attributes


# ---------------------------------------------------------------------------
# record_generation_span
# ---------------------------------------------------------------------------


class TestRecordGenerationSpan:
    def test_records_span_with_correct_name(self):
        tracer, exporter = build_in_memory_tracer()
        record_generation_span(
            tracer, question="q", answer="a", model="gpt-4", context_chunk_count=3
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "generation"

    def test_records_question_attribute(self):
        tracer, exporter = build_in_memory_tracer()
        record_generation_span(
            tracer, question="What is the policy?", answer="a", model="m", context_chunk_count=1
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["generation.question"] == "What is the policy?"

    def test_records_model_attribute(self):
        tracer, exporter = build_in_memory_tracer()
        record_generation_span(
            tracer, question="q", answer="a", model="gpt-4.1-mini", context_chunk_count=2
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["generation.model"] == "gpt-4.1-mini"

    def test_records_answer_word_count(self):
        tracer, exporter = build_in_memory_tracer()
        record_generation_span(
            tracer, question="q", answer="one two three", model="m", context_chunk_count=1
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["generation.answer_word_count"] == 3

    def test_records_context_chunk_count(self):
        tracer, exporter = build_in_memory_tracer()
        record_generation_span(
            tracer, question="q", answer="a", model="m", context_chunk_count=5
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["generation.context_chunk_count"] == 5

    def test_empty_answer_has_zero_word_count(self):
        tracer, exporter = build_in_memory_tracer()
        record_generation_span(
            tracer, question="q", answer="", model="m", context_chunk_count=0
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["generation.answer_word_count"] == 0


# ---------------------------------------------------------------------------
# record_agent_step_span
# ---------------------------------------------------------------------------


class TestRecordAgentStepSpan:
    def test_records_span_with_correct_name(self):
        tracer, exporter = build_in_memory_tracer()
        record_agent_step_span(
            tracer,
            step_number=1,
            thought="t",
            action="retrieve",
            action_input="q",
            observation="o",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "agent_step"

    def test_records_step_number(self):
        tracer, exporter = build_in_memory_tracer()
        record_agent_step_span(
            tracer,
            step_number=3,
            thought="t",
            action="a",
            action_input="i",
            observation="o",
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["agent.step_number"] == 3

    def test_records_thought(self):
        tracer, exporter = build_in_memory_tracer()
        record_agent_step_span(
            tracer,
            step_number=1,
            thought="I need to look this up.",
            action="a",
            action_input="i",
            observation="o",
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["agent.thought"] == "I need to look this up."

    def test_records_action(self):
        tracer, exporter = build_in_memory_tracer()
        record_agent_step_span(
            tracer,
            step_number=1,
            thought="t",
            action="retrieve",
            action_input="q",
            observation="o",
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["agent.action"] == "retrieve"

    def test_records_action_input(self):
        tracer, exporter = build_in_memory_tracer()
        record_agent_step_span(
            tracer,
            step_number=1,
            thought="t",
            action="a",
            action_input="international work limit",
            observation="o",
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["agent.action_input"] == "international work limit"

    def test_records_observation_length(self):
        tracer, exporter = build_in_memory_tracer()
        observation = "hello world"
        record_agent_step_span(
            tracer,
            step_number=1,
            thought="t",
            action="a",
            action_input="i",
            observation=observation,
        )
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["agent.observation_length"] == len(observation)


# ---------------------------------------------------------------------------
# spans_to_dicts
# ---------------------------------------------------------------------------


class TestSpansToDicts:
    def test_empty_input_returns_empty_list(self):
        assert spans_to_dicts([]) == []

    def test_returns_list_of_dicts(self):
        tracer, exporter = build_in_memory_tracer()
        with tracer.start_as_current_span("test"):
            pass
        result = spans_to_dicts(exporter.get_finished_spans())
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_dict_has_expected_keys(self):
        tracer, exporter = build_in_memory_tracer()
        with tracer.start_as_current_span("my_span"):
            pass
        result = spans_to_dicts(exporter.get_finished_spans())
        assert set(result[0].keys()) == {"name", "status", "duration_ms", "attributes"}

    def test_name_matches_span(self):
        tracer, exporter = build_in_memory_tracer()
        with tracer.start_as_current_span("unique_name"):
            pass
        result = spans_to_dicts(exporter.get_finished_spans())
        assert result[0]["name"] == "unique_name"

    def test_duration_ms_is_non_negative(self):
        tracer, exporter = build_in_memory_tracer()
        with tracer.start_as_current_span("timed"):
            time.sleep(0.01)
        result = spans_to_dicts(exporter.get_finished_spans())
        assert result[0]["duration_ms"] >= 0

    def test_attributes_dict_matches_set_attributes(self):
        tracer, exporter = build_in_memory_tracer()
        with tracer.start_as_current_span("with_attrs") as span:
            span.set_attribute("foo", "bar")
            span.set_attribute("num", 42)
        result = spans_to_dicts(exporter.get_finished_spans())
        assert result[0]["attributes"]["foo"] == "bar"
        assert result[0]["attributes"]["num"] == 42

    def test_multiple_spans_returns_all(self):
        tracer, exporter = build_in_memory_tracer()
        with tracer.start_as_current_span("s1"):
            pass
        with tracer.start_as_current_span("s2"):
            pass
        result = spans_to_dicts(exporter.get_finished_spans())
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"s1", "s2"}

    def test_status_is_string(self):
        tracer, exporter = build_in_memory_tracer()
        with tracer.start_as_current_span("ok_span"):
            pass
        result = spans_to_dicts(exporter.get_finished_spans())
        assert isinstance(result[0]["status"], str)
