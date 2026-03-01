"""Tests for tracing.py — configure_tracing, get_tracer, traced_retrieval,
traced_generation, build_traced_rag_pipeline.

No OpenAI API key is needed.  OTel spans are collected with InMemorySpanExporter
so tests run fully offline.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_tutorials.schema import RetrievalResult
from rag_tutorials.tracing import (
    ATTR_INPUT_VALUE,
    ATTR_LLM_MODEL_NAME,
    ATTR_OUTPUT_VALUE,
    ATTR_RETRIEVAL_DOCUMENTS,
    build_traced_rag_pipeline,
    configure_tracing,
    get_tracer,
    traced_generation,
    traced_retrieval,
)


# ---------------------------------------------------------------------------
# Shared in-memory exporter fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem_exporter() -> InMemorySpanExporter:
    """Fresh InMemorySpanExporter and a configured global TracerProvider."""
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter, service_name="test-service")
    return exporter


# ---------------------------------------------------------------------------
# configure_tracing
# ---------------------------------------------------------------------------


class TestConfigureTracing:
    def test_returns_tracer_provider(self, mem_exporter):
        from opentelemetry.sdk.trace import TracerProvider

        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)

    def test_console_exporter_used_when_no_endpoint(self):
        """configure_tracing with no args must not raise."""
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        exporter = ConsoleSpanExporter()
        provider = configure_tracing(exporter=exporter, service_name="console-test")
        assert provider is not None

    def test_missing_otlp_package_raises_import_error(self, monkeypatch):
        """When otlp exporter package is absent, a helpful ImportError is raised."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "otlp" in name:
                raise ImportError("mocked missing package")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="opentelemetry-exporter-otlp"):
            configure_tracing(endpoint="http://localhost:6006/v1/traces")


# ---------------------------------------------------------------------------
# get_tracer
# ---------------------------------------------------------------------------


class TestGetTracer:
    def test_returns_tracer(self, mem_exporter):
        from opentelemetry.trace import Tracer

        tracer = get_tracer("test.component")
        assert tracer is not None
        # The returned object must expose the span-creation interface
        assert hasattr(tracer, "start_as_current_span")

    def test_different_names_give_different_scope(self, mem_exporter):
        t1 = get_tracer("component.a")
        t2 = get_tracer("component.b")
        # Both are valid tracers; names are different scopes
        assert t1 is not t2


# ---------------------------------------------------------------------------
# traced_retrieval
# ---------------------------------------------------------------------------


class TestTracedRetrieval:
    def _fake_retriever(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=f"C-{i}",
                score=1.0 - i * 0.1,
                source="dense",
                text=f"chunk {i} text",
            )
            for i in range(top_k)
        ]

    def test_returns_same_results(self, mem_exporter):
        tracer = get_tracer("retrieval")
        wrapped = traced_retrieval(self._fake_retriever, tracer)
        results = wrapped("vpn policy", top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "C-0"

    def test_span_is_created(self, mem_exporter):
        tracer = get_tracer("retrieval")
        wrapped = traced_retrieval(self._fake_retriever, tracer)
        wrapped("vpn policy", top_k=3)

        spans = mem_exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert "retrieval" in span_names

    def test_span_records_input_value(self, mem_exporter):
        tracer = get_tracer("retrieval")
        wrapped = traced_retrieval(self._fake_retriever, tracer)
        wrapped("annual leave policy", top_k=2)

        spans = mem_exporter.get_finished_spans()
        retrieval_span = next(s for s in spans if s.name == "retrieval")
        assert retrieval_span.attributes.get(ATTR_INPUT_VALUE) == "annual leave policy"

    def test_span_records_document_count(self, mem_exporter):
        tracer = get_tracer("retrieval")
        wrapped = traced_retrieval(self._fake_retriever, tracer)
        wrapped("policy", top_k=3)

        spans = mem_exporter.get_finished_spans()
        retrieval_span = next(s for s in spans if s.name == "retrieval")
        assert retrieval_span.attributes.get(ATTR_RETRIEVAL_DOCUMENTS) == 3

    def test_span_status_error_on_exception(self, mem_exporter):
        def bad_retriever(question: str, **kwargs):
            raise RuntimeError("network error")

        tracer = get_tracer("retrieval")
        wrapped = traced_retrieval(bad_retriever, tracer)

        with pytest.raises(RuntimeError):
            wrapped("query")

        spans = mem_exporter.get_finished_spans()
        retrieval_span = next(s for s in spans if s.name == "retrieval")
        from opentelemetry.trace import StatusCode

        assert retrieval_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# traced_generation
# ---------------------------------------------------------------------------


class TestTracedGeneration:
    def _fake_answer_fn(self, question: str, context: list[str], **kwargs) -> str:
        return f"Answer to '{question}' using {len(context)} context chunks."

    def test_returns_same_answer(self, mem_exporter):
        tracer = get_tracer("generation")
        wrapped = traced_generation(self._fake_answer_fn, tracer)
        answer = wrapped("What is the VPN policy?", ["VPN required for all."])
        assert "VPN policy" in answer

    def test_span_is_created(self, mem_exporter):
        tracer = get_tracer("generation")
        wrapped = traced_generation(self._fake_answer_fn, tracer)
        wrapped("question?", ["context chunk"])

        spans = mem_exporter.get_finished_spans()
        assert any(s.name == "generation" for s in spans)

    def test_span_records_model_name(self, mem_exporter):
        tracer = get_tracer("generation")
        wrapped = traced_generation(self._fake_answer_fn, tracer, model_name="gpt-4.1-mini")
        wrapped("question?", ["chunk"])

        spans = mem_exporter.get_finished_spans()
        gen_span = next(s for s in spans if s.name == "generation")
        assert gen_span.attributes.get(ATTR_LLM_MODEL_NAME) == "gpt-4.1-mini"

    def test_span_records_output_value(self, mem_exporter):
        tracer = get_tracer("generation")
        wrapped = traced_generation(self._fake_answer_fn, tracer)
        wrapped("What is leave?", ["14 days."])

        spans = mem_exporter.get_finished_spans()
        gen_span = next(s for s in spans if s.name == "generation")
        output = gen_span.attributes.get(ATTR_OUTPUT_VALUE, "")
        assert len(output) > 0

    def test_output_value_truncated_at_500_chars(self, mem_exporter):
        long_answer = "x" * 1000

        def long_answer_fn(question, context, **kwargs):
            return long_answer

        tracer = get_tracer("generation")
        wrapped = traced_generation(long_answer_fn, tracer)
        wrapped("q", ["c"])

        spans = mem_exporter.get_finished_spans()
        gen_span = next(s for s in spans if s.name == "generation")
        stored = gen_span.attributes.get(ATTR_OUTPUT_VALUE, "")
        assert len(stored) == 500

    def test_span_status_error_on_exception(self, mem_exporter):
        def broken_fn(question, context, **kwargs):
            raise ValueError("model error")

        tracer = get_tracer("generation")
        wrapped = traced_generation(broken_fn, tracer)

        with pytest.raises(ValueError):
            wrapped("q", ["c"])

        spans = mem_exporter.get_finished_spans()
        gen_span = next(s for s in spans if s.name == "generation")
        from opentelemetry.trace import StatusCode

        assert gen_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# build_traced_rag_pipeline
# ---------------------------------------------------------------------------


class TestBuildTracedRagPipeline:
    def _fake_retriever(self, question: str, **kwargs) -> list[RetrievalResult]:
        return [RetrievalResult(chunk_id="C-0", score=0.9, source="dense", text="VPN required.")]

    def _fake_answer_fn(self, question: str, context: list[str], **kwargs) -> str:
        return "VPN is required for all remote connections."

    def test_pipeline_returns_answer(self, mem_exporter):
        tracer = get_tracer("pipeline")
        pipeline = build_traced_rag_pipeline(
            retriever=self._fake_retriever,
            answer_fn=self._fake_answer_fn,
            tracer=tracer,
        )
        answer = pipeline("What is the VPN policy?")
        assert "VPN" in answer

    def test_pipeline_creates_parent_and_child_spans(self, mem_exporter):
        tracer = get_tracer("pipeline")
        pipeline = build_traced_rag_pipeline(
            retriever=self._fake_retriever,
            answer_fn=self._fake_answer_fn,
            tracer=tracer,
        )
        pipeline("What is the VPN policy?")

        spans = mem_exporter.get_finished_spans()
        span_names = {s.name for s in spans}
        assert "rag-pipeline" in span_names
        assert "retrieval" in span_names
        assert "generation" in span_names

    def test_child_spans_nested_inside_parent(self, mem_exporter):
        tracer = get_tracer("pipeline")
        pipeline = build_traced_rag_pipeline(
            retriever=self._fake_retriever,
            answer_fn=self._fake_answer_fn,
            tracer=tracer,
        )
        pipeline("What is the VPN policy?")

        spans = mem_exporter.get_finished_spans()
        parent_span = next(s for s in spans if s.name == "rag-pipeline")
        child_spans = [s for s in spans if s.name in ("retrieval", "generation")]

        parent_ctx = parent_span.context
        for child in child_spans:
            assert child.parent is not None
            assert child.parent.trace_id == parent_ctx.trace_id

    def test_model_name_attached_to_generation_span(self, mem_exporter):
        tracer = get_tracer("pipeline")
        pipeline = build_traced_rag_pipeline(
            retriever=self._fake_retriever,
            answer_fn=self._fake_answer_fn,
            tracer=tracer,
            model_name="gpt-4.1-mini",
        )
        pipeline("q")

        spans = mem_exporter.get_finished_spans()
        gen_span = next(s for s in spans if s.name == "generation")
        assert gen_span.attributes.get(ATTR_LLM_MODEL_NAME) == "gpt-4.1-mini"
