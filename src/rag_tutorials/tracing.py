"""Tracing utilities for the RAG and agent workflow tutorials.

Provides lightweight wrappers around OpenTelemetry spans so that retrieval,
answer-generation, and agent steps can be recorded and inspected.

For full Phoenix integration, call phoenix.otel.register() before using any
traced pipeline steps and use opentelemetry.trace.get_tracer(TRACER_NAME) to
obtain a tracer wired to the Phoenix collector.

For tests and offline demos, use build_in_memory_tracer() to capture spans
without a running server.
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

TRACER_NAME = "rag_tutorials"


def build_in_memory_tracer() -> tuple[trace.Tracer, InMemorySpanExporter]:
    """Create a TracerProvider backed by an in-memory span exporter.

    Returns a (tracer, exporter) pair. All spans started with the returned
    tracer are stored in the exporter buffer and can be read back via
    exporter.get_finished_spans().

    This is useful for unit tests and notebook demos where no Phoenix server
    is running. To send traces to Phoenix instead, call
    phoenix.otel.register() and obtain a tracer from the global provider.

    Returns:
        Tuple of (tracer, exporter).
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(TRACER_NAME)
    return tracer, exporter


def record_retrieval_span(
    tracer: trace.Tracer,
    query: str,
    results: list,
    top_k: int,
) -> None:
    """Record a single retrieval step as an OpenTelemetry span.

    Each call to a retriever is wrapped in a span named 'retrieval'. The span
    captures the query, how many results were requested, how many were
    returned, and the highest similarity score in the result set.

    Args:
        tracer: Active OpenTelemetry tracer.
        query: Query string passed to the retriever.
        results: List of RetrievalResult objects returned by the retriever.
        top_k: Maximum number of results requested.
    """
    with tracer.start_as_current_span("retrieval") as span:
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.result_count", len(results))
        if results:
            span.set_attribute("retrieval.top_score", float(results[0].score))


def record_generation_span(
    tracer: trace.Tracer,
    question: str,
    answer: str,
    model: str,
    context_chunk_count: int,
) -> None:
    """Record an answer-generation call as an OpenTelemetry span.

    Args:
        tracer: Active OpenTelemetry tracer.
        question: User question passed to the LLM.
        answer: Generated answer text returned by the LLM.
        model: Chat model name used for generation.
        context_chunk_count: Number of context chunks provided in the prompt.
    """
    with tracer.start_as_current_span("generation") as span:
        span.set_attribute("generation.question", question)
        span.set_attribute("generation.model", model)
        span.set_attribute("generation.context_chunk_count", context_chunk_count)
        span.set_attribute("generation.answer_word_count", len(answer.split()))


def record_agent_step_span(
    tracer: trace.Tracer,
    step_number: int,
    thought: str,
    action: str,
    action_input: str,
    observation: str,
) -> None:
    """Record a single ReAct agent step as an OpenTelemetry span.

    Each Thought-Action-Observation cycle is wrapped in a span named
    'agent_step'. The span captures what the agent was thinking, which tool
    it called, what input it passed to the tool, and what the tool returned.

    Args:
        tracer: Active OpenTelemetry tracer.
        step_number: Index of this step within the agent loop (starts at 1).
        thought: Agent's reasoning text from this step.
        action: Tool name called by the agent ('retrieve', 'finish', etc.).
        action_input: Input string passed to the tool.
        observation: Output string returned by the tool.
    """
    with tracer.start_as_current_span("agent_step") as span:
        span.set_attribute("agent.step_number", step_number)
        span.set_attribute("agent.thought", thought)
        span.set_attribute("agent.action", action)
        span.set_attribute("agent.action_input", action_input)
        span.set_attribute("agent.observation_length", len(observation))


def spans_to_dicts(spans) -> list[dict]:
    """Convert a sequence of finished ReadableSpan objects to plain dicts.

    Each dict contains the span name, status, wall-clock duration in
    milliseconds, and all recorded attributes. This is useful for displaying
    trace data in a notebook table without importing OpenTelemetry types.

    Args:
        spans: Iterable of finished ReadableSpan objects from an exporter.

    Returns:
        List of dicts with keys: 'name', 'status', 'duration_ms', 'attributes'.
    """
    rows = []
    for span in spans:
        duration_ns = (span.end_time or 0) - (span.start_time or 0)
        rows.append(
            {
                "name": span.name,
                "status": span.status.status_code.name,
                "duration_ms": round(duration_ns / 1_000_000, 2),
                "attributes": dict(span.attributes or {}),
            }
        )
    return rows
