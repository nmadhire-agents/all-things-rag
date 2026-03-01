"""OpenTelemetry tracing helpers for the RAG pipeline tutorials.

This module provides lightweight utilities for adding observability to the RAG
pipeline using the OpenTelemetry (OTel) Python SDK.

Key concepts:
- Span         : a single named, timed unit of work (one retrieval call, one LLM call)
- Trace        : a tree of spans that together describe one end-to-end request
- TracerProvider: the entry point that configures how spans are created and exported
- Tracer       : created from the provider; used to start new spans
- Exporter     : receives completed spans and forwards them to an observability backend

Usage with Arize Phoenix (local backend):

    from rag_tutorials.tracing import configure_tracing, get_tracer
    from rag_tutorials.tracing import traced_retrieval, traced_generation

    configure_tracing(
        endpoint="http://localhost:6006/v1/traces",
        service_name="rag-pipeline",
    )
    tracer = get_tracer("rag-pipeline.retrieval")
    wrapped = traced_retrieval(retriever, tracer)
    results = wrapped("remote work policy", top_k=5)

Usage without a backend (development / testing):

    from rag_tutorials.tracing import configure_tracing, get_tracer

    configure_tracing()   # uses ConsoleSpanExporter by default
    tracer = get_tracer("dev")
    with tracer.start_as_current_span("my-operation") as span:
        span.set_attribute("input", "hello")
"""
from __future__ import annotations

from typing import Callable

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

from .schema import RetrievalResult

# ---------------------------------------------------------------------------
# OpenInference semantic-convention attribute names
# (subset used by the RAG tutorials; full spec at openinference.ai)
# ---------------------------------------------------------------------------

ATTR_INPUT_VALUE = "input.value"
ATTR_OUTPUT_VALUE = "output.value"
ATTR_LLM_MODEL_NAME = "llm.model_name"
ATTR_LLM_PROMPT_TEMPLATE = "llm.prompt_template.template"
ATTR_RETRIEVAL_DOCUMENTS = "retrieval.documents"
ATTR_EMBEDDING_MODEL_NAME = "embedding.model_name"
ATTR_EMBEDDING_EMBEDDINGS = "embedding.embeddings"

# ---------------------------------------------------------------------------
# Provider lifecycle helpers
# ---------------------------------------------------------------------------

_provider: TracerProvider | None = None


def configure_tracing(
    endpoint: str | None = None,
    service_name: str = "rag-pipeline",
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """Create and register a global TracerProvider.

    Call this once at the start of your notebook or application before
    creating any tracers.

    Args:
        endpoint: OTLP HTTP endpoint URL to send traces to.  Pass the URL of
            your Arize Phoenix instance (e.g. ``http://localhost:6006/v1/traces``).
            When *None* and no custom *exporter* is given, spans are printed to
            stdout via :class:`~opentelemetry.sdk.trace.export.ConsoleSpanExporter`.
        service_name: A human-readable label that identifies this application
            in the observability backend.  Shows up as the service column in
            Phoenix.
        exporter: An already-constructed :class:`~opentelemetry.sdk.trace.export.SpanExporter`
            instance.  Use this for testing (e.g., pass an
            ``InMemorySpanExporter``) or for custom backends.  When provided,
            *endpoint* is ignored.

    Returns:
        The configured :class:`~opentelemetry.sdk.trace.TracerProvider`.  The
        same provider is also set as the global OTel provider so that
        :func:`get_tracer` can retrieve it without requiring you to pass it
        around explicitly.
    """
    global _provider

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if exporter is not None:
        chosen_exporter: SpanExporter = exporter
    elif endpoint is not None:
        # Import lazily so the package is optional for users who only use the
        # console exporter during development.
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "opentelemetry-exporter-otlp-proto-http is required to export "
                "traces to an OTLP endpoint.  Install it with:\n"
                "  uv add opentelemetry-exporter-otlp-proto-http"
            ) from exc
        chosen_exporter = OTLPSpanExporter(endpoint=endpoint)
    else:
        chosen_exporter = ConsoleSpanExporter()

    # SimpleSpanProcessor exports each span immediately (synchronous).
    # This is the correct choice for development, testing, and notebooks.
    # BatchSpanProcessor is more efficient for high-throughput production
    # workloads but requires an explicit flush before reading collected spans.
    provider.add_span_processor(SimpleSpanProcessor(chosen_exporter))
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def get_tracer(name: str) -> trace.Tracer:
    """Return an OTel :class:`~opentelemetry.trace.Tracer` for *name*.

    The returned tracer uses the provider most recently configured by
    :func:`configure_tracing`.  If :func:`configure_tracing` has not been
    called, the no-op global provider is used (spans are silently discarded).

    Args:
        name: Instrumentation scope name, typically the module or component
            being traced (e.g. ``"retrieval"``, ``"generation"``).

    Returns:
        A :class:`~opentelemetry.trace.Tracer` instance.
    """
    if _provider is not None:
        return _provider.get_tracer(name)
    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# Span-wrapping helpers for common RAG operations
# ---------------------------------------------------------------------------


def traced_retrieval(
    retriever: Callable[..., list[RetrievalResult]],
    tracer: trace.Tracer,
) -> Callable[..., list[RetrievalResult]]:
    """Wrap a retriever callable so every call is recorded as an OTel span.

    The resulting callable has the same signature as *retriever* and returns
    the same results.  It additionally creates a span named
    ``"retrieval"`` that records:

    - ``input.value``: the query string passed to the retriever
    - ``retrieval.documents``: the number of results returned
    - span status: OK on success, ERROR on exception

    Args:
        retriever: Any callable with signature
            ``(question: str, top_k: int = ...) -> list[RetrievalResult]``.
        tracer: OTel tracer to use for span creation.

    Returns:
        A wrapped callable with identical behaviour plus tracing.

    Example::

        tracer = get_tracer("retrieval")
        wrapped = traced_retrieval(dense_retriever, tracer)
        results = wrapped("remote work VPN", top_k=5)
    """

    def _wrapped(question: str, **kwargs) -> list[RetrievalResult]:
        with tracer.start_as_current_span("retrieval") as span:
            span.set_attribute(ATTR_INPUT_VALUE, question)
            try:
                results = retriever(question, **kwargs)
                span.set_attribute(ATTR_RETRIEVAL_DOCUMENTS, len(results))
                span.set_status(trace.StatusCode.OK)
                return results
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    return _wrapped


def traced_generation(
    answer_fn: Callable[..., str],
    tracer: trace.Tracer,
    model_name: str = "",
) -> Callable[..., str]:
    """Wrap an answer-generation callable so every call is recorded as an OTel span.

    The resulting callable has the same signature as *answer_fn* and returns
    the same result.  It additionally creates a span named ``"generation"``
    that records:

    - ``input.value``: the question passed to the function
    - ``llm.model_name``: the model name (when provided)
    - ``output.value``: the generated answer (first 500 characters)
    - span status: OK on success, ERROR on exception

    Args:
        answer_fn: Any callable with signature
            ``(question: str, context: list[str]) -> str``.
        tracer: OTel tracer to use for span creation.
        model_name: Optional model identifier to attach as span metadata.

    Returns:
        A wrapped callable with identical behaviour plus tracing.

    Example::

        tracer = get_tracer("generation")
        wrapped = traced_generation(answer_with_context, tracer, model_name="gpt-4.1-mini")
        answer = wrapped("What is the VPN policy?", ["VPN is required..."])
    """

    def _wrapped(question: str, context: list[str], **kwargs) -> str:
        with tracer.start_as_current_span("generation") as span:
            span.set_attribute(ATTR_INPUT_VALUE, question)
            if model_name:
                span.set_attribute(ATTR_LLM_MODEL_NAME, model_name)
            try:
                answer = answer_fn(question, context, **kwargs)
                span.set_attribute(ATTR_OUTPUT_VALUE, answer[:500])
                span.set_status(trace.StatusCode.OK)
                return answer
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    return _wrapped


def build_traced_rag_pipeline(
    retriever: Callable[..., list[RetrievalResult]],
    answer_fn: Callable[..., str],
    tracer: trace.Tracer,
    model_name: str = "",
) -> Callable[[str], str]:
    """Combine traced retrieval and generation into a single traced RAG pipeline.

    The pipeline creates a parent span named ``"rag-pipeline"`` that contains
    two child spans: one for retrieval and one for generation.  This lets you
    see the complete request as a single trace tree in the observability UI.

    Args:
        retriever: Dense or hybrid retriever callable.
        answer_fn: Answer generation callable.
        tracer: OTel tracer shared by both child operations.
        model_name: LLM model name to attach as generation span metadata.

    Returns:
        A callable ``(question: str) -> str`` that runs the full RAG pipeline
        under a single parent trace.

    Example::

        pipeline = build_traced_rag_pipeline(
            retriever=dense_retriever,
            answer_fn=answer_with_context,
            tracer=get_tracer("rag-pipeline"),
            model_name="gpt-4.1-mini",
        )
        answer = pipeline("What is the remote work policy?")
    """
    w_retriever = traced_retrieval(retriever, tracer)
    w_answer = traced_generation(answer_fn, tracer, model_name=model_name)

    def _pipeline(question: str) -> str:
        with tracer.start_as_current_span("rag-pipeline") as span:
            span.set_attribute(ATTR_INPUT_VALUE, question)
            results = w_retriever(question)
            context = [r.text for r in results]
            answer = w_answer(question, context)
            span.set_attribute(ATTR_OUTPUT_VALUE, answer[:500])
            return answer

    return _pipeline
