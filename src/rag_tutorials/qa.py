from __future__ import annotations

from openai import OpenAI


def build_context(chunks: list[str]) -> str:
    return "\n\n".join([f"Chunk {idx + 1}: {chunk}" for idx, chunk in enumerate(chunks)])


def answer_with_context(question: str, context_chunks: list[str], model: str = "gpt-4.1-mini") -> str:
    context_block = build_context(context_chunks)
    prompt = (
        "You are a policy assistant. Answer only from the provided context. "
        "If the answer is not present, say you do not have enough context.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Provide a concise answer and include a short citation like [Chunk 1]."
    )

    client = OpenAI()
    response = client.responses.create(model=model, input=prompt)
    return response.output_text
