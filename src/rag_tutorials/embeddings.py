from __future__ import annotations

import numpy as np
from openai import OpenAI


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Generate embedding vectors for input texts using OpenAI embeddings API.

    Args:
        texts: Input strings to embed.
        model: Embedding model name.

    Returns:
        A `float32` NumPy matrix shaped `(len(texts), embedding_dim)`.
    """
    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    vectors = [row.embedding for row in response.data]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and many vectors.

    Args:
        query_vector: Query embedding vector.
        matrix: Candidate embedding matrix where each row is one vector.

    Returns:
        A 1D array of cosine similarity scores aligned to matrix rows.
    """
    query_norm = np.linalg.norm(query_vector)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(query_norm * matrix_norm, 1e-12)
    return (matrix @ query_vector) / denominator
