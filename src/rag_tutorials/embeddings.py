from __future__ import annotations

import numpy as np
from openai import OpenAI


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    vectors = [row.embedding for row in response.data]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(query_norm * matrix_norm, 1e-12)
    return (matrix @ query_vector) / denominator
