from __future__ import annotations

import numpy as np

from .chunking import fixed_chunk_documents, semantic_chunk_documents
from .embeddings import cosine_similarity, embed_texts
from .io_utils import load_handbook_documents
from .retrieval import bm25_search, build_bm25, reciprocal_rank_fusion
from .schema import Chunk, RetrievalResult
from .vector_store import build_chroma_collection, dense_search


def prepare_chunks(mode: str, handbook_path: str = "data/handbook_manual.txt") -> list[Chunk]:
    documents = load_handbook_documents(handbook_path)
    if mode == "semantic":
        return semantic_chunk_documents(documents)
    return fixed_chunk_documents(documents)


def build_dense_retriever(chunks: list[Chunk], collection_name: str, embedding_model: str):
    vectors = embed_texts([chunk.text for chunk in chunks], model=embedding_model)
    collection = build_chroma_collection(
        chunks=chunks,
        embeddings=vectors.tolist(),
        collection_name=collection_name,
    )

    def retrieve(question: str, top_k: int = 5) -> list[RetrievalResult]:
        query_vector = embed_texts([question], model=embedding_model)[0]
        return dense_search(collection=collection, query_embedding=query_vector.tolist(), top_k=top_k)

    return retrieve, vectors


def build_hybrid_retriever(chunks: list[Chunk], dense_retriever):
    bm25_index, corpus, chunk_ids = build_bm25(chunks)

    def retrieve(question: str, top_k: int = 5) -> list[RetrievalResult]:
        dense_results = dense_retriever(question, top_k=top_k)
        keyword_results = bm25_search(bm25_index, question, corpus, chunk_ids, top_k=top_k)
        return reciprocal_rank_fusion(dense_results, keyword_results, k=60)[:top_k]

    return retrieve


def top_scores_preview(question: str, chunks: list[Chunk], vectors: np.ndarray, embedding_model: str, top_k: int = 5):
    query_vector = embed_texts([question], model=embedding_model)[0]
    scores = cosine_similarity(query_vector, vectors)
    indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank": rank + 1,
            "chunk_id": chunks[idx].chunk_id,
            "score": float(scores[idx]),
            "text": chunks[idx].text,
        }
        for rank, idx in enumerate(indices)
    ]
