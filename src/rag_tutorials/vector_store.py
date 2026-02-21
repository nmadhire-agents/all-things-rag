from __future__ import annotations

from pathlib import Path

import chromadb

from .schema import Chunk, RetrievalResult


def build_chroma_collection(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    collection_name: str,
    persist_dir: str = "artifacts/chroma",
):
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    existing = {collection.name for collection in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(name=collection_name)
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk.text for chunk in chunks],
        metadatas=[{"doc_id": chunk.doc_id, "section": chunk.section} for chunk in chunks],
    )
    return collection


def dense_search(
    collection,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[RetrievalResult]:
    response = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    ids = response["ids"][0]
    docs = response["documents"][0]
    scores = response["distances"][0]

    return [
        RetrievalResult(
            chunk_id=chunk_id,
            score=float(1.0 - distance),
            source="dense",
            text=text,
        )
        for chunk_id, text, distance in zip(ids, docs, scores, strict=True)
    ]
