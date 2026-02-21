from __future__ import annotations

from .schema import Chunk, Document


def fixed_chunk_documents(documents: list[Document], chunk_size: int = 260) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        text = document.text
        start = 0
        part = 0
        while start < len(text):
            segment = text[start : start + chunk_size]
            chunk_id = f"{document.doc_id}-FIX-{part:02d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    section=document.section,
                    text=segment,
                )
            )
            start += chunk_size
            part += 1
    return chunks


def semantic_chunk_documents(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        sentence_like_parts = [piece.strip() for piece in document.text.split(". ") if piece.strip()]
        if len(sentence_like_parts) <= 2:
            merged_groups = [". ".join(sentence_like_parts)]
        else:
            merged_groups = [
                ". ".join(sentence_like_parts[:2]),
                ". ".join(sentence_like_parts[2:]),
            ]

        for idx, group in enumerate(merged_groups):
            chunk_id = f"{document.doc_id}-SEM-{idx:02d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    section=document.section,
                    text=group,
                )
            )

    return chunks
