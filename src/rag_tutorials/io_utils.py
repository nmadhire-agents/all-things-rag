from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .data_generation import parse_handbook_to_documents
from .schema import Chunk, Document, QueryExample


def _load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            records.append(json.loads(line))
    return records


def load_documents(path: str | Path = "data/documents.jsonl") -> list[Document]:
    return [Document(**record) for record in _load_jsonl(path)]


def load_handbook_documents(path: str | Path = "data/handbook_manual.txt") -> list[Document]:
    handbook_text = Path(path).read_text(encoding="utf-8")
    return parse_handbook_to_documents(handbook_text)


def load_queries(path: str | Path = "data/queries.jsonl") -> list[QueryExample]:
    return [QueryExample(**record) for record in _load_jsonl(path)]


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file_handle:
        for chunk in chunks:
            file_handle.write(json.dumps(asdict(chunk)) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    return [Chunk(**record) for record in _load_jsonl(path)]
