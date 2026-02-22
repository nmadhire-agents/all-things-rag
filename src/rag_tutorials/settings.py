from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True)
class OpenAISettings:
    """Runtime model configuration for embedding and generation calls."""

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"


@dataclass(slots=True)
class Paths:
    """Common project paths used by tutorial code."""

    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


def load_settings() -> tuple[OpenAISettings, Paths]:
    """Load environment-backed settings and return typed config objects.

    Returns:
        Tuple containing OpenAI model settings and common path settings.
    """
    load_dotenv()
    return (
        OpenAISettings(
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        ),
        Paths(),
    )
