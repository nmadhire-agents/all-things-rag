"""Tests for settings.py — load_settings defaults and env overrides."""
from __future__ import annotations

import pytest

from rag_tutorials.settings import OpenAISettings, Paths, load_settings


class TestOpenAISettings:
    def test_default_embedding_model(self):
        s = OpenAISettings()
        assert s.embedding_model == "text-embedding-3-small"

    def test_default_chat_model(self):
        s = OpenAISettings()
        assert s.chat_model == "gpt-4.1-mini"

    def test_custom_values(self):
        s = OpenAISettings(embedding_model="text-embedding-3-large", chat_model="gpt-4o")
        assert s.embedding_model == "text-embedding-3-large"
        assert s.chat_model == "gpt-4o"


class TestPaths:
    def test_default_data_dir(self):
        p = Paths()
        assert p.data_dir == "data"

    def test_default_artifacts_dir(self):
        p = Paths()
        assert p.artifacts_dir == "artifacts"


class TestLoadSettings:
    def test_returns_tuple_of_settings_and_paths(self, monkeypatch):
        monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
        settings, paths = load_settings()
        assert isinstance(settings, OpenAISettings)
        assert isinstance(paths, Paths)

    def test_defaults_when_env_vars_absent(self, monkeypatch):
        monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
        settings, _ = load_settings()
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.chat_model == "gpt-4.1-mini"

    def test_env_vars_override_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o")
        settings, _ = load_settings()
        assert settings.embedding_model == "text-embedding-3-large"
        assert settings.chat_model == "gpt-4o"
