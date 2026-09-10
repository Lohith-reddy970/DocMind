"""Shared test fixtures with fake embedder and LLM (no network required)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from docmind.config import Settings


class FakeEmbedder:
    """Deterministic bag-of-words-ish embedder for tests."""

    dim = 64

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed_one(t) for t in texts])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_one(text)

    def embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode()).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector


class FakeLLM:
    def __init__(self, reply: str = "test answer"):
        self.reply = reply
        self.last_prompt = ""

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.last_prompt = prompt
        return self.reply

    def stream(self, prompt: str, system: str | None = None):
        self.last_prompt = prompt
        for token in self.reply.split():
            yield token + " "


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(chunk_size=200, chunk_overlap=40, top_k=2, min_score=0.0, index_dir=tmp_path / "index")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
