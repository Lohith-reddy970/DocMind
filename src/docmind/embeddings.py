"""Embedding model client backed by a local Ollama server."""

from __future__ import annotations

import httpx
import numpy as np


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


def default_prefixes(model: str) -> tuple[str, str]:
    """Return ``(query_prefix, document_prefix)`` appropriate for ``model``.

    Nomic's embedding models are trained with task prefixes and retrieve better
    with them; most other models use no prefix.
    """
    if "nomic" in model.lower():
        return "search_query: ", "search_document: "
    return "", ""


class OllamaEmbedder:
    """Generate dense embeddings via Ollama's ``/api/embed`` endpoint."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 120.0,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        default_query, default_doc = default_prefixes(model)
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.doc_prefix = default_doc if doc_prefix is None else doc_prefix

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages for indexing, applying the document task prefix."""
        return self.embed([self.doc_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query, applying the query task prefix."""
        return self.embed([self.query_prefix + text])[0]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning an ``(n, dim)`` float32 array."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        payload = {"model": self.model, "input": texts}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.host}/api/embed", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Failed to reach Ollama embedding model '{self.model}' at {self.host}. "
                f"Is Ollama running? Try: ollama pull {self.model}"
            ) from exc

        vectors = data.get("embeddings")
        if not vectors:
            raise OllamaError(f"Ollama returned no embeddings for model '{self.model}'.")
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string, returning a 1-D vector."""
        return self.embed([text])[0]
