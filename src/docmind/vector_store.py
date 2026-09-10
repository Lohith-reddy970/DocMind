"""A tiny, dependency-light vector store using cosine similarity over NumPy arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from docmind.chunking import Chunk


class VectorStore:
    """Store chunk embeddings and retrieve the nearest neighbours for a query."""

    def __init__(self, dim: int | None = None):
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._dim = dim

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def dim(self) -> int | None:
        return self._dim

    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Append embeddings and their chunks to the store."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if len(vectors) != len(chunks):
            raise ValueError("Number of vectors must match number of chunks")

        vectors = self._normalize(vectors)
        if self._vectors is None:
            self._vectors = vectors
            self._dim = vectors.shape[1]
        else:
            if vectors.shape[1] != self._dim:
                raise ValueError(f"Embedding dim mismatch: expected {self._dim}, got {vectors.shape[1]}")
            self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` most similar chunks with cosine scores."""
        if self._vectors is None or not self._chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        query = self._normalize(query)
        scores = (self._vectors @ query.T).ravel()
        top_k = min(top_k, len(scores))
        best = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in best]

    def clear(self) -> None:
        self._vectors = None
        self._chunks = []
        self._dim = None

    def save(self, vector_path: str | Path, chunk_path: str | Path) -> None:
        """Persist the index to an ``.npy`` file plus a JSONL chunk file."""
        vector_path = Path(vector_path)
        chunk_path = Path(chunk_path)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0), dtype=np.float32)
        np.save(vector_path, vectors)
        with chunk_path.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, vector_path: str | Path, chunk_path: str | Path) -> VectorStore:
        """Load a previously saved index. Returns an empty store if files are missing."""
        store = cls()
        vector_path = Path(vector_path)
        chunk_path = Path(chunk_path)
        if not vector_path.exists() or not chunk_path.exists():
            return store

        vectors = np.load(vector_path)
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if vectors.size and len(chunks):
            store._vectors = vectors.astype(np.float32)
            store._chunks = chunks
            store._dim = vectors.shape[1]
        return store
