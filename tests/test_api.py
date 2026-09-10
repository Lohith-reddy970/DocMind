import numpy as np
import pytest
from fastapi.testclient import TestClient

from docmind import api
from docmind.config import Settings
from docmind.rag import RAGEngine
from docmind.vector_store import VectorStore


class _Embedder:
    dim = 8

    def embed(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32)

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_query(self, text):
        return np.ones((self.dim,), dtype=np.float32)

    def embed_one(self, text):
        return self.embed_query(text)


class _LLM:
    def generate(self, prompt, system=None):
        return "ok"

    def stream(self, prompt, system=None):
        yield "ok"


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(chunk_size=200, chunk_overlap=20, min_score=0.0, index_dir=tmp_path / "index")
    engine = RAGEngine(settings=settings, embedder=_Embedder(), llm=_LLM(), store=VectorStore(), autoload=False)
    monkeypatch.setattr(api, "engine", engine)
    return TestClient(api.app), engine


def test_upload_indexes_supported_file(client):
    c, engine = client
    files = {"files": ("notes.txt", b"hello world " * 50, "text/plain")}
    response = c.post("/ingest/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["files"] == 1
    assert data["ingested"] >= 1
    assert engine.store.size >= 1


def test_upload_skips_unsupported_files(client):
    c, _ = client
    files = {"files": ("image.png", b"\x89PNG\r\n", "image/png")}
    response = c.post("/ingest/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["files"] == 0
    assert "image.png" in data["skipped"]


def test_upload_multiple_files(client):
    c, _ = client
    files = [
        ("files", ("a.txt", b"alpha " * 40, "text/plain")),
        ("files", ("b.md", b"# beta\n\n" + b"content " * 40, "text/markdown")),
    ]
    response = c.post("/ingest/upload", files=files)
    assert response.status_code == 200
    assert response.json()["files"] == 2


def test_sources_endpoint_lists_uploaded_file(client):
    c, _ = client
    files = {"files": ("notes.txt", b"hello world " * 50, "text/plain")}
    c.post("/ingest/upload", files=files)
    sources = c.get("/sources").json()["sources"]
    assert len(sources) == 1
    assert sources[0]["name"] == "notes.txt"
    assert sources[0]["chunks"] >= 1


def test_health_endpoint(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok"}
