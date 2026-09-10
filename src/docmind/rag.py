"""The retrieval-augmented generation engine that ties everything together."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from docmind.chunking import Chunk, chunk_documents
from docmind.config import Settings
from docmind.documents import Document, load_path
from docmind.embeddings import OllamaEmbedder
from docmind.llm import OllamaLLM, build_prompt
from docmind.netguard import EGRESS, is_local_host
from docmind.vector_store import VectorStore


@dataclass
class Source:
    """A retrieved chunk used to ground an answer."""

    source: str
    score: float
    text: str

    def to_dict(self) -> dict:
        return {"source": self.source, "score": round(self.score, 4), "text": self.text}


@dataclass
class Answer:
    """A generated answer together with the sources that grounded it."""

    question: str
    text: str
    sources: list[Source] = field(default_factory=list)


class RAGEngine:
    """High-level API for ingesting documents and answering questions."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: OllamaEmbedder | None = None,
        llm: OllamaLLM | None = None,
        store: VectorStore | None = None,
        autoload: bool = True,
    ):
        self.settings = settings or Settings()
        self.embedder = embedder or OllamaEmbedder(
            host=self.settings.ollama_host,
            model=self.settings.embed_model,
            timeout=self.settings.request_timeout,
            query_prefix=self.settings.query_prefix,
            doc_prefix=self.settings.doc_prefix,
            local_only=self.settings.local_only,
        )
        self.llm = llm or OllamaLLM(
            host=self.settings.ollama_host,
            model=self.settings.llm_model,
            timeout=self.settings.request_timeout,
            options=self.settings.llm_options(),
            think=self.settings.think,
            local_only=self.settings.local_only,
        )
        self.store = store or VectorStore()
        if autoload and store is None:
            vector_path, chunk_path = self.settings.index_files()
            self.store = VectorStore.load(vector_path, chunk_path)

    def ingest_documents(self, documents: list[Document]) -> int:
        """Chunk, embed and index a list of documents. Returns the chunk count."""
        chunks = chunk_documents(
            documents, chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )
        return self.ingest_chunks(chunks)

    def ingest_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and index pre-made chunks. Returns the chunk count."""
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        self.store.add(vectors, chunks)
        return len(chunks)

    def ingest_path(self, path: str | Path) -> int:
        """Load and index a file or directory."""
        return self.ingest_documents(load_path(path))

    def ingest_text(self, text: str, source: str = "inline") -> int:
        """Index a raw string under a given source label."""
        return self.ingest_documents([Document(text=text, source=source, metadata={"type": "text"})])

    def retrieve(self, question: str, top_k: int | None = None) -> list[Source]:
        """Return the most relevant chunks, dropping weak matches below ``min_score``."""
        top_k = top_k or self.settings.top_k
        if self.store.size == 0:
            return []
        query_vector = self.embedder.embed_query(question)
        return [
            Source(source=chunk.source, score=score, text=chunk.text)
            for chunk, score in self.store.search(query_vector, top_k=top_k)
            if score >= self.settings.min_score
        ]

    def query(self, question: str, top_k: int | None = None) -> Answer:
        """Retrieve context and generate a grounded answer."""
        sources = self.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, [(s.source, s.text) for s in sources])
        text = self.llm.generate(prompt)
        return Answer(question=question, text=text, sources=sources)

    def stream_query(self, question: str, top_k: int | None = None) -> Iterator[str]:
        """Yield answer tokens, then a final JSON line with the sources."""
        sources = self.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, [(s.source, s.text) for s in sources])
        yield from self.llm.stream(prompt)
        import json

        yield "\n__SOURCES__" + json.dumps([s.to_dict() for s in sources], ensure_ascii=False)

    def save(self) -> None:
        """Persist the index to disk."""
        vector_path, chunk_path = self.settings.index_files()
        self.store.save(vector_path, chunk_path)

    def reset(self) -> None:
        """Clear the in-memory index and delete persisted files."""
        self.store.clear()
        vector_path, chunk_path = self.settings.index_files()
        for path in (vector_path, chunk_path):
            Path(path).unlink(missing_ok=True)

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            "chunks": self.store.size,
            "dim": self.store.dim,
            "sources": len({chunk.source for chunk in self.store.chunks()}),
            "embed_model": self.settings.embed_model,
            "llm_model": self.settings.llm_model,
        }

    def network_report(self) -> dict:
        """Describe every outbound destination DocMind can or did contact."""
        host = urlparse(self.settings.ollama_host).hostname
        return {
            "local_only": self.settings.local_only,
            "ollama_host": self.settings.ollama_host,
            "host_is_local": is_local_host(host),
            "blocked_requests": EGRESS.blocked,
            "endpoints": EGRESS.snapshot(),
            "message": (
                "Only the configured Ollama server is contacted. "
                "No telemetry, analytics, or cloud APIs."
            ),
        }
