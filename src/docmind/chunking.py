"""Split documents into overlapping chunks suitable for embedding."""

from __future__ import annotations

from dataclasses import dataclass

from docmind.documents import Document

_PARAGRAPH_SEP = "\n\n"
_MIN_CHUNK_CHARS = 40


@dataclass
class Chunk:
    """A retrievable piece of text with provenance metadata."""

    text: str
    source: str
    chunk_index: int
    metadata: dict

    @property
    def id(self) -> str:
        return f"{self.source}::{self.chunk_index}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(
            text=data["text"],
            source=data["source"],
            chunk_index=data["chunk_index"],
            metadata=data.get("metadata", {}),
        )


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Greedily pack paragraphs into chunks of at most ``chunk_size`` characters.

    Paragraphs longer than ``chunk_size`` are hard-wrapped. Consecutive chunks
    share ``overlap`` characters so context is not lost across boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    units: list[str] = []
    for paragraph in text.split(_PARAGRAPH_SEP):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        step = chunk_size - overlap
        for start in range(0, len(paragraph), step):
            piece = paragraph[start : start + chunk_size].strip()
            if piece:
                units.append(piece)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n\n{unit}".strip() if tail else unit
    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.strip()) >= _MIN_CHUNK_CHARS] or ([text.strip()] if text.strip() else [])


def chunk_document(document: Document, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Chunk a single :class:`Document`."""
    pieces = split_text(document.text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            text=piece,
            source=document.source,
            chunk_index=i,
            metadata={**document.metadata, "chunk_of": document.source},
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_documents(documents: list[Document], chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Chunk a list of documents into a flat list of chunks."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks
