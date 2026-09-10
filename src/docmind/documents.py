"""Load raw text from plain-text, Markdown and PDF files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Document:
    """A single source document, before chunking."""

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def _normalize_path(path: str | Path) -> Path:
    """Clean a user-supplied path: trim whitespace and stray surrounding quotes."""
    if isinstance(path, str):
        cleaned = path.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()
        return Path(cleaned)
    return Path(path)


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError("PDF support requires 'pypdf'. Install it with: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def load_file(path: str | Path) -> Document:
    """Load a single supported file into a :class:`Document`."""
    path = _normalize_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    text = _load_pdf(path) if suffix == ".pdf" else _load_text_file(path)
    return Document(text=text, source=str(path), metadata={"type": suffix.lstrip(".")})


def load_path(path: str | Path) -> list[Document]:
    """Load a file, or every supported file recursively under a directory."""
    path = _normalize_path(path)
    if path.is_file():
        return [load_file(path)]
    if path.is_dir():
        docs: list[Document] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    docs.append(load_file(child))
                except (ValueError, RuntimeError):
                    continue
        return docs
    raise FileNotFoundError(f"No such file or directory: {path}")
