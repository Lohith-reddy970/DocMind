"""DocMind - a local, privacy-first RAG assistant over your own documents."""

from docmind.config import Settings
from docmind.rag import Answer, RAGEngine, Source

__all__ = ["Answer", "RAGEngine", "Settings", "Source"]
__version__ = "0.1.0"
