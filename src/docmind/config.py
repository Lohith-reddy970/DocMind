"""Runtime configuration, loaded from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_PREFIX = "DOCMIND_"


def _env(name: str, default: str) -> str:
    return os.environ.get(_ENV_PREFIX + name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool | None) -> bool | None:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_opt(name: str, default: str | None) -> str | None:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None:
        return default
    return raw


@dataclass
class Settings:
    """Configuration for embeddings, retrieval, generation and storage."""

    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:8b"))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    min_score: float = field(default_factory=lambda: _env_float("MIN_SCORE", 0.5))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.2))
    top_p: float = field(default_factory=lambda: _env_float("TOP_P", 0.9))
    repeat_penalty: float = field(default_factory=lambda: _env_float("REPEAT_PENALTY", 1.2))
    num_predict: int = field(default_factory=lambda: _env_int("NUM_PREDICT", 512))
    think: bool | None = field(default_factory=lambda: _env_bool("THINK", False))
    query_prefix: str | None = field(default_factory=lambda: _env_opt("QUERY_PREFIX", None))
    doc_prefix: str | None = field(default_factory=lambda: _env_opt("DOC_PREFIX", None))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", "index")))
    request_timeout: float = field(default_factory=lambda: float(_env("REQUEST_TIMEOUT", "120")))

    def llm_options(self) -> dict:
        """Sampling/decoding options passed to Ollama's chat endpoint."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": 256,
            "num_predict": self.num_predict,
            "num_ctx": 4096,
        }

    def index_files(self) -> tuple[Path, Path]:
        """Return the (vectors, chunks) index file paths."""
        return self.index_dir / "vectors.npy", self.index_dir / "chunks.jsonl"
