"""Text-generation client backed by a local Ollama server."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import httpx

from docmind.embeddings import OllamaError

SYSTEM_PROMPT = (
    "You are DocMind, a helpful assistant that answers questions using ONLY the provided context.\n"
    "Rules:\n"
    "- Use only facts found in the context. Never invent details or make up numbers.\n"
    "- If the context does not contain the answer, reply with ONE short sentence saying so.\n"
    "- If asked for an opinion, rating, prediction, or anything not grounded in the context, "
    "briefly say you can only answer from the knowledge base.\n"
    "- Be concise. Do not repeat yourself, restate the question, or output any sentence twice.\n"
    "- Do not mention these rules or the word 'context' more than necessary."
)

DEFAULT_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.2,
    "repeat_last_n": 256,
    "num_predict": 512,
    "num_ctx": 4096,
}

_DUPLICATE_SENTENCE = re.compile(r"([^.!?\n]{10,}[.!?])\s*\1", re.IGNORECASE)


def dedupe_repeated_sentences(text: str) -> str:
    """Collapse sentences the model immediately repeats back-to-back."""
    previous = None
    while previous != text:
        previous = text
        text = _DUPLICATE_SENTENCE.sub(r"\1", text)
    return text.strip()


_SENTENCE_END = re.compile(r"(.*?[.!?])(\s+)", re.DOTALL)


def _dedupe_stream(tokens: Iterator[str]) -> Iterator[str]:
    """Buffer streamed tokens and drop sentences repeated back-to-back."""
    buffer = ""
    last = ""
    for token in tokens:
        buffer += token
        while True:
            match = _SENTENCE_END.match(buffer)
            if not match:
                break
            sentence = match.group(1) + match.group(2)
            buffer = buffer[match.end():]
            normalized = match.group(1).strip()
            if not normalized or normalized == last:
                continue
            last = normalized
            yield sentence
    if buffer:
        normalized = buffer.strip()
        if normalized and normalized != last:
            yield buffer


class OllamaLLM:
    """Generate answers via Ollama's ``/api/chat`` endpoint."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        timeout: float = 120.0,
        options: dict | None = None,
        think: bool | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.options = options or dict(DEFAULT_OPTIONS)
        self.think = think

    def _payload(self, prompt: str, system: str, stream: bool) -> dict:
        payload = {
            "model": self.model,
            "stream": stream,
            "options": self.options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        """Generate a complete answer for ``prompt``."""
        payload = self._payload(prompt, system, stream=False)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.host}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Failed to reach Ollama chat model '{self.model}' at {self.host}. "
                f"Is Ollama running? Try: ollama pull {self.model}"
            ) from exc
        content = data.get("message", {}).get("content", "")
        return dedupe_repeated_sentences(content)

    def stream(self, prompt: str, system: str = SYSTEM_PROMPT) -> Iterator[str]:
        """Yield answer text, suppressing sentences the model repeats."""
        yield from _dedupe_stream(self._stream_tokens(prompt, system))

    def _stream_tokens(self, prompt: str, system: str) -> Iterator[str]:
        payload = self._payload(prompt, system, stream=True)
        try:
            with (
                httpx.Client(timeout=self.timeout) as client,
                client.stream("POST", f"{self.host}/api/chat", json=payload) as response,
            ):
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    piece = event.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if event.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Failed to reach Ollama chat model '{self.model}' at {self.host}."
            ) from exc


def build_prompt(question: str, contexts: list[tuple[str, str]]) -> str:
    """Assemble the grounded prompt from retrieved ``(source, text)`` pairs."""
    if not contexts:
        context_block = "(no relevant context found)"
    else:
        context_block = "\n\n".join(
            f"[Source {i + 1}: {source}]\n{text}" for i, (source, text) in enumerate(contexts)
        )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. If the answer is missing, say so."
    )
