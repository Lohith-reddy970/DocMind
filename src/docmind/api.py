"""FastAPI application exposing the RAG engine over HTTP."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from docmind.config import Settings
from docmind.rag import RAGEngine

STATIC_DIR = Path(__file__).parent / "static"
SENTINEL = "\n__SOURCES__"

app = FastAPI(
    title="DocMind",
    description="A local, privacy-first RAG assistant over your own documents.",
    version="0.1.0",
)
engine = RAGEngine(Settings())


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "inline"


class PathRequest(BaseModel):
    path: str
    reset: bool = False


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    return engine.stats()


@app.get("/sources")
def sources() -> dict:
    counts: dict[str, int] = {}
    for chunk in engine.store.chunks():
        counts[chunk.source] = counts.get(chunk.source, 0) + 1
    return {"sources": [{"name": name, "chunks": count} for name, count in sorted(counts.items())]}


@app.post("/reset")
def reset() -> dict:
    engine.reset()
    return {"status": "reset"}


@app.post("/ingest/text")
def ingest_text(request: TextRequest) -> dict:
    count = engine.ingest_text(request.text, source=request.source)
    engine.save()
    return {"ingested": count}


@app.post("/ingest/path")
def ingest_path(request: PathRequest) -> dict:
    if request.reset:
        engine.reset()
    try:
        count = engine.ingest_path(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    engine.save()
    return {"ingested": count}


@app.post("/ingest/upload")
async def ingest_upload(files: list[UploadFile]) -> dict:
    from docmind.documents import SUPPORTED_EXTENSIONS, load_file

    ingested = 0
    processed = 0
    skipped: list[str] = []

    for upload in files:
        name = upload.filename or "upload.txt"
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            skipped.append(name)
            continue

        data = await upload.read()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            document = load_file(tmp_path)
        except (ValueError, RuntimeError):
            skipped.append(name)
            continue
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        document.source = name
        ingested += engine.ingest_documents([document])
        processed += 1

    engine.save()
    return {"ingested": ingested, "files": processed, "skipped": skipped}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    if engine.store.size == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Ingest documents first.")
    answer = engine.query(request.question, top_k=request.top_k)
    return {"answer": answer.text, "sources": [s.to_dict() for s in answer.sources]}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    if engine.store.size == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Ingest documents first.")

    def event_stream():
        buffer = ""
        for token in engine.stream_query(request.question, top_k=request.top_k):
            buffer += token
            if SENTINEL in buffer:
                text, _, rest = buffer.partition(SENTINEL)
                if text:
                    yield _sse({"token": text})
                yield _sse({"sources": json.loads(rest)})
                return
            yield _sse({"token": token})
        yield _sse({"sources": []})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
