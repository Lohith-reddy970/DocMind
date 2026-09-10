# DocMind — Local RAG Assistant

DocMind answers questions from your own documents (`.txt`, `.md`, `.pdf`) using
[Ollama](https://ollama.com). It runs entirely on your machine — no API keys, no
cloud calls. Every answer cites the source chunks it was grounded in.

## Features

- Ingest files, folders, or pasted text (CLI, REST API, or web UI)
- Overlapping chunking, embeddings via `nomic-embed-text`, NumPy cosine vector store
- Grounded answers from `qwen3:8b` with streaming output and source citations
- CLI, FastAPI server, and a built-in web UI
- Offline unit tests with fake embedder/LLM

## How it works

```
ingest:  documents -> chunk -> embed (Ollama) -> NumPy vector index (index/)
query:   question -> embed -> top-k search -> prompt + LLM -> answer + sources
```

## Quickstart

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/download) running.

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b

python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
```

Ingest the sample docs and ask a question:

```bash
docmind ingest data/sample_docs --reset
docmind ask "How many vacation days do full-time employees get?" --show-sources
```

Or launch the web UI at http://127.0.0.1:8000:

```bash
docmind serve
```

The web UI lets you upload files or a whole folder from your computer, ingest a
server-side folder path, or paste text, then chat with streaming answers and
clickable sources.

> Low on VRAM? Use a smaller model: `DOCMIND_LLM_MODEL=llama3.2:3b`.

## CLI

```bash
docmind ingest <path> [--reset]          # index a file or directory
docmind ask "<question>" [-k N] [--show-sources]
docmind chat                             # interactive session
docmind serve [--host H] [--port P] [--reload]
docmind stats                            # index statistics
docmind eval [--model M] [--no-generate] # score against eval/golden_qa.jsonl
```

`docmind eval` runs the labeled questions in `eval/golden_qa.jsonl` and reports
retrieval and answer accuracy, useful for comparing models:

```bash
docmind eval --model qwen3:8b
docmind eval --no-generate          # retrieval-only, no LLM calls (fast)
```

## REST API

| Method | Endpoint          | Description                              |
| ------ | ----------------- | ---------------------------------------- |
| GET    | `/health`         | Liveness probe                           |
| GET    | `/stats`          | Chunk / source counts, model names       |
| GET    | `/sources`        | Indexed files with chunk counts          |
| POST   | `/ingest/path`    | `{ "path": "...", "reset": false }`      |
| POST   | `/ingest/text`    | `{ "text": "...", "source": "name" }`    |
| POST   | `/ingest/upload`  | Multipart file/folder upload (skips unsupported types) |
| POST   | `/chat`           | `{ "question": "...", "top_k": 4 }`      |
| POST   | `/chat/stream`    | Server-Sent Events token stream          |
| POST   | `/reset`          | Clear the index                          |

Example:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the remote work policy?\"}"
```

## Configuration

All settings are environment variables (see `.env.example`):

| Variable                  | Default                  | Description                     |
| ------------------------- | ------------------------ | ------------------------------- |
| `DOCMIND_OLLAMA_HOST`     | `http://localhost:11434` | Ollama server URL               |
| `DOCMIND_EMBED_MODEL`     | `nomic-embed-text`       | Embedding model                 |
| `DOCMIND_LLM_MODEL`       | `qwen3:8b`               | Generation model                |
| `DOCMIND_CHUNK_SIZE`      | `800`                    | Max characters per chunk        |
| `DOCMIND_CHUNK_OVERLAP`   | `120`                    | Overlap between chunks          |
| `DOCMIND_TOP_K`           | `4`                      | Chunks retrieved per query      |
| `DOCMIND_MIN_SCORE`       | `0.5`                    | Min cosine score to keep a chunk |
| `DOCMIND_TEMPERATURE`     | `0.2`                    | LLM sampling temperature        |
| `DOCMIND_TOP_P`           | `0.9`                    | Nucleus sampling                |
| `DOCMIND_REPEAT_PENALTY`  | `1.2`                    | Discourages repeated text       |
| `DOCMIND_NUM_PREDICT`     | `512`                    | Max tokens generated            |
| `DOCMIND_THINK`           | `false`                  | Disable qwen3 "thinking" mode   |
| `DOCMIND_QUERY_PREFIX`    | auto                     | Query embedding prefix (nomic: `search_query:`) |
| `DOCMIND_DOC_PREFIX`      | auto                     | Document embedding prefix (nomic: `search_document:`) |
| `DOCMIND_INDEX_DIR`       | `index`                  | Where the vector index is saved |

## Testing

```bash
pytest -q          # 39 tests, no network or model required
ruff check src tests
```

Tests inject a deterministic fake embedder and fake LLM, so the suite runs
offline while still covering chunking, vector search, persistence, the API, and
the RAG flow.

## Docker

```bash
docker compose up --build
# API + UI at http://localhost:8000, Ollama at http://localhost:11434
```

Pull models inside the Ollama container after the stack is up:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen3:8b
```

## Project structure

```
docmind/
├── src/docmind/
│   ├── config.py         # env-driven settings
│   ├── documents.py      # txt / md / pdf loaders
│   ├── chunking.py       # overlapping text splitter
│   ├── embeddings.py     # Ollama embedding client
│   ├── vector_store.py   # NumPy cosine vector store + persistence
│   ├── llm.py            # Ollama chat client + prompt builder
│   ├── rag.py            # RAGEngine orchestration
│   ├── api.py            # FastAPI app
│   ├── cli.py            # command-line interface
│   ├── eval.py           # golden-set evaluation harness
│   └── static/index.html # web UI
├── tests/                # pytest suite
├── eval/golden_qa.jsonl  # labeled Q&A set for model comparison
├── data/sample_docs/     # sample knowledge base
├── Dockerfile
└── docker-compose.yml
```

## License

MIT — see [LICENSE](LICENSE).
