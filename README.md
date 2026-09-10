# DocMind — Local Retrieval-Augmented Generation (RAG) Assistant

DocMind is a privacy-first AI assistant that answers questions **strictly from your own
documents**. It runs entirely on your machine using [Ollama](https://ollama.com) — no
API keys, no cloud calls, nothing leaves your computer. Every answer cites the source
chunks it was grounded in.

Built as a portfolio project to demonstrate the full lifecycle of an LLM application:
document ingestion, chunking, embeddings, vector search, prompt grounding, a REST API,
streaming responses, and a web UI.

---

## Why this project stands out

- **End-to-end, not a notebook.** Real ingestion pipeline + vector index + FastAPI
  service + streaming web UI.
- **Grounded generation.** The model is instructed to answer only from retrieved
  context and to say "I don't know" otherwise — the core anti-hallucination technique.
- **Tuned, non-repetitive output.** Low-temperature sampling (`0.2`), a repetition
  penalty, and a stream-level sentence deduper stop small models from looping or
  doubling their sentences.
- **Fully local & reproducible.** Uses Ollama for both embeddings (`nomic-embed-text`)
  and generation (`qwen3:8b`).
- **Tested.** 14 unit tests run with no network or model dependency (fake embedder/LLM).
- **Pluggable by design.** Swap the embedder, vector store, or LLM behind small interfaces.

---

## Architecture

```
                 ┌─────────────────────────────────────────────────────────┐
                 │                      DocMind                            │
                 └─────────────────────────────────────────────────────────┘

  ingest (CLI / API)
  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌───────────────┐
  │ Documents│──▶│  Chunking │──▶│ Embeddings │──▶│ Vector Store  │──▶ index/vectors.npy
  │ txt/md/  │   │ (overlap) │   │  (Ollama)  │   │ cosine (NumPy)│     index/chunks.jsonl
  │  pdf     │   └───────────┘   └────────────┘   └───────────────┘
  └──────────┘

  query (CLI / API / Web UI)
  ┌──────────┐   ┌────────────┐   ┌───────────────┐   ┌───────────────┐
  │ Question │──▶│  Embed q   │──▶│ Top-K search  │──▶│ Prompt + LLM  │──▶ Answer + sources
  └──────────┘   └────────────┘   │ (cosine)      │   │  (Ollama)     │
                                  └───────────────┘   └───────────────┘
```

**Pipeline stages**

1. **Load** — read `.txt`, `.md`, and `.pdf` files (recursively from a directory).
2. **Chunk** — greedily pack paragraphs into overlapping chunks (default 800 chars,
   120 overlap) so meaning is preserved across boundaries.
3. **Embed** — call Ollama's `/api/embed` with `nomic-embed-text` to get 768-d vectors.
4. **Index** — store normalized vectors in a NumPy cosine-similarity index, persisted to disk.
5. **Retrieve** — embed the question and fetch the Top-K nearest chunks.
6. **Generate** — build a grounded prompt and stream the answer from the local LLM.

---

## Tech stack

`Python` · `FastAPI` · `Uvicorn` · `httpx` · `NumPy` · `pypdf` · `Ollama` ·
`qwen3` · `nomic-embed-text` · `pytest` · `Docker`

---

## Quickstart

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

> Short on VRAM? `llama3.2:3b` is a much lighter drop-in — set `DOCMIND_LLM_MODEL=llama3.2:3b`.

### 2. Install

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Ingest documents and ask

```bash
docmind ingest data/sample_docs --reset
docmind ask "How many vacation days do full-time employees get?" --show-sources
```

### 4. Launch the web app

```bash
docmind serve
# open http://127.0.0.1:8000
```

The web UI lets you **browse and upload files or an entire folder** from your
computer (native file picker), point at a server-side folder path, or paste raw text,
then chat with live streaming answers and clickable source citations. Unsupported files
(e.g. images) are skipped automatically.

---

## Usage

### CLI

```bash
docmind ingest <path> [--reset]          # index a file or directory
docmind ask "<question>" [-k N] [--show-sources]
docmind chat                             # interactive session
docmind serve [--host H] [--port P] [--reload]
docmind stats                            # index statistics
docmind eval [--model M] [--no-generate] # score against eval/golden_qa.jsonl
```

### Comparing models

`docmind eval` runs a golden Q&A set (`eval/golden_qa.jsonl`) and reports retrieval
and answer accuracy, so you can A/B different models or settings objectively:

```bash
docmind eval --model llama3.2:3b
docmind eval --model qwen3:8b
docmind eval --no-generate          # retrieval-only, no LLM calls (fast)
```

Benchmarked on 10 labeled questions (`eval/golden_qa.jsonl`) over the sample docs:

| Model | Retrieval hit@4 | Answer accuracy |
| ----- | --------------- | --------------- |
| `llama3.2:3b` | 100% | 90% |
| `qwen3:8b`    | 100% | **100%** |

### REST API

| Method | Endpoint          | Description                              |
| ------ | ----------------- | ---------------------------------------- |
| GET    | `/health`         | Liveness probe                           |
| GET    | `/stats`          | Chunk / source counts, model names       |
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

---

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

---

## Testing

```bash
pytest -q          # 14 tests, no network/model required
ruff check src     # lint
```

Tests inject a deterministic fake embedder and fake LLM, so the suite is fast and
runs offline while still covering chunking, vector search, persistence, and the RAG flow.

---

## Docker

```bash
docker compose up --build
# API + UI at http://localhost:8000, Ollama at http://localhost:11434
```

After the stack is up, pull models inside the Ollama container:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen3:8b
```

---

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
│   └── static/index.html # streaming web UI
├── tests/                # pytest suite
├── eval/golden_qa.jsonl  # labeled Q&A set for model comparison
├── data/sample_docs/     # sample knowledge base
├── Dockerfile
└── docker-compose.yml
```

---

## Resume bullet points

> **DocMind — Local RAG Question-Answering System** | Python, FastAPI, NumPy, Ollama, Docker
>
> - Built an end-to-end Retrieval-Augmented Generation assistant that ingests PDF/Text/Markdown
>   documents, generates embeddings with `nomic-embed-text`, and answers questions using a local
>   `qwen3:8b` LLM — 100% on-device with no cloud dependencies.
> - Built a golden-set evaluation harness and benchmarked models, improving answer
>   accuracy from 90% (`llama3.2:3b`) to 100% (`qwen3:8b`) on a 10-question labeled set.
> - Designed a NumPy cosine-similarity vector store with overlapping-chunk indexing and disk
>   persistence, retrieving Top-K context to ground every answer with source citations.
> - Exposed the pipeline through a FastAPI service with Server-Sent Events token streaming and a
>   custom web UI; containerized with Docker Compose.
> - Wrote 14 offline unit tests using injected fake models, achieving reproducible CI-friendly
>   coverage of chunking, retrieval, persistence, and generation logic.

---

## License

MIT — see [LICENSE](LICENSE).
