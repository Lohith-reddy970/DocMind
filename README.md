# DocMind — Local RAG Assistant

DocMind is a Retrieval-Augmented Generation (RAG) assistant that answers questions
from your own documents (`.txt`, `.md`, `.pdf`). It runs entirely on your machine
through [Ollama](https://ollama.com) — no API keys and no cloud calls. Every answer
is generated only from retrieved chunks and is returned with the sources that
grounded it.

- [Features](#features)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Network behavior](#network-behavior)
- [Web UI](#web-ui)
- [CLI](#cli)
- [REST API](#rest-api)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Docker](#docker)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Data and privacy](#data-and-privacy)
- [License](#license)

## Features

- Ingest files or whole folders from the CLI, the REST API, or a browser upload
  (native file/folder picker)
- Overlapping paragraph-aware chunking
- Embeddings via Ollama (`nomic-embed-text`, 768-dim) with automatic task prefixes
- Dependency-light NumPy cosine vector store persisted to disk
- Grounded generation via Ollama (`qwen3:8b` by default) with:
  - a system prompt that forbids inventing facts
  - a minimum similarity threshold so off-topic questions return no context
  - sampling tuned to avoid repetition, plus sentence-level deduplication
  - token streaming over Server-Sent Events
- FastAPI server with a built-in web UI (light/dark theme, source citations,
  upload notifications)
- Local-only egress guard plus a `/network` audit of every destination contacted
- Golden-set evaluation harness for comparing models and retrieval settings
- 47 offline unit tests (no network or models required)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- ~6 GB of free disk space for the default models (more GPU VRAM if you want
  faster generation)

Pull the default models:

```bash
ollama pull nomic-embed-text   # embeddings (~274 MB)
ollama pull qwen3:8b           # generation (~5.2 GB)
```

Any Ollama chat model works — see [Configuration](#configuration) for using a
lighter one such as `llama3.2:3b`.

## Quickstart

```bash
git clone https://github.com/Lohith-reddy970/DocMind.git
cd DocMind

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

Index the bundled sample documents and ask a question:

```bash
docmind ingest data/sample_docs --reset
docmind ask "How many vacation days do full-time employees get?" --show-sources
```

Launch the web UI:

```bash
docmind serve
# open http://127.0.0.1:8000
```

If you prefer not to install the package, run it from the source tree instead:

```bash
# Windows PowerShell
$env:PYTHONPATH="src"; python -m docmind ingest data/sample_docs --reset
$env:PYTHONPATH="src"; python -m docmind serve

# macOS/Linux
PYTHONPATH=src python -m docmind serve
```

## How it works

```
ingest
  documents ──▶ chunking ──▶ embeddings ──▶ vector store ──▶ index/vectors.npy
  txt/md/pdf    (overlap)     (Ollama)       (NumPy cosine)   index/chunks.jsonl

query
  question ──▶ embed query ──▶ top-k search ──▶ prompt + LLM ──▶ answer + sources
                                (cosine)         (Ollama)
```

### 1. Loading (`documents.py`)

`load_file` reads a single file and `load_path` walks a directory recursively.
Supported extensions are `.txt`, `.md`, `.markdown`, and `.pdf`. PDF text is
extracted with `pypdf`, page by page. Each document records its path as the
`source` label, which is what later appears in citations.

### 2. Chunking (`chunking.py`)

Documents are split by blank lines into paragraphs, which are greedily packed
into chunks of at most `DOCMIND_CHUNK_SIZE` characters (default `800`).
Paragraphs longer than the limit are hard-wrapped. Consecutive chunks share
`DOCMIND_CHUNK_OVERLAP` characters (default `120`) so meaning is not lost at
boundaries. Chunks shorter than 40 characters are dropped (unless the whole
document is tiny). Every chunk stores `source`, `chunk_index`, and metadata such
as the original file type.

### 3. Embeddings (`embeddings.py`)

Each chunk is embedded with Ollama's `/api/embed` endpoint. For Nomic models the
document text is prefixed with `search_document: ` and queries with
`search_query: `, which is how those models were trained; other models use no
prefix by default. Prefixes can be overridden or disabled with
`DOCMIND_QUERY_PREFIX` and `DOCMIND_DOC_PREFIX`. `nomic-embed-text` produces
768-dimensional vectors.

### 4. Vector store (`vector_store.py`)

Vectors are normalized on insert, so a dot product is exactly cosine similarity.
A search computes scores for every stored chunk, sorts them, and returns the top
`k`. The store persists to two files: `index/vectors.npy` (a float32 matrix) and
`index/chunks.jsonl` (one JSON chunk per line). Missing files simply load as an
empty index.

### 5. Retrieval (`rag.py`)

The question is embedded with the query prefix, the top `DOCMIND_TOP_K` chunks
(default `4`) are retrieved, and any chunk scoring below `DOCMIND_MIN_SCORE`
(default `0.5`) is discarded. When nothing passes the threshold the prompt is
sent with an explicit "(no relevant context found)" block, so off-topic
questions are refused rather than answered from model memory.

### 6. Generation (`llm.py`)

Retrieved chunks are assembled into a prompt of numbered `[Source N: path]`
blocks plus the question. The system prompt instructs the model to use only the
context, to refuse opinions or predictions, to stay concise, and never to repeat
a sentence. Decoding options are tuned for small models:

| Option             | Default | Purpose                          |
| ------------------ | ------- | -------------------------------- |
| `temperature`      | `0.2`   | Less random sampling             |
| `top_p`            | `0.9`   | Nucleus sampling                 |
| `repeat_penalty`   | `1.2`   | Discourages repeated text        |
| `repeat_last_n`    | `256`   | Window for the repetition penalty |
| `num_predict`      | `512`   | Maximum generated tokens         |
| `num_ctx`          | `4096`  | Context window                   |

A sentence-level deduper runs on both the complete answer and the token stream,
collapsing sentences the model emits back-to-back. For thinking-capable models
(such as Qwen3), `DOCMIND_THINK=false` disables reasoning mode for faster and
more tightly grounded answers.

Streaming works by yielding tokens as they arrive; when generation finishes, the
engine appends a sentinel line with the sources, which the API converts into a
final Server-Sent Event.

## Network behavior

DocMind has exactly one kind of outbound connection: HTTP to the configured
Ollama server, used for embeddings and chat generation. There is no telemetry,
no update check, and no other client, and document text is never sent anywhere
else.

Every request made by the embedding and generation clients passes through a
request hook (`netguard.py`) that:

1. **Records** the destination host, purpose (`embeddings` or `generation`), and
   request count in an in-process tally.
2. **Blocks** the request before it is sent when `DOCMIND_LOCAL_ONLY=true` (the
   default) and the host is not local. Local means loopback (`localhost`,
   `127.0.0.1`, `::1`), private ranges (`10.x`, `172.16-31.x`, `192.168.x`),
   link-local addresses, or single-label/mDNS names such as the `ollama` service
   in Docker.

The tally is exposed at `GET /network` and shown in the web UI's **Network**
panel, so you can see every host that was contacted at any point:

```jsonc
{
  "local_only": true,
  "ollama_host": "http://localhost:11434",
  "host_is_local": true,
  "blocked_requests": 0,
  "endpoints": [
    { "host": "localhost", "purpose": "embeddings", "requests": 12, "last_at": 1757500000.0 },
    { "host": "localhost", "purpose": "generation", "requests": 4, "last_at": 1757500003.0 }
  ],
  "message": "Only the configured Ollama server is contacted. No telemetry, analytics, or cloud APIs."
}
```

`DOCMIND_LOCAL_ONLY=false` disables the block (for example, if you intentionally
point `DOCMIND_OLLAMA_HOST` at a remote server you control) but the audit keeps
recording everything.

## Web UI

`docmind serve` hosts a single-page app at `http://127.0.0.1:8000`:

- **Overview** — chunk count, file count, and the active models
- **Network** — the local-only status plus every host contacted so far
- **Add documents** in three tabs:
  - *Upload* — native file picker or folder picker; unsupported files are
    skipped and reported; successful uploads raise a notification showing how
    many files and chunks were indexed
  - *Folder path* — ingest a path on the server (e.g. `data/sample_docs`)
  - *Text* — paste raw text under a chosen source name
- **Knowledge base** — every indexed source with its chunk count, plus a
  "Clear all" button
- **Chat** — streaming answers with clickable source chips; clicking a chip
  expands the exact retrieved context
- **Theme toggle** in the header — defaults to your OS preference and remembers
  your choice in `localStorage`

## CLI

```
docmind ingest <path> [--reset]            index a file or directory
docmind ask "<question>" [-k N] [--show-sources]
docmind chat                               interactive session (exit to quit)
docmind serve [--host H] [--port P] [--reload]
docmind stats                              index statistics
docmind eval [--golden PATH] [--model M] [-k N] [--min-score F] [--no-generate]
```

Examples:

```bash
docmind ingest data/sample_docs --reset
docmind ask "What is the remote work stipend?" --show-sources
docmind ask "How does RAG work?" -k 6
docmind stats
```

## REST API

`docmind serve` exposes:

| Method | Endpoint          | Description                                          |
| ------ | ----------------- | ---------------------------------------------------- |
| GET    | `/health`         | Liveness probe — `{"status": "ok"}`                  |
| GET    | `/stats`          | Chunk/file counts and model names                    |
| GET    | `/sources`        | Indexed sources with per-file chunk counts           |
| GET    | `/network`        | Outbound destinations and local-only status          |
| POST   | `/ingest/path`    | `{ "path": "...", "reset": false }`                  |
| POST   | `/ingest/text`    | `{ "text": "...", "source": "name" }`                |
| POST   | `/ingest/upload`  | Multipart upload; unsupported files are skipped      |
| POST   | `/chat`           | `{ "question": "...", "top_k": 4 }`                  |
| POST   | `/chat/stream`    | Same body; Server-Sent Events token stream           |
| POST   | `/reset`          | Clear the index                                      |

Example requests:

```bash
curl http://127.0.0.1:8000/stats

curl -X POST http://127.0.0.1:8000/ingest/path \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"data/sample_docs\"}"

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the remote work policy?\"}"
```

Response shapes:

```jsonc
// GET /stats
{ "chunks": 10, "dim": 768, "sources": 3, "embed_model": "nomic-embed-text", "llm_model": "qwen3:8b" }

// GET /sources
{ "sources": [ { "name": "data/sample_docs/handbook.md", "chunks": 3 } ] }

// GET /network
{ "local_only": true, "ollama_host": "http://localhost:11434", "host_is_local": true,
  "blocked_requests": 0,
  "endpoints": [ { "host": "localhost", "purpose": "generation", "requests": 7, "last_at": 1757500000.0 } ],
  "message": "Only the configured Ollama server is contacted. No telemetry, analytics, or cloud APIs." }

// POST /chat
{ "answer": "...", "sources": [ { "source": "handbook.md", "score": 0.71, "text": "..." } ] }

// POST /ingest/upload
{ "ingested": 12, "files": 2, "skipped": ["photo.png"] }
```

`/chat/stream` emits `data: {"token": "..."}` events and finishes with
`data: {"sources": [...]}`.

## Configuration

All settings are environment variables prefixed with `DOCMIND_` (see
`.env.example`); invalid numeric values fall back to the defaults.

| Variable                  | Default                  | Description                                      |
| ------------------------- | ------------------------ | ------------------------------------------------ |
| `DOCMIND_OLLAMA_HOST`     | `http://localhost:11434` | Ollama server URL                                |
| `DOCMIND_EMBED_MODEL`     | `nomic-embed-text`       | Embedding model                                  |
| `DOCMIND_LLM_MODEL`       | `qwen3:8b`               | Generation model                                 |
| `DOCMIND_CHUNK_SIZE`      | `800`                    | Max characters per chunk                         |
| `DOCMIND_CHUNK_OVERLAP`   | `120`                    | Overlap between consecutive chunks               |
| `DOCMIND_TOP_K`           | `4`                      | Chunks retrieved per question                    |
| `DOCMIND_MIN_SCORE`       | `0.5`                    | Minimum cosine score for a chunk to be used      |
| `DOCMIND_TEMPERATURE`     | `0.2`                    | Sampling temperature                             |
| `DOCMIND_TOP_P`           | `0.9`                    | Nucleus sampling                                 |
| `DOCMIND_REPEAT_PENALTY`  | `1.2`                    | Repetition penalty                               |
| `DOCMIND_NUM_PREDICT`     | `512`                    | Maximum generated tokens                         |
| `DOCMIND_THINK`           | `false`                  | Toggle reasoning mode (`true`/`false`)           |
| `DOCMIND_QUERY_PREFIX`    | auto                     | Query embedding prefix (Nomic: `search_query: `) |
| `DOCMIND_DOC_PREFIX`      | auto                     | Document embedding prefix (Nomic: `search_document: `) |
| `DOCMIND_INDEX_DIR`       | `index`                  | Directory for the persisted index                |
| `DOCMIND_REQUEST_TIMEOUT` | `120`                    | HTTP timeout in seconds for Ollama calls         |
| `DOCMIND_LOCAL_ONLY`      | `true`                   | Block requests to non-local Ollama hosts         |

Example — a lighter footprint:

```bash
# Windows PowerShell
$env:DOCMIND_LLM_MODEL="llama3.2:3b"; python -m docmind serve

# macOS/Linux
DOCMIND_LLM_MODEL=llama3.2:3b docmind serve
```

## Evaluation

`docmind eval` scores retrieval and answer quality against a labeled JSONL set.
The bundled `eval/golden_qa.jsonl` contains ten questions over the sample
documents:

```json
{"question": "How many vacation days do full-time employees get?", "keywords": ["20 days", "twenty days"], "source": "handbook.md"}
```

- `question` — the query
- `keywords` — any of these substrings in the retrieved context or answer counts as a hit
- `source` — optional expected file; checks whether it was retrieved

```bash
docmind eval --model qwen3:8b        # full run (retrieval + generation)
docmind eval --no-generate           # retrieval only, no LLM calls
docmind eval -k 6 --min-score 0.4    # try different retrieval settings
```

The report lists per-question hits and three aggregates: retrieval hit@k,
expected source hit rate, and answer hit rate. Retrieval-only runs are fast and
useful for tuning `top_k`/`min_score`; full runs let you compare models.

## Testing

```bash
pytest -q          # 47 tests, no network or models required
ruff check src tests
```

The suite injects deterministic fake embedder and LLM implementations, so it
runs offline while still covering loading, chunking, vector search and
persistence, prompt building, streaming dedupe, the egress guard, the
evaluation logic, and the FastAPI endpoints. `coverage` can be produced with
`pytest --cov=docmind`.

## Docker

A `docker-compose.yml` runs DocMind alongside an Ollama container:

```bash
docker compose up --build
```

- Web UI + API: `http://localhost:8000`
- Ollama: `http://localhost:11434`

Pull models inside the Ollama container after the stack is up:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen3:8b
```

The index is persisted in `./index` on the host; model weights live in the named
volume `ollama-data`. The compose file sets `DOCMIND_OLLAMA_HOST` to the
service name `ollama` automatically.

## Project structure

```
docmind/
├── src/docmind/
│   ├── config.py          # env-driven Settings and Ollama options
│   ├── documents.py       # txt / md / pdf loaders
│   ├── chunking.py        # paragraph-aware overlapping splitter
│   ├── embeddings.py       # Ollama embedding client and task prefixes
│   ├── netguard.py         # outbound request log and local-only egress guard
│   ├── vector_store.py     # NumPy cosine store with disk persistence
│   ├── llm.py             # Ollama chat client, prompt, reply dedupe
│   ├── rag.py             # RAGEngine orchestration (ingest/retrieve/query)
│   ├── api.py             # FastAPI app and SSE streaming
│   ├── cli.py             # command-line interface
│   ├── eval.py            # golden-set evaluation harness
│   └── static/index.html  # web UI (single file, no build step)
├── tests/                 # pytest suite with fake embedder/LLM
├── eval/golden_qa.jsonl   # labeled evaluation set
├── data/sample_docs/      # sample knowledge base
├── Dockerfile
├── docker-compose.yml
├── requirements.txt       # runtime dependencies
├── requirements-dev.txt   # test/lint dependencies
└── .env.example           # configuration template
```

## Troubleshooting

**"Failed to reach Ollama ..."** — make sure Ollama is running (`ollama list`).
If it listens on another host or port, set `DOCMIND_OLLAMA_HOST`.

**Model not found (404)** — pull it: `ollama pull nomic-embed-text`,
`ollama pull qwen3:8b`. The model names must match exactly what `ollama list`
shows.

**Answers say the context is empty** — the knowledge base has no chunks or
nothing passed `DOCMIND_MIN_SCORE`. Ingest documents first
(`docmind ingest <path> --reset`) and check `docmind stats`. Lowering
`DOCMIND_MIN_SCORE` makes retrieval more permissive.

**`docmind` is not recognized** — either install the package
(`pip install -e .`) or use `python -m docmind ...` with `PYTHONPATH=src`.

**Slow generation or out-of-memory errors on the GPU** — use a smaller model,
e.g. `DOCMIND_LLM_MODEL=llama3.2:3b`, or reduce `DOCMIND_NUM_PREDICT`.

**PDF produces no text** — `pypdf` extracts embedded text only. Scanned or
image-only PDFs need OCR before ingestion.

**Port 8000 already in use** — start with another port:
`docmind serve --port 8010`.

**Answers are cut off** — increase `DOCMIND_NUM_PREDICT` (default 512 tokens).

## Data and privacy

- Documents are read locally and embedded by your own Ollama instance; the text
  never leaves the machine.
- All outbound traffic goes through the local-only egress guard (see
  [Network behavior](#network-behavior)); `GET /network` or the UI's Network
  panel shows exactly which hosts were contacted, and the guard blocks any
  non-local destination by default.
- Browser uploads are written to a temporary file, embedded, then deleted;
  nothing is copied into the repository.
- The only persistent artifacts are `index/vectors.npy` and
  `index/chunks.jsonl`, both of which are git-ignored. Delete them (or use
  "Clear all" in the UI / `POST /reset`) to remove the knowledge base.

## License

MIT — see [LICENSE](LICENSE).
