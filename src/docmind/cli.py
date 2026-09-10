"""Command-line interface for DocMind."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docmind.config import Settings
from docmind.rag import RAGEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docmind", description="Local RAG assistant over your documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index a file or directory of documents.")
    ingest.add_argument("path", type=Path, help="File or directory to ingest.")
    ingest.add_argument("--reset", action="store_true", help="Clear the existing index first.")

    ask = sub.add_parser("ask", help="Ask a single question.")
    ask.add_argument("question", help="The question to ask.")
    ask.add_argument("-k", "--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    ask.add_argument("--show-sources", action="store_true", help="Print retrieved sources with scores.")

    sub.add_parser("chat", help="Start an interactive chat session.")

    serve = sub.add_parser("serve", help="Run the FastAPI web server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    sub.add_parser("stats", help="Show index statistics.")

    ev = sub.add_parser("eval", help="Score a model against a golden Q&A set.")
    ev.add_argument("--golden", type=Path, default=Path("eval/golden_qa.jsonl"), help="Path to JSONL golden set.")
    ev.add_argument("--model", default=None, help="Override the generation model for this run.")
    ev.add_argument("-k", "--top-k", type=int, default=None, help="Chunks to retrieve per question.")
    ev.add_argument("--min-score", type=float, default=None, help="Minimum similarity score to keep.")
    ev.add_argument("--no-generate", action="store_true", help="Score retrieval only (fast, no LLM calls).")
    return parser


def _cmd_ingest(args: argparse.Namespace, engine: RAGEngine) -> int:
    if args.reset:
        engine.reset()
    count = engine.ingest_path(args.path)
    engine.save()
    print(f"Indexed {count} chunks from '{args.path}'.")
    return 0


def _print_answer(answer) -> None:
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for i, source in enumerate(answer.sources, start=1):
            print(f"  [{i}] {source.source}  (score: {source.score:.3f})")


def _cmd_ask(args: argparse.Namespace, engine: RAGEngine) -> int:
    answer = engine.query(args.question, top_k=args.top_k)
    if args.show_sources:
        _print_answer(answer)
    else:
        print(answer.text)
    return 0


def _cmd_chat(engine: RAGEngine) -> int:
    print("DocMind chat. Type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", ":q"}:
            return 0
        if not question:
            continue
        answer = engine.query(question)
        print(f"\ndocmind> {answer.text}\n")


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("docmind.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_stats(engine: RAGEngine) -> int:
    for key, value in engine.stats().items():
        print(f"{key:>12}: {value}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from docmind.eval import evaluate, format_report, load_golden

    if not args.golden.exists():
        print(f"Golden set not found: {args.golden}")
        return 1

    settings = Settings()
    if args.model:
        settings.llm_model = args.model
    if args.top_k:
        settings.top_k = args.top_k
    if args.min_score is not None:
        settings.min_score = args.min_score

    engine = RAGEngine(settings)
    if engine.store.size == 0:
        print("Index is empty. Run: docmind ingest data/sample_docs --reset")
        return 1

    items = load_golden(args.golden)
    results = evaluate(engine, items, generate=not args.no_generate)
    print(format_report(results, settings.llm_model, settings.llm_options()))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "eval":
        return _cmd_eval(args)

    engine = RAGEngine(Settings())
    if args.command == "ingest":
        return _cmd_ingest(args, engine)
    if args.command == "ask":
        return _cmd_ask(args, engine)
    if args.command == "chat":
        return _cmd_chat(engine)
    if args.command == "stats":
        return _cmd_stats(engine)
    return 1


if __name__ == "__main__":
    sys.exit(main())
